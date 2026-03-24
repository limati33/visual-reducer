# processor/effects/mode_22_graph.py
import cv2
import numpy as np
from pathlib import Path
import warnings

def _fit_curve(xs, ys, max_degree=4):
    if len(xs) < 8:
        return None

    # сортировка и удаление дубликатов по X
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    keep = np.insert(np.diff(xs) > 0.5, 0, True)
    xs = xs[keep]
    ys = ys[keep]

    if len(xs) < 8:
        return None

    deg = min(max_degree, max(1, len(xs) // 20))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(xs, ys, deg)
        return np.poly1d(coeffs)
    except Exception:
        return None


def apply_graph(img, w, h, out_dir, base_name):

    h_img, w_img = img.shape[:2]

    # Resize при необходимости
    if w and h and (w_img != w or h_img != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        h_img, w_img = img.shape[:2]

    # RGB -> gray (важно: в pipeline картинка приходит RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Контраст + инверсия
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    inverted = 255 - enhanced

    # Эдges
    edges1 = cv2.Canny(enhanced, 45, 130)
    edges2 = cv2.Canny(enhanced, 20, 80)
    edges = cv2.addWeighted(edges1, 0.7, edges2, 0.3, 0)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    # База
    base = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    tint = np.full_like(base, (190, 220, 200))  # мягкий техничный оттенок
    result = cv2.addWeighted(base, 0.86, tint, 0.14, 0)

    # Линии
    result = cv2.subtract(result, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) // 2)

    equations = []
    curve_count = 0

    # Несколько уровней яркости
    levels = np.linspace(60, 210, 6).astype(np.int32)

    for thresh in levels:
        _, bw = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 40:
                continue

            pts = cnt.reshape(-1, 2)
            xs = pts[:, 0].astype(np.float64)
            ys = pts[:, 1].astype(np.float64)

            poly = _fit_curve(xs, ys, max_degree=4)
            if poly is None:
                continue

            x_min = int(np.clip(xs.min(), 0, w_img - 1))
            x_max = int(np.clip(xs.max(), 0, w_img - 1))
            if x_max - x_min < 10:
                continue

            xs_eval = np.linspace(x_min, x_max, max(120, x_max - x_min + 1))
            ys_eval = poly(xs_eval)

            pts_draw = np.column_stack([xs_eval, ys_eval]).astype(np.int32)
            pts_draw[:, 0] = np.clip(pts_draw[:, 0], 0, w_img - 1)
            pts_draw[:, 1] = np.clip(pts_draw[:, 1], 0, h_img - 1)

            if len(pts_draw) < 2:
                continue

            cv2.polylines(
                result,
                [pts_draw.reshape(-1, 1, 2)],
                isClosed=False,
                color=(255, 110, 60),
                thickness=1,
                lineType=cv2.LINE_AA
            )

            coeffs = np.poly1d(poly).coeffs
            coeffs_str = " + ".join(
                f"{c:.4e}*x^{len(coeffs)-i-1}" if len(coeffs)-i-1 > 1 else
                f"{c:.4e}*x" if len(coeffs)-i-1 == 1 else
                f"{c:.4e}"
                for i, c in enumerate(coeffs)
            )
            equations.append(f"level={int(thresh)} contour={curve_count}: y = {coeffs_str}")
            curve_count += 1

    # Fallback, если кривых не нашлось
    if curve_count == 0:
        profile = np.mean(enhanced, axis=0)
        profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
        ys_profile = (profile * (h_img - 1)).astype(np.int32)
        xs_profile = np.arange(w_img, dtype=np.int32)

        pts_draw = np.column_stack([xs_profile, ys_profile]).astype(np.int32)
        cv2.polylines(
            result,
            [pts_draw.reshape(-1, 1, 2)],
            isClosed=False,
            color=(255, 110, 60),
            thickness=1,
            lineType=cv2.LINE_AA
        )
        equations.append("fallback_profile: y = mean_intensity_profile(x)")

    # Лёгкий общий контраст
    result = cv2.convertScaleAbs(result, alpha=1.03, beta=2)

    # Сохранение
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        name_stem = Path(base_name).stem if base_name else "graph"
        txt_path = Path(out_dir) / f"graph_{name_stem}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {base_name}\n")
            f.write(f"Image size: {w_img}x{h_img}\n")
            f.write(f"Levels: {len(levels)}\n")
            f.write(f"Contours fitted: {curve_count}\n\n")
            for line in equations:
                f.write(line + "\n")

    return result
