# processor/effects/mode_37_wireframe.py
import cv2
import numpy as np
import math
from pathlib import Path

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _is_video_source(source_path):
    if not source_path:
        return False
    return Path(str(source_path)).suffix.lower() in VIDEO_EXTS


def _find_contours(mask, mode, method):
    found = cv2.findContours(mask, mode, method)
    return found[0] if len(found) == 2 else found[1]


def _shift_channel(channel, dx, dy):
    h, w = channel.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        channel,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )


def _posterize_kmeans(img, k=6, seed=42):
    ih, iw = img.shape[:2]
    work = img

    if iw * ih > 800 * 800:
        work = cv2.resize(
            img,
            (max(1, iw // 2), max(1, ih // 2)),
            interpolation=cv2.INTER_AREA
        )

    Z = work.reshape((-1, 3)).astype(np.float32)

    K = max(2, int(k))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        12,
        1.0
    )

    cv2.setRNGSeed(int(seed))
    _, labels, centers = cv2.kmeans(
        Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(work.shape)

    if quantized.shape[:2] != (ih, iw):
        quantized = cv2.resize(quantized, (iw, ih), interpolation=cv2.INTER_NEAREST)

    return quantized


def _save_gif_from_frames(frames_rgb, gif_path, fps=8):
    if not _HAVE_PIL or not frames_rgb:
        return False

    pil_frames = []
    for fr in frames_rgb:
        if fr is None:
            continue

        if fr.dtype != np.uint8:
            fr = np.clip(fr, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(fr).convert("P", palette=Image.ADAPTIVE, colors=256)
        pil_frames.append(pil_img)

    if len(pil_frames) < 2:
        return False

    duration_ms = max(10, int(1000 / max(1, fps)))

    pil_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2
    )
    return True


def _offset_contour(cnt, dx, dy, w, h):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    pts[:, 0] += float(dx)
    pts[:, 1] += float(dy)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
    return pts.astype(np.int32).reshape(-1, 1, 2)


def _build_wireframe_frame(
    img,
    w,
    h,
    seed=42,
    edge_thresh1=45,
    edge_thresh2=130,
    min_line_length=30,
    poly_epsilon=0.010,
    thickness=2,
    kmeans_k=6,
    base_alpha=0.78,
    region_min_area_ratio=130,
    region_thickness=2,
    animate_lines=False,
    frame_phase=0.0
):
    ih, iw = img.shape[:2]

    if w is not None and h is not None and (iw != int(w) or ih != int(h)):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]

    rng = np.random.default_rng(seed)

    # База
    base = img.copy()
    smooth = cv2.bilateralFilter(base, d=11, sigmaColor=95, sigmaSpace=95)
    smooth = cv2.GaussianBlur(smooth, (0, 0), 0.8)

    quantized = _posterize_kmeans(smooth, k=kmeans_k, seed=seed)
    result = cv2.addWeighted(quantized, base_alpha, base, 1.0 - base_alpha, 0)

    # Лёгкое смещение каналов
    b, g, r = cv2.split(result)
    b = _shift_channel(b, -1, 0)
    r = _shift_channel(r, 1, 0)
    result = cv2.merge([b, g, r])

    # Крупные области
    unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
    area_min = max(50, (iw * ih) // int(region_min_area_ratio))

    accent_palette = np.array([
        [255, 255, 255],
        [30, 30, 30],
        [255, 80, 180],
        [80, 220, 255],
        [255, 220, 80],
        [180, 100, 255],
        [80, 255, 170],
        [60, 120, 255],
    ], dtype=np.uint8)

    for idx, col in enumerate(unique_colors):
        mask = cv2.inRange(quantized, col, col)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours = _find_contours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        draw_color = accent_palette[(idx + seed) % len(accent_palette)].tolist()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_min:
                continue

            arc = cv2.arcLength(cnt, True)
            if arc < max(20, iw // 8):
                continue

            cv2.drawContours(
                result,
                [cnt],
                -1,
                (20, 20, 20),
                region_thickness + 2,
                lineType=cv2.LINE_AA
            )

            cv2.drawContours(
                result,
                [cnt],
                -1,
                draw_color,
                region_thickness,
                lineType=cv2.LINE_AA
            )

    # Wireframe-линии
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=65, sigmaSpace=65)

    # Для GIF чуть “дышим” порогами, чтобы линии менялись
    if animate_lines:
        edge_thresh1 = int(round(edge_thresh1 + 4.0 * math.sin(frame_phase * 2.1 + 0.7)))
        edge_thresh2 = int(round(edge_thresh2 + 7.0 * math.cos(frame_phase * 1.7 + 1.2)))
        min_line_length = max(10, int(round(min_line_length + 2.5 * math.sin(frame_phase * 2.7 + 0.4))))
        poly_epsilon = max(0.002, poly_epsilon * (1.0 + 0.14 * math.sin(frame_phase * 2.4 + 0.9)))

    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    contours = _find_contours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    wire_palette = np.array([
        [255, 255, 255],
        [50, 50, 50],
        [255, 100, 200],
        [100, 240, 255],
        [255, 210, 100],
        [160, 120, 255],
    ], dtype=np.uint8)

    for idx, cnt in enumerate(contours):
        arc = cv2.arcLength(cnt, False)
        if arc < min_line_length:
            continue

        epsilon = poly_epsilon * max(1.0, cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, epsilon, closed=False)

        if len(approx) < 2:
            continue

        line_color = wire_palette[(idx + seed) % len(wire_palette)].tolist()

        # Основная линия
        cv2.polylines(
            result,
            [approx],
            isClosed=False,
            color=(10, 10, 10),
            thickness=thickness + 2,
            lineType=cv2.LINE_AA
        )

        cv2.polylines(
            result,
            [approx],
            isClosed=False,
            color=line_color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

        # Для GIF: дополнительный дрожащий слой линий, чтобы они реально "жили"
        if animate_lines:
            dx = int(round(math.sin(frame_phase * 3.2 + idx * 0.19) * 1.5))
            dy = int(round(math.cos(frame_phase * 2.8 + idx * 0.13) * 1.5))
            jittered = _offset_contour(approx, dx, dy, iw, ih)

            alt_color = wire_palette[(idx + seed + 2) % len(wire_palette)].tolist()
            alt_thickness = max(1, thickness - 1)

            cv2.polylines(
                result,
                [jittered],
                isClosed=False,
                color=alt_color,
                thickness=alt_thickness,
                lineType=cv2.LINE_AA
            )

            # Микро-искра/подсветка, чтобы было ближе к комикс-анимации
            if (idx + seed) % 5 == 0:
                cv2.polylines(
                    result,
                    [jittered],
                    isClosed=False,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

    # Зерно и полутон
    h_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    noise = rng.normal(0, 7, (ih, iw, 3)).astype(np.int16)
    result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    shadow = 255 - h_gray
    dot_mask = np.zeros((ih, iw), dtype=np.uint8)
    step = max(8, min(iw, ih) // 120)

    for y in range(step // 2, ih, step):
        for x in range(step // 2, iw, step):
            t = int(shadow[y, x])
            if t < 50:
                continue
            radius = max(1, (t // 65))
            cv2.circle(dot_mask, (x, y), radius, 255, -1, lineType=cv2.LINE_AA)

    result = np.where(dot_mask[..., None] > 0, (result * 0.90).astype(np.uint8), result)

    # Виньетка
    yy, xx = np.indices((ih, iw), dtype=np.float32)
    cx, cy = iw / 2.0, ih / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / max(1e-6, dist.max())

    vignette = 1.0 - 0.18 * (dist ** 1.6)
    result = np.clip(result.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

    # Возвращаем RGB, чтобы обычное сохранение через PIL было с правильными цветами.
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def apply_wireframe(
    img,
    w=None,
    h=None,
    out_dir=None,
    base_name=None,
    edge_thresh1=45,
    edge_thresh2=130,
    min_line_length=30,
    poly_epsilon=0.010,
    thickness=2,
    kmeans_k=6,
    base_alpha=0.78,
    region_min_area_ratio=130,
    region_thickness=2,
    seed=42,
    source_path=None,
    make_gif=None,
    gif_frames=8,
):
    if make_gif is None:
        make_gif = not _is_video_source(source_path)

    result = _build_wireframe_frame(
        img,
        w=w,
        h=h,
        seed=seed,
        edge_thresh1=edge_thresh1,
        edge_thresh2=edge_thresh2,
        min_line_length=min_line_length,
        poly_epsilon=poly_epsilon,
        thickness=thickness,
        kmeans_k=kmeans_k,
        base_alpha=base_alpha,
        region_min_area_ratio=region_min_area_ratio,
        region_thickness=region_thickness,
        animate_lines=False,
        frame_phase=0.0
    )

    # # GIF: статичная картинка, но линии меняются от кадра к кадру
    # if make_gif and out_dir and base_name and not _is_video_source(source_path):
    #     frames = []
    #     total = max(8, int(gif_frames) * 2)

    #     for i in range(total):
    #         phase = (2.0 * math.pi * i) / max(1, total)

    #         frame = _build_wireframe_frame(
    #             img,
    #             w=w,
    #             h=h,
    #             seed=seed + i * 17,
    #             edge_thresh1=edge_thresh1,
    #             edge_thresh2=edge_thresh2,
    #             min_line_length=min_line_length,
    #             poly_epsilon=poly_epsilon,
    #             thickness=thickness,
    #             kmeans_k=kmeans_k,
    #             base_alpha=base_alpha,
    #             region_min_area_ratio=region_min_area_ratio,
    #             region_thickness=region_thickness,
    #             animate_lines=True,
    #             frame_phase=phase
    #         )

    #         frames.append(frame)

    #     gif_path = Path(out_dir) / f"{Path(base_name).stem}_wireframe.gif"
    #     _save_gif_from_frames(frames, gif_path, fps=10)

    return result
