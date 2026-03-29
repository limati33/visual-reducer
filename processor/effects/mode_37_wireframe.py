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


def _apply_camera_transform(img, zoom=1.0, dx=0.0, dy=0.0, angle=0.0):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    m = cv2.getRotationMatrix2D((cx, cy), float(angle), float(zoom))
    m[0, 2] += float(dx)
    m[1, 2] += float(dy)

    return cv2.warpAffine(
        img,
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


def _save_gif_from_frames(frames, gif_path, fps=12, input_is_bgr=True):
    if not _HAVE_PIL or not frames:
        return False

    pil_frames = []
    for fr in frames:
        if fr is None:
            continue

        if fr.dtype != np.uint8:
            fr = np.clip(fr, 0, 255).astype(np.uint8)

        if input_is_bgr:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

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
    region_thickness=2
):
    ih, iw = img.shape[:2]

    if w is not None and h is not None and (iw != int(w) or ih != int(h)):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]

    rng = np.random.default_rng(seed)

    base = img.copy()
    smooth = cv2.bilateralFilter(base, d=11, sigmaColor=95, sigmaSpace=95)
    smooth = cv2.GaussianBlur(smooth, (0, 0), 0.8)

    quantized = _posterize_kmeans(smooth, k=kmeans_k, seed=seed)
    result = cv2.addWeighted(quantized, base_alpha, base, 1.0 - base_alpha, 0)

    b, g, r = cv2.split(result)
    b = _shift_channel(b, -1, 0)
    r = _shift_channel(r, 1, 0)
    result = cv2.merge([b, g, r])

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

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=65, sigmaSpace=65)

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

    yy, xx = np.indices((ih, iw), dtype=np.float32)
    cx, cy = iw / 2.0, ih / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / max(1e-6, dist.max())

    vignette = 1.0 - 0.18 * (dist ** 1.6)
    result = np.clip(result.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

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
        region_thickness=region_thickness
    )

    if make_gif and out_dir and base_name and not _is_video_source(source_path):
        frames = []
        total = max(18, int(gif_frames) * 2)

        for i in range(total):
            t = i / total
            phase = 2.0 * math.pi * t

            zoom = 1.0 + 0.012 * math.sin(phase + math.pi * 0.5)
            dx = 0.8 * math.sin(phase * 0.5 + 1.1)
            dy = 7.0 * math.sin(phase) + 2.0 * math.sin(phase * 2.0 + 0.7)
            angle = 0.12 * math.sin(phase * 0.5 + 0.3)

            camera_img = _apply_camera_transform(
                img,
                zoom=zoom,
                dx=dx,
                dy=dy,
                angle=angle
            )

            edge1 = edge_thresh1 + 4.0 * math.sin(phase + 0.2)
            edge2 = edge_thresh2 + 6.0 * math.sin(phase + 1.4)
            line_len = max(10, int(round(min_line_length + 3.0 * math.sin(phase + 0.8))))
            poly = max(0.002, poly_epsilon * (1.0 + 0.18 * math.sin(phase + 2.1)))

            frame = _build_wireframe_frame(
                camera_img,
                w=w,
                h=h,
                seed=seed + i * 17,
                edge_thresh1=int(round(edge1)),
                edge_thresh2=int(round(edge2)),
                min_line_length=line_len,
                poly_epsilon=poly,
                thickness=thickness,
                kmeans_k=kmeans_k,
                base_alpha=base_alpha,
                region_min_area_ratio=region_min_area_ratio,
                region_thickness=region_thickness
            )

            frames.append(frame)

        gif_path = Path(out_dir) / f"{Path(base_name).stem}_wireframe.gif"
        _save_gif_from_frames(frames, gif_path, fps=18, input_is_bgr=True)

    return result
