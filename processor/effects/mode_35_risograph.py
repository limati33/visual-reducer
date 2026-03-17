# processor/effects/mode_35_risograph.py
import numpy as np
import cv2
import colorsys
import math
from pathlib import Path

# --- конфигурация "базовых" RISO оттенков (h в 0..1) ---
BASE_INK_HUES = [
    0.95,  # pink
    0.02,  # red
    0.08,  # orange
    0.13,  # yellow
    0.33,  # green
    0.48,  # teal
    0.61,  # blue
    0.75,  # purple
]

def _hue_distance(a, b):
    d = abs(a - b)
    return min(d, 1 - d)

def _rgb_to_hsv_array(rgb_unit):
    """rgb_unit: Nx3 float in 0..1 -> returns Nx3 hsv (h 0..1, s 0..1, v 0..1)"""
    out = np.zeros_like(rgb_unit)
    for i, (r, g, b) in enumerate(rgb_unit):
        out[i] = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    return out

def _hsv_to_rgb_array(hsv_arr):
    out = np.zeros_like(hsv_arr)
    for i, (h, s, v) in enumerate(hsv_arr):
        out[i] = colorsys.hsv_to_rgb(float(h), float(s), float(v))
    return out

def _make_bayer8_threshold_map():
    """Возвращает 8x8 Bayer matrix, scaled 0..255"""
    b = np.array([
        [0,48,12,60,3,51,15,63],
        [32,16,44,28,35,19,47,31],
        [8,56,4,52,11,59,7,55],
        [40,24,36,20,43,27,39,23],
        [2,50,14,62,1,49,13,61],
        [34,18,46,30,33,17,45,29],
        [10,58,6,54,9,57,5,53],
        [42,26,38,22,41,25,37,21]
    ], dtype=np.float32)
    return ((b + 0.5) / 64.0 * 255.0).astype(np.uint8)

BAYER8 = _make_bayer8_threshold_map()

def _tile_threshold_map(shape, bayer=BAYER8, scale=1.0):
    h, w = shape
    by, bx = bayer.shape
    reps_y = math.ceil(h / by)
    reps_x = math.ceil(w / bx)
    tiled = np.tile(bayer, (reps_y, reps_x))
    tiled = tiled[:h, :w].astype(np.float32)
    if scale != 1.0:
        tiled = np.clip(tiled * scale, 0, 255)
    return tiled.astype(np.uint8)

# ------------------------------------------
# Палитры: две версии — со сдвигом и без сдвига
# ------------------------------------------

def make_alternate_palette_caesar(centers_rgb, n_colors, ink_hues=None, ink_dark_threshold=0.18):
    """
    Версия 'со сдвигом' (Цезарь). Гарантированно применяет ненулевой сдвиг:
      offset = int(n_colors) % L, но если результат 0 — используем offset=1.
    Это обеспечивает изменение палитры при любых n_colors.
    centers_rgb: Nx3 uint8
    n_colors: int (может быть None) — если None, offset будет = 1 (гарантированный сдвиг)
    ink_hues: optional list базовых оттенков
    """
    if ink_hues is None:
        ink_hues = BASE_INK_HUES
    L = len(ink_hues)

    # вычисляем offset; гарантируем ненулевой сдвиг
    if n_colors is None:
        offset = 1
    else:
        try:
            offset = int(n_colors) % L
        except Exception:
            offset = 1
        if offset == 0:
            offset = 1

    centers = np.asarray(centers_rgb, dtype=np.float32) / 255.0
    hsv = _rgb_to_hsv_array(centers)
    h = hsv[:, 0].copy()
    s = hsv[:, 1].copy()
    v = hsv[:, 2].copy()

    new_h = np.zeros_like(h)
    for i, orig_h in enumerate(h):
        best_idx = 0
        best_dist = 10.0
        for j, ih in enumerate(ink_hues):
            d = _hue_distance(orig_h, ih)
            if d < best_dist:
                best_dist = d
                best_idx = j
        target_idx = (best_idx + offset) % L
        new_h[i] = ink_hues[target_idx]

    dark_mask = v <= ink_dark_threshold
    new_s = np.clip(s * 1.15 + 0.03, 0.02, 0.98)
    new_v = np.clip(v * 0.9 + 0.02, 0.02, 0.98)
    new_hsv = np.stack([new_h, new_s, new_v], axis=1)
    if np.any(dark_mask):
        new_hsv[dark_mask, 1] = np.clip(new_hsv[dark_mask, 1] * 0.2, 0.0, 0.25)
        new_hsv[dark_mask, 2] = np.clip(new_hsv[dark_mask, 2] * 0.6, 0.02, 0.25)

    new_rgb = _hsv_to_rgb_array(new_hsv)
    return np.clip((new_rgb * 255.0).round(), 0, 255).astype(np.uint8)

def make_alternate_palette_no_shift(centers_rgb, ink_hues=None, ink_dark_threshold=0.18):
    """
    Версия 'без сдвига' — сопоставляет каждый центр ближайшему базовому ink_hue,
    но НЕ сдвигает индекс. Используется, когда нужен классический (без Цезаря) маппинг.
    centers_rgb: Nx3 uint8
    """
    if ink_hues is None:
        ink_hues = BASE_INK_HUES

    centers = np.asarray(centers_rgb, dtype=np.float32) / 255.0
    hsv = _rgb_to_hsv_array(centers)
    h = hsv[:, 0].copy()
    s = hsv[:, 1].copy()
    v = hsv[:, 2].copy()

    new_h = np.zeros_like(h)
    for i, orig_h in enumerate(h):
        best_idx = 0
        best_dist = 10.0
        for j, ih in enumerate(ink_hues):
            d = _hue_distance(orig_h, ih)
            if d < best_dist:
                best_dist = d
                best_idx = j
        new_h[i] = ink_hues[best_idx]

    dark_mask = v <= ink_dark_threshold
    new_s = np.clip(s * 1.15 + 0.03, 0.02, 0.98)
    new_v = np.clip(v * 0.9 + 0.02, 0.02, 0.98)
    new_hsv = np.stack([new_h, new_s, new_v], axis=1)
    if np.any(dark_mask):
        new_hsv[dark_mask, 1] = np.clip(new_hsv[dark_mask, 1] * 0.2, 0.0, 0.25)
        new_hsv[dark_mask, 2] = np.clip(new_hsv[dark_mask, 2] * 0.6, 0.02, 0.25)

    new_rgb = _hsv_to_rgb_array(new_hsv)
    return np.clip((new_rgb * 255.0).round(), 0, 255).astype(np.uint8)

# ------------------------------------------
# Главная функция — apply_risograph поддерживает выбор режима палитры
# ------------------------------------------

def apply_risograph(img, w, h, out_dir=None, base_name=None, **kwargs):
    """
    Риcо-эффект.
    Возвращает RGB uint8 image (с сдвигом).
    Также можно сохранять вариант без сдвига отдельно.
    """
    palette_mode = kwargs.get('palette_mode', 'shift')  # 'shift' или 'no_shift'
    n_colors_kw = kwargs.get('n_colors', None)
    dot_scale = float(kwargs.get('dot_scale', 1.0))
    max_shift = int(kwargs.get('max_shift', 3))
    ink_hues = kwargs.get('ink_hues', None)

    img = np.asarray(img, dtype=np.uint8)
    H, W = img.shape[:2]

    flat = img.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    num_unique = unique_colors.shape[0]

    if n_colors_kw is None:
        n_colors = num_unique
    else:
        try:
            n_colors = int(n_colors_kw)
        except Exception:
            n_colors = num_unique

    if num_unique > n_colors:
        unique_colors = unique_colors[:n_colors]
        d = np.linalg.norm(flat[:, None, :].astype(np.int16) - unique_colors[None, ...].astype(np.int16), axis=2)
        inverse = np.argmin(d, axis=1)
        num_unique = unique_colors.shape[0]

    # --- палитра со сдвигом ---
    alt_palette_shift = make_alternate_palette_caesar(unique_colors, n_colors, ink_hues=ink_hues)

    # --- палитра без сдвига ---
    alt_palette_no_shift = make_alternate_palette_no_shift(unique_colors, ink_hues=ink_hues)

    # Bayer threshold
    th_map = _tile_threshold_map((H, W), BAYER8, scale=dot_scale)
    hsv_cv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    V_map = hsv_cv[:, :, 2].astype(np.uint8)
    inverse_2d = inverse.reshape(H, W)

    def render(alt_palette):
        out = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(num_unique):
            mask_base = (inverse_2d == i)
            if not np.any(mask_base):
                continue

            intensity = (V_map * mask_base).astype(np.uint8)
            dot_mask = (intensity > th_map).astype(np.uint8)

            dx = ((i * 3) % (2 * max_shift + 1)) - max_shift
            dy = ((i * 5) % (2 * max_shift + 1)) - max_shift
            if dx != 0 or dy != 0:
                dot_mask = np.roll(dot_mask, shift=dx, axis=1)
                dot_mask = np.roll(dot_mask, shift=dy, axis=0)

            color = alt_palette[i].astype(np.float32) / 255.0
            for c in range(3):
                out[:, :, c] += dot_mask.astype(np.float32) * (color[c] * 255.0)

        out = np.clip(out, 0, 255).astype(np.uint8)

        paper_noise = kwargs.get('paper_noise', 0.03)
        if paper_noise and paper_noise > 0.0:
            noise = (np.random.default_rng(123).normal(loc=0.0, scale=paper_noise * 255.0, size=out.shape)).astype(np.int16)
            out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return out

    # --- основной результат (со сдвигом) ---
    out_shift = render(alt_palette_shift)

    # --- вариант без сдвига ---
    out_no_shift = render(alt_palette_no_shift)

    # --- сохранение, если out_dir указан ---
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        name_stem = Path(base_name).stem if base_name else "risograph_result"
        # сохраняем вариант без сдвига с приставкой
        cv2.imwrite(str(Path(out_dir) / f"{name_stem}_no_shift.png"), cv2.cvtColor(out_no_shift, cv2.COLOR_RGB2BGR))

    # возвращаем основной результат
    return out_shift