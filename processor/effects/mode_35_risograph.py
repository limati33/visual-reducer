# processor/effects/mode_35_risograph.py
import numpy as np
import cv2
import colorsys
import math

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
    # vectorized via colorsys in loop (palette sizes small)
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
    # normalize to 0..255
    return ((b + 0.5) / 64.0 * 255.0).astype(np.uint8)

BAYER8 = _make_bayer8_threshold_map()

def _tile_threshold_map(shape, bayer=BAYER8, scale=1.0):
    """Tile bayer map to the desired image shape and optionally scale (0..1) brightness threshold"""
    h, w = shape
    by, bx = bayer.shape
    # tile
    reps_y = math.ceil(h / by)
    reps_x = math.ceil(w / bx)
    tiled = np.tile(bayer, (reps_y, reps_x))
    tiled = tiled[:h, :w].astype(np.float32)
    if scale != 1.0:
        tiled = np.clip(tiled * scale, 0, 255)
    return tiled.astype(np.uint8)

def make_alternate_palette_caesar(centers_rgb, n_colors, ink_hues=None, ink_dark_threshold=0.18):
    """
    Построить альтернативную палитру в стиле "Цезаря":
    - centers_rgb: Nx3 uint8
    - n_colors: количество цветов (int) -> определяет сдвиг offset = n_colors % len(ink_hues)
    - ink_hues: optional list базовых оттенков
    Возвращает Nx3 uint8.
    """
    if ink_hues is None:
        ink_hues = BASE_INK_HUES
    L = len(ink_hues)
    offset = int(n_colors) % L if n_colors is not None else 0

    centers = np.asarray(centers_rgb, dtype=np.float32) / 255.0
    hsv = _rgb_to_hsv_array(centers)  # Nx3
    h = hsv[:, 0].copy()
    s = hsv[:, 1].copy()
    v = hsv[:, 2].copy()

    new_h = np.zeros_like(h)
    for i, orig_h in enumerate(h):
        # найти ближайший ink_hues индекс
        best_idx = 0
        best_dist = 10.0
        for j, ih in enumerate(ink_hues):
            d = _hue_distance(orig_h, ih)
            if d < best_dist:
                best_dist = d
                best_idx = j
        target_idx = (best_idx + offset) % L
        new_h[i] = ink_hues[target_idx]

    # для тёмных — понижаем насыщенность/яркость, чтобы не получить "цветной" черный
    dark_mask = v <= ink_dark_threshold
    new_s = np.clip(s * 1.15 + 0.03, 0.02, 0.98)
    new_v = np.clip(v * 0.9 + 0.02, 0.02, 0.98)
    new_hsv = np.stack([new_h, new_s, new_v], axis=1)
    if np.any(dark_mask):
        new_hsv[dark_mask, 1] = np.clip(new_hsv[dark_mask, 1] * 0.2, 0.0, 0.25)
        new_hsv[dark_mask, 2] = np.clip(new_hsv[dark_mask, 2] * 0.6, 0.02, 0.25)

    new_rgb = _hsv_to_rgb_array(new_hsv)
    return np.clip((new_rgb * 255.0).round(), 0, 255).astype(np.uint8)

def apply_risograph(img, w, h, out_dir, base_name, **kwargs):
    """
    Риcо-эффект.
    Поддерживаемые kwargs:
      - n_colors: int или None (если None — определяем по уникальным цветам)
      - dot_scale: float (0.5..1.5) — масштаб порога Bayer (меняет плотность точек)
      - max_shift: int — макс. пикселей для смещения misregistration
      - ink_hues: optional list базовых ink оттенков
    Возвращает RGB uint8 image.
    """
    # параметры
    n_colors_kw = kwargs.get('n_colors', None)
    dot_scale = float(kwargs.get('dot_scale', 1.0))
    max_shift = int(kwargs.get('max_shift', 3))
    ink_hues = kwargs.get('ink_hues', None)

    # Преобразования
    img = np.asarray(img, dtype=np.uint8)
    H, W = img.shape[:2]

    # получаем уникальные цвета и обратные индексы
    flat = img.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    num_unique = unique_colors.shape[0]
    # определяем n_colors (источник правды)
    if n_colors_kw is None:
        n_colors = num_unique
    else:
        try:
            n_colors = int(n_colors_kw)
        except Exception:
            n_colors = num_unique

    # если слишком много уникальных цветов — оставляем первые n_colors (защита)
    if num_unique > n_colors:
        # в нормальных условиях unique_colors coming from quantized image => num_unique == n_colors
        unique_colors = unique_colors[:n_colors]
        # нужно пересчитать inverse: найдем ближайший цвет из unique_colors для каждого пикселя
        # (возможно тяжеловато, но редкий случай)
        # рассчитываем евклид. расстояние между flat и unique_colors
        d = np.linalg.norm(flat[:, None, :].astype(np.int16) - unique_colors[None, ...].astype(np.int16), axis=2)
        inverse = np.argmin(d, axis=1)
        num_unique = unique_colors.shape[0]

    # 1) строим альтернативную палитру (Caesar)
    alt_palette = make_alternate_palette_caesar(unique_colors, n_colors, ink_hues=ink_hues)

    # 2) подготовим карту порогов (Bayer), можно масштабировать dot_scale
    th_map = _tile_threshold_map((H, W), BAYER8, scale=dot_scale)

    # 3) Получим Value (яркость) из HSV quantized img — используем его как "интенсивность" для каждой позиции
    #    Конвертация: RGB->HSV через cv2: H:0-179, S:0-255, V:0-255
    hsv_cv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    V_map = hsv_cv[:, :, 2].astype(np.uint8)

    # 4) Собираем итоговое изображение по слоям (dot screens)
    out = np.zeros((H, W, 3), dtype=np.float32)

    inverse_2d = inverse.reshape(H, W)

    # для вариативности misregistration используем циклические смещения, зависящие от индекса
    for i in range(num_unique):
        # интенсивность этого слоя: значение яркости там, где пиксели принадлежат этой кластерной метке
        mask_base = (inverse_2d == i)
        if not np.any(mask_base):
            continue  # нет пикселей этого кластера

        # intensity map (0..255) — берём V_map там, где mask, иначе 0
        intensity = (V_map * mask_base).astype(np.uint8)

        # ordered dither: dot mask = intensity > threshold_map
        # но чтобы точки не появлялись исключительно как бинар на границе, можно немного смягчить
        dot_mask = (intensity > th_map).astype(np.uint8)  # 0/1

        # смещение слоя (misregistration): небольшое циклическое смещение по (dx,dy)
        # разумно сделать зависимость от i, чтобы разные цвета смещались по-разному
        dx = ((i * 3) % (2 * max_shift + 1)) - max_shift
        dy = ((i * 5) % (2 * max_shift + 1)) - max_shift
        if dx != 0 or dy != 0:
            dot_mask = np.roll(dot_mask, shift=dx, axis=1)
            dot_mask = np.roll(dot_mask, shift=dy, axis=0)

        # слой цвета
        color = alt_palette[i].astype(np.float32) / 255.0  # 0..1
        # добавляем в итог (аддитивное смешение, как при печати ризо)
        # масштабируем dot_mask (0/1) на color*255
        for c in range(3):
            out[:, :, c] += dot_mask.astype(np.float32) * (color[c] * 255.0)

    # после наложения точек — возможные артефакты яркости >255. Clip
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Optional: немного текстуры/шума, чтобы имитировать бумагу (тонкий шум)
    paper_noise = kwargs.get('paper_noise', 0.03)  # 0..0.2
    if paper_noise and paper_noise > 0.0:
        noise = (np.random.default_rng(123).normal(loc=0.0, scale=paper_noise * 255.0, size=out.shape)).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return out
