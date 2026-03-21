# processor/effects/mode_34_neon_grid.py
import cv2
import numpy as np
import os

def _build_palette_lut():
    # Палитра в BGR: тёмно-синий -> циан -> фиолетовый -> бело-голубой
    stops = np.array([
        [20, 10, 60],
        [255, 180, 40],
        [255, 90, 180],
        [255, 255, 255],
    ], dtype=np.float32)

    xs = np.linspace(0.0, 1.0, 256)
    t = np.linspace(0.0, 1.0, len(stops))

    lut = np.zeros((256, 3), dtype=np.uint8)
    for i, x in enumerate(xs):
        idx = np.searchsorted(t, x) - 1
        idx = np.clip(idx, 0, len(stops) - 2)
        x0, x1 = t[idx], t[idx + 1]
        a = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        color = (1 - a) * stops[idx] + a * stops[idx + 1]
        lut[i] = np.clip(color, 0, 255)

    return lut

_PALETTE_LUT = _build_palette_lut()

def apply_colorize(img, w, h, out_dir, base_name):
    # 1. Resize
    ih, iw = img.shape[:2]
    if w and h and (iw != w or ih != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # 2. Работаем через LAB: берём только яркость
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0]

    # Немного усилим контраст яркости
    l = cv2.equalizeHist(l)

    # 3. Маппинг яркости в палитру
    colored = _PALETTE_LUT[l]

    # 4. Очень мягкий scanline-эффект
    scan = (0.92 + 0.08 * np.sin(np.linspace(0, np.pi * 12, h))).astype(np.float32)
    colored = colored.astype(np.float32)
    colored *= scan[:, np.newaxis, np.newaxis]

    # 5. Лёгкий шум, чтобы не было "стерильно"
    noise = np.random.normal(0, 4, (h, w, 3)).astype(np.float32)
    colored = np.clip(colored + noise, 0, 255)

    # 6. Мягкий glow
    glow = cv2.GaussianBlur(colored.astype(np.uint8), (0, 0), 2.0)
    colored = cv2.addWeighted(colored.astype(np.uint8), 0.82, glow, 0.18, 0)

    # 7. Слегка затемняем края, чтобы был объём
    yy, xx = np.indices((h, w))
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / (dist.max() + 1e-8)
    vignette = np.clip(1.0 - 0.25 * dist, 0.75, 1.0).astype(np.float32)
    colored = np.clip(colored.astype(np.float32) * vignette[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # 8. Сохраняем результат
    if out_dir and base_name:
        out_path = os.path.join(out_dir, f"{base_name}_mode34_hologram.png")
        cv2.imwrite(out_path, colored)

    return colored
