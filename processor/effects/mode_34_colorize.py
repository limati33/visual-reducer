# processor/effects/mode_34_neon_grid.py
import cv2
import numpy as np
import os

def apply_colorize(img, w=None, h=None, out_dir=None, base_name=None):

    # 1. Resize
    ih, iw = img.shape[:2]
    if w and h and (iw != w or ih != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # 2. Градация в холодные тона
    # Переводим в LAB для лучшего контроля яркости
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Создаём холодный градиент (циан/синий/фиолет)
    gradient = np.zeros_like(img, dtype=np.uint8)
    gradient[:, :, 0] = 180  # B
    gradient[:, :, 1] = 255  # G
    gradient[:, :, 2] = 255  # R (будет смягчён через LUT ниже)

    # 3. Применяем LUT по яркости
    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    for i in range(256):
        # тёмные → синий, средние → циан, светлые → бело-голубой
        b = np.clip(180 + i//2, 0, 255)
        g = np.clip(100 + i//1.2, 0, 255)
        r = np.clip(i//2, 0, 255)
        lut[0, i] = [b, g, r]

    colored = cv2.LUT(cv2.cvtColor(l_channel, cv2.COLOR_GRAY2BGR), lut)

    # 4. Добавляем horizontal scanlines
    scanline = np.sin(np.linspace(0, np.pi*8, h)) * 20  # амплитуда 20
    scanline = scanline[:, np.newaxis] / 255.0
    for c in range(3):
        colored[:, :, c] = np.clip(colored[:, :, c] + (colored[:, :, c] * scanline), 0, 255)

    # 5. Добавляем шум для «живости»
    noise = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
    colored = np.clip(colored.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 6. Лёгкий вертикальный glow (размытие сверху вниз)
    kernel = cv2.getGaussianKernel(9, 2)
    colored = cv2.filter2D(colored, -1, kernel @ kernel.T)

    # 7. Сохраняем результат, если out_dir задан
    if out_dir and base_name:
        out_path = os.path.join(out_dir, f"{base_name}_mode34_hologram.png")
        cv2.imwrite(out_path, colored)

    return colored