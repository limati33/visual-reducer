# processor/effects/mode_43_engraving.py
import cv2
import numpy as np

def apply_engraving(img, w=None, h=None, out_dir=None, base_name=None,
                     cell=9, max_len=7, jitter=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    h_img, w_img = gray.shape
    canvas = np.full((h_img, w_img, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(0)

    def draw_stroke(cx, cy, angle, length, thickness=1):
        dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        cv2.line(canvas, p1, p2, (20, 20, 20), thickness, cv2.LINE_AA)

    for y in range(cell // 2, h_img, cell):
        for x in range(cell // 2, w_img, cell):
            darkness = 1.0 - gray[y, x] / 255.0
            if darkness < 0.08:
                continue
            jx = x + rng.integers(-jitter, jitter + 1)
            jy = y + rng.integers(-jitter, jitter + 1)
            angle = np.arctan2(gy[y, x], gx[y, x]) + np.pi / 2
            length = max_len * min(darkness * 1.4, 1.0)

            # базовый слой штриховки — всегда
            draw_stroke(jx, jy, angle, length)

            # кросс-хетч только в тёмных зонах — второй слой под ~90°
            if darkness > 0.55:
                draw_stroke(jx, jy, angle + np.pi / 2, length * 0.8)
            # третий слой для самых тёмных зон — плотная штриховка
            if darkness > 0.8:
                draw_stroke(jx, jy, angle + np.pi / 4, length * 0.6)

    return canvas