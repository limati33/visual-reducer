# processor/effects/mode_46_oscilloscope.py
import cv2
import numpy as np

def apply_oscilloscope(img, w=None, h=None, out_dir=None, base_name=None,
                        row_step=4, amplitude=20, glow=True, palette=None, **kwargs):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape
    canvas = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    # 1. Если палитра передана из K-Means — переводим RGB -> BGR
    # 2. Если нет — ставим дефолтную неоновую палитру
    if palette is not None and len(palette) > 0:
        bgr_palette = [tuple(int(c) for c in color[::-1]) for color in palette]
    else:
        bgr_palette = [
            (0, 255, 255),   # Yellow
            (255, 0, 255),   # Magenta
            (255, 255, 0),   # Cyan
            (0, 255, 0)      # Green
        ]

    color_index = 0

    for y in range(0, h_img, row_step):
        row = gray[y].astype(np.float32) / 255.0
        offsets = ((row - 0.5) * amplitude).astype(np.int32)
        ys = np.clip(y + offsets, 0, h_img - 1)
        pts = np.stack([np.arange(w_img), ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

        # Выбираем цвет по очереди из переданной палитры K-Means
        color = bgr_palette[color_index % len(bgr_palette)]
        color_index += 1

        cv2.polylines(canvas, [pts], False, color, 1, cv2.LINE_AA)

    if glow:
        blur = cv2.GaussianBlur(canvas, (0, 0), sigmaX=3)
        canvas = cv2.addWeighted(canvas, 1.0, blur, 0.6, 0)

    return canvas