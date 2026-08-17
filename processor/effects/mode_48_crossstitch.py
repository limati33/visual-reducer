# processor/effects/mode_48_crossstitch.py
import cv2
import numpy as np

def apply_crossstitch(img, w=None, h=None, out_dir=None, base_name=None,
                       cell=10, canvas_color=(238, 232, 220)):
    h_img, w_img = img.shape[:2]
    canvas = np.full_like(img, canvas_color)

    # текстура ткани — тонкая сетка переплетения, видна там, где нет стежков
    for y in range(0, h_img, 3):
        cv2.line(canvas, (0, y), (w_img, y), (220, 214, 200), 1)
    for x in range(0, w_img, 3):
        cv2.line(canvas, (x, 0), (x, h_img), (220, 214, 200), 1)

    for y in range(0, h_img, cell):
        for x in range(0, w_img, cell):
            block = img[y:y + cell, x:x + cell]
            if block.size == 0:
                continue
            color = block.reshape(-1, 3).mean(axis=0).astype(int)
            brightness = color.mean()

            # светлые клетки — канва остаётся пустой, стежок не кладём
            if brightness > 225:
                continue

            thickness = 2 if brightness < 90 else 1
            c = tuple(int(v) for v in color)
            cv2.line(canvas, (x, y), (x + cell, y + cell), c, thickness, cv2.LINE_AA)
            cv2.line(canvas, (x + cell, y), (x, y + cell), c, thickness, cv2.LINE_AA)

            # в тёмных клетках — двойной крестик для плотности нити
            if brightness < 90:
                cv2.line(canvas, (x + 2, y), (x + cell, y + cell - 2), c, 1, cv2.LINE_AA)
                cv2.line(canvas, (x + cell, y + 2), (x + 2, y + cell), c, 1, cv2.LINE_AA)

    return canvas