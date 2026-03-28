# processor/effects/mode_38_bufferglitch.py
import cv2
import numpy as np
import random

def apply_bufferglitch(img, w, h, out_dir, base_name):
    ih, iw = img.shape[:2]
    if w and h and (iw != w or ih != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    result = img.copy()

    # --- 1. Небольшие полосы ---
    for _ in range(max(1, h // 120)):
        y = random.randint(0, h - 1)
        height = random.randint(4, 14)
        shift = random.randint(-w // 10, w // 10)

        y_end = min(h, y + height)
        strip = result[y:y_end, :].copy()
        result[y:y_end, :] = np.roll(strip, shift, axis=1)

    # --- 2. Несколько блоков ---
    for _ in range(6):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        bw = random.randint(max(20, w // 20), max(40, w // 6))
        bh = random.randint(max(20, h // 20), max(40, h // 6))

        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        block = result[y:y2, x:x2].copy()
        bh2, bw2 = block.shape[:2]
        if bh2 == 0 or bw2 == 0:
            continue

        dx = random.randint(-w // 15, w // 15)
        dy = random.randint(-h // 15, h // 15)

        tx = int(np.clip(x + dx, 0, max(0, w - bw2)))
        ty = int(np.clip(y + dy, 0, max(0, h - bh2)))

        # смешиваем, а не затираем полностью
        target = result[ty:ty + bh2, tx:tx + bw2]
        if target.shape == block.shape:
            result[ty:ty + bh2, tx:tx + bw2] = cv2.addWeighted(target, 0.45, block, 0.55, 0)

    # --- 3. Лёгкие залипшие куски ---
    for _ in range(3):
        y = random.randint(0, max(0, h - 30))
        block = result[y:y + 20, :].copy()

        for i in range(2):
            yy = y + i * 12
            if yy + block.shape[0] <= h:
                result[yy:yy + block.shape[0], :] = cv2.addWeighted(
                    result[yy:yy + block.shape[0], :], 0.55, block, 0.45, 0
                )

    # --- 4. Небольшие битые зоны ---
    for _ in range(2):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        bw = random.randint(max(20, w // 12), max(50, w // 5))
        bh = random.randint(max(20, h // 12), max(50, h // 5))

        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        if x2 <= x or y2 <= y:
            continue

        if random.random() < 0.5:
            block = np.zeros((y2 - y, x2 - x, 3), dtype=np.uint8)
        else:
            block = np.random.randint(0, 255, (y2 - y, x2 - x, 3), dtype=np.uint8)

        target = result[y:y2, x:x2]
        result[y:y2, x:x2] = cv2.addWeighted(target, 0.7, block, 0.3, 0)

    # --- 5. Лёгкий RGB split ---
    shift = 3
    b, g, r = cv2.split(result)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=0)
    result = cv2.merge([b, g, r])

    return result
