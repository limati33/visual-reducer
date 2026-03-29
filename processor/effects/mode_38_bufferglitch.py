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

    # --- 1. ЖЁСТКИЕ полосы ---
    for _ in range(max(3, h // 80)):
        y = random.randint(0, h - 1)
        height = random.randint(6, 22)
        shift = random.randint(-w // 6, w // 6)

        y_end = min(h, y + height)
        strip = result[y:y_end, :].copy()

        # иногда рвём строку на части
        if random.random() < 0.4:
            cut = random.randint(0, w)
            strip[:, :cut] = np.roll(strip[:, :cut], shift, axis=1)
        else:
            strip = np.roll(strip, shift, axis=1)

        result[y:y_end, :] = strip

    # --- 2. Глитч-блоки (с телепортом) ---
    for _ in range(10):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        bw = random.randint(w // 20, w // 4)
        bh = random.randint(h // 20, h // 4)

        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        block = result[y:y2, x:x2].copy()
        if block.size == 0:
            continue

        # иногда просто переносим (жёстко)
        if random.random() < 0.35:
            tx = random.randint(0, max(0, w - block.shape[1]))
            ty = random.randint(0, max(0, h - block.shape[0]))
            result[ty:ty + block.shape[0], tx:tx + block.shape[1]] = block
        else:
            dx = random.randint(-w // 8, w // 8)
            dy = random.randint(-h // 8, h // 8)

            tx = int(np.clip(x + dx, 0, max(0, w - block.shape[1])))
            ty = int(np.clip(y + dy, 0, max(0, h - block.shape[0])))

            target = result[ty:ty + block.shape[0], tx:tx + block.shape[1]]
            if target.shape == block.shape:
                result[ty:ty + block.shape[0], tx:tx + block.shape[1]] = cv2.addWeighted(
                    target, 0.3, block, 0.7, 0
                )

    # --- 3. DATA MOSH (размазка движения) ---
    for _ in range(3):
        y = random.randint(0, h - 1)
        height = random.randint(10, 40)

        y2 = min(h, y + height)
        strip = result[y:y2, :].copy()

        shift = random.randint(-w // 10, w // 10)
        smear = np.roll(strip, shift, axis=1)

        result[y:y2, :] = cv2.addWeighted(strip, 0.4, smear, 0.6, 0)

    # --- 4. Битые строки ---
    for _ in range(h // 50):
        y = random.randint(0, h - 1)
        if random.random() < 0.5:
            result[y:y+1, :] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
        else:
            result[y:y+1, :] = 0

    # --- 5. Битые зоны ---
    for _ in range(4):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        bw = random.randint(w // 12, w // 3)
        bh = random.randint(h // 12, h // 3)

        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        if x2 <= x or y2 <= y:
            continue

        if random.random() < 0.4:
            block = np.zeros((y2 - y, x2 - x, 3), dtype=np.uint8)
        else:
            block = np.random.randint(0, 255, (y2 - y, x2 - x, 3), dtype=np.uint8)

        result[y:y2, x:x2] = cv2.addWeighted(result[y:y2, x:x2], 0.6, block, 0.4, 0)

    # --- 6. Усиленный RGB split + jitter ---
    shift_x = random.randint(2, 6)
    shift_y = random.randint(2, 6)

    b, g, r = cv2.split(result)

    r = np.roll(r, shift_x, axis=1)
    b = np.roll(b, -shift_y, axis=0)

    # лёгкий jitter
    g = np.roll(g, random.randint(-2, 2), axis=1)

    result = cv2.merge([b, g, r])

    return result
