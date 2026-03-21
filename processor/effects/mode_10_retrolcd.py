# processor/effects/mode_10_retrolcd
import cv2
import numpy as np
import re
from pathlib import Path

def _get_idx(base_name):
    if not base_name:
        return 0
    m = re.search(r'(\d+)', base_name)
    return int(m.group(1)) if m else 0

def apply_retrolcd(img, w, h, out_dir, base_name):
    small = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    up = up.astype(np.float32) / 255.0

    line = np.linspace(0.9, 1.05, h, dtype=np.float32).reshape(h, 1)
    lcd = up * line[..., None] + 0.08
    lcd = np.clip(lcd, 0, 1)

    gray = cv2.cvtColor(lcd, cv2.COLOR_BGR2GRAY)

    palettes = [
        # 0: Классический GameBoy (Зеленый)
        [[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]],  
        
        # 1: Amber (Старый терминал)
        [[0, 10, 30], [0, 60, 120], [0, 120, 200], [120, 200, 255]],  
        
        # 2: Virtual Boy (Кроваво-красный — агрессивно и стильно)
        [[0, 0, 0], [0, 0, 80], [0, 0, 180], [20, 20, 255]], 

        # 3: Matrix / Fallout (Ядовито-зеленый терминал)
        [[5, 20, 5], [10, 80, 10], [30, 180, 30], [180, 255, 180]],

        # 4: Cyberpunk / Neon (Фиолетовый и бирюза)
        [[50, 10, 40], [120, 40, 120], [200, 200, 50], [255, 200, 255]],

        # 5: E-Ink / Kindle (Газетная бумага, высокий контраст)
        [[25, 25, 25], [80, 80, 80], [160, 160, 160], [220, 225, 230]],

        # 6: CGA Mode 4 (Классика DOS: Маджента и Циан)
        [[0, 0, 0], [180, 50, 180], [180, 180, 50], [240, 240, 240]],

        # 7: Cold Ocean (Глубокий синий)
        [[40, 20, 0], [100, 60, 0], [180, 140, 50], [255, 240, 200]],
        
        # 8: Crimson Night (Черно-бордовый, нуар)
        [[10, 0, 20], [40, 10, 60], [80, 20, 120], [180, 150, 255]],

        # 9: Моя новая палитра "Токсичный океан"
        [[40, 20, 0], [100, 100, 0], [200, 255, 100], [255, 255, 200]],

        # 10:
        [[10, 10, 10], [99, 99, 90], [151, 156, 98], [220, 227, 141]]
    ]

    # --- основной результат ---
    palette_main = np.array(palettes[0], dtype=np.float32) / 255.0
    i = np.floor(gray * (len(palette_main)-1)).astype(np.int32)
    i = np.clip(i, 0, len(palette_main)-1)
    result_main = palette_main[i]

    # --- дополнительные сохраняем ---
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for idx, pal in enumerate(palettes[1:], start=1):
            palette = np.array(pal, dtype=np.float32) / 255.0
            j = np.floor(gray * (len(palette)-1)).astype(np.int32)
            j = np.clip(j, 0, len(palette)-1)
            res = (palette[j] * 255).astype(np.uint8)

            path = Path(out_dir) / f"{Path(base_name).stem}_lcd_{idx}.png"
            cv2.imwrite(str(path), res)

    return (result_main * 255).astype(np.uint8)
