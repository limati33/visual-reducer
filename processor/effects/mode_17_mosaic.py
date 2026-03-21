# processor/effects/mode_17_mosaic.py
import cv2
import numpy as np
import random
import math

def apply_mosaic(img, w, h, out_dir, base_name):
    # Если переданы размеры, масштабируем
    if w and h:
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    hq, wq = img.shape[:2]
    
    # Размер ячейки. Можно сделать параметром, сейчас жестко 8
    base = 8 
    
    # Фон мозаики (затирка между плитками). Сделаем темно-серой.
    abstract = np.full_like(img, (30, 30, 30))

    shapes = ["circle", "square", "triangle", "diamond"]

    # Проходим по сетке
    for y in range(0, hq, base):
        for x in range(0, wq, base):
            
            # Границы блока
            x1 = min(wq, x + base)
            y1 = min(hq, y + base)
            
            # Защита от нулевых блоков на краях
            if x1 - x <= 0 or y1 - y <= 0:
                continue

            block = img[y:y1, x:x1]
            
            # Берем средний цвет блока
            b_mean, g_mean, r_mean = block.mean(axis=(0, 1))
            
            # ДОБАВЛЯЕМ ВИТРАЖНЫЙ ЭФФЕКТ (Легкая случайная вариация яркости)
            # Это заменило сломанный блок if lum > 130
            brightness_shift = random.randint(-15, 15)
            b = np.clip(b_mean + brightness_shift, 0, 255)
            g = np.clip(g_mean + brightness_shift, 0, 255)
            r = np.clip(r_mean + brightness_shift, 0, 255)
            
            shape_col = (int(b), int(g), int(r))

            # Параметры фигуры
            shape = random.choice(shapes)
            cx_px = x + (x1 - x) // 2
            cy_px = y + (y1 - y) // 2
            
            # Немного уменьшаем размер, чтобы было видно фон (затирку)
            scale = random.uniform(0.6, 0.9)
            hw = int((base / 2) * scale)
            
            # Рисуем фигуру
            if shape == "circle":
                cv2.circle(abstract, (cx_px, cy_px), hw, shape_col, -1)
            
            elif shape == "square":
                pt1 = (cx_px - hw, cy_px - hw)
                pt2 = (cx_px + hw, cy_px + hw)
                cv2.rectangle(abstract, pt1, pt2, shape_col, -1)
                
            elif shape == "diamond":
                pts = np.array([
                    [cx_px, cy_px - hw],
                    [cx_px + hw, cy_px],
                    [cx_px, cy_px + hw],
                    [cx_px - hw, cy_px]
                ], np.int32)
                cv2.fillConvexPoly(abstract, pts, shape_col)
                
            elif shape == "triangle":
                # Рандомное направление треугольника
                if random.choice([True, False]):
                    pts = np.array([
                        [cx_px, cy_px - hw], [cx_px - hw, cy_px + hw], [cx_px + hw, cy_px + hw]
                    ], np.int32)
                else:
                    pts = np.array([
                        [cx_px, cy_px + hw], [cx_px - hw, cy_px - hw], [cx_px + hw, cy_px - hw]
                    ], np.int32)
                cv2.fillConvexPoly(abstract, pts, shape_col)

    # Лёгкий шум для фактуры камня
    noise = np.random.normal(0, 5, abstract.shape).astype(np.int16)
    result = np.clip(abstract.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return result