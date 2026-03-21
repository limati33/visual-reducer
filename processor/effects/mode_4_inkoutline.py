# processor/effects/mode_4_ink.py
import cv2
import numpy as np

def apply_inkoutline(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Линии через Разность Гауссиан (DoG)
    # Это находит контуры лучше, чем Canny, и выглядит как штриховка
    blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur2 = cv2.GaussianBlur(gray, (13, 13), 0)
    
    # Вычитаем одно размытие из другого
    dog = blur1.astype(np.float32) - blur2.astype(np.float32)
    
    # Все, что меньше определенного порога - это линия тушью
    # Делаем линии черными (0), остальное белым (255)
    lines = np.where(dog < -3, 0, 255).astype(np.uint8)

    # 2. Глубокие заливные тени (как в графическом романе)
    # Сглаживаем оригинал, чтобы тени не были рваными
    smooth_gray = cv2.medianBlur(gray, 5)
    _, shadows = cv2.threshold(smooth_gray, 80, 255, cv2.THRESH_BINARY)

    # 3. Смешиваем тени и линии
    # Если пиксель черный в тенях ИЛИ черный в линиях -> он будет черным
    ink_result = cv2.bitwise_and(lines, shadows)

    # Возвращаем 3 канала, чтобы не ломать твой пайплайн
    result = cv2.cvtColor(ink_result, cv2.COLOR_GRAY2BGR)

    return result