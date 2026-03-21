# processor/effects/mode_9_plastic.py
import cv2
import numpy as np

def apply_plastic(img, w, h, out_dir, base_name):
    # 1. Сильное сглаживание (основа пластика)
    smooth = cv2.bilateralFilter(img, 9, 100, 100)

    # 2. Усиление цвета
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)

    # 3. Блики (по яркости)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    spec = cv2.GaussianBlur(gray, (0, 0), 10)
    spec = cv2.normalize(spec, None, 0, 80, cv2.NORM_MINMAX)

    # добавляем в V канал
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], spec)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4. Лёгкий detail для глянца
    result = cv2.detailEnhance(result, sigma_s=8, sigma_r=0.2)

    return result
