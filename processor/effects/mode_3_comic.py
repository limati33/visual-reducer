# processor/effects/mode_3_comic.py
import cv2
import numpy as np

def apply_comic(img, w, h, out_dir, base_name):
    # 1. Генерируем красивые, "живые" контуры (вместо Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Медианный блюр убивает мелкий шум, оставляя ровные края
    gray_blur = cv2.medianBlur(gray, 7)
    # Адаптивный порог дает линии разной толщины (как нажим кисти)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11, C=9
    )

    # 2. Делаем "плоские" комиксные цвета (Постеризация)
    # Сначала сильно размываем, чтобы убрать текстуру кожи/ткани
    color = cv2.bilateralFilter(img, d=9, sigmaColor=150, sigmaSpace=150)
    
    # Сокращаем количество цветов (до 8 уровней на канал)
    Z = color.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8 
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    flat_color = center[label.flatten()].reshape((img.shape))

    # 3. Накладываем контуры на плоские цвета
    # edges у нас 1 канал, переводим в 3 для наложения
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(flat_color, edges_color)

    return result