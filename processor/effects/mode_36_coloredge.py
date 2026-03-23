# processor/effects/mode_36_coloredge.py
import cv2
import numpy as np

def apply_coloredge(img, w, h, out_dir, base_name):
    ih, iw = img.shape[:2]
    # Приводим к целевому размеру сразу, если нужно
    if w and h and (iw != int(w) or ih != int(h)):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]

    # 1. Агрессивное упрощение через пирамиду
    # Чтобы избежать ValueError, принудительно ресайзим результат обратно в (iw, ih)
    small = cv2.pyrDown(img)
    small = cv2.pyrDown(small)
    small = cv2.pyrUp(small)
    small = cv2.pyrUp(small)
    # ГАРАНТИЯ РАЗМЕРА:
    small = cv2.resize(small, (iw, ih), interpolation=cv2.INTER_NEAREST)

    # 2. Убираем шум (Bilateral)
    smooth = cv2.bilateralFilter(small, d=15, sigmaColor=100, sigmaSpace=100)

    # 3. K-Means
    Z = smooth.reshape((-1, 3)).astype(np.float32)
    K = 10 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    
    # Теперь reshape точно не упадет
    quantized = res.reshape((ih, iw, 3))

    # 4. Отрисовка
    result = np.zeros((ih, iw, 3), dtype=np.uint8)
    rng = np.random.default_rng() 
    unique_colors = np.unique(centers, axis=0)

    for col in unique_colors:
        mask = cv2.inRange(quantized, col, col)
        
        # Чистим маску от "перхоти" (мелких точек)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Только внешние контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_val = int(rng.integers(0, 180))
        color_hsv = np.uint8([[[h_val, 255, 255]]])
        draw_color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

        for cnt in contours:
            # Игнорируем контуры длиной меньше 1/20 от ширины изображения
            if cv2.arcLength(cnt, True) > (iw // 20):
                cv2.drawContours(result, [cnt], -1, draw_color, 1, lineType=cv2.LINE_AA)

    return result