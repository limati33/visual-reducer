import cv2
import numpy as np

def apply_quadplates(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 40: Полупрозрачные ориентированные четырехугольники
    Оригинал удален из фона. Фигуры стакаются друг на друга.
    """
    if img is None:
        return None

    # --- Resize ---
    h0, w0 = img.shape[:2]
    if w and h and (w0 != w or h0 != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]

    # --- Цветовая сегментация ---
    shifted = cv2.pyrMeanShiftFiltering(img, sp=10, sr=20)
    lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)

    Z = lab.reshape((-1, 3)).astype(np.float32)
    K = 16  # Количество цветов

    _, labels, centers = cv2.kmeans(
        Z, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(lab.shape[:2])

    # --- Создаем чистый холст вместо оригинального фото ---
    canvas = np.zeros_like(img) # Черный фон
    # canvas = np.full_like(img, 255) # Раскомментируй, если нужен белый фон

    alpha = 0.65 # Прозрачность накладываемых фигур

    # --- Обрабатываем сегменты и стакаем фигуры ---
    for i in range(K):
        mask = (labels == i).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 40:
                continue

            # Находим габариты и поворот
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Вычисляем цвет
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            mean_bgr = cv2.mean(img, mask=cnt_mask)[:3]
            color = [int(c) for c in mean_bgr]

            # Создаем маску конкретно для этого четырехугольника
            quad_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(quad_mask, [box], 255)

            # Временный слой, на котором нарисована только одна текущая фигура
            plate = np.zeros_like(img)
            cv2.fillPoly(plate, [box], color)

            # Смешиваем текущий холст с новой фигурой с учетом прозрачности
            blended = cv2.addWeighted(plate, alpha, canvas, 1 - alpha, 0)

            # Обновляем холст ТОЛЬКО в пределах текущего четырехугольника
            canvas = np.where(quad_mask[:, :, None] > 0, blended, canvas)

    return canvas