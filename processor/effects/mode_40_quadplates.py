import cv2
import numpy as np

def apply_quadplates(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 40: Полупрозрачные ориентированные четырехугольники по цветовым сегментам
    Вариант с более мелкими фрагментами и без подмешивания оригинала.
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

    K = 32  # больше кластеров -> мельче сегменты
    _, labels, centers = cv2.kmeans(
        Z, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(lab.shape[:2])

    # Итоговое изображение: только пластины, без оригинала
    result = np.zeros_like(img)

    # Маска для отсечения пустот
    quads_mask = np.zeros((h, w), dtype=np.uint8)

    rng = np.random.default_rng()

    # Параметры "мелкости"
    step = 6          # шаг сетки внутри сегмента
    min_size = 4      # минимальный размер пластины
    max_size = 10     # максимальный размер пластины

    # Небольшая предобработка, чтобы убрать мусор
    kernel = np.ones((3, 3), np.uint8)

    for i in range(K):
        mask = (labels == i).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 40:
                continue

            # Средний цвет сегмента
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            mean_bgr = cv2.mean(img, mask=cnt_mask)[:3]
            color = tuple(int(c) for c in mean_bgr)

            # Ограничивающий прямоугольник сегмента
            x, y, cw, ch = cv2.boundingRect(cnt)

            # Заполняем сегмент множеством мелких повернутых четырехугольников
            for py in range(y, y + ch, step):
                for px in range(x, x + cw, step):
                    if cv2.pointPolygonTest(cnt, (float(px), float(py)), False) < 0:
                        continue

                    wq = int(rng.integers(min_size, max_size + 1))
                    hq = int(rng.integers(min_size, max_size + 1))
                    angle = float(rng.uniform(0, 180))

                    rect = ((float(px), float(py)), (float(wq), float(hq)), angle)
                    box = cv2.boxPoints(rect).astype(np.int32)

                    cv2.fillPoly(result, [box], color)
                    cv2.fillPoly(quads_mask, [box], 255)

    # Если где-то остались дырки, можно слегка подстраховаться
    # Но оригинал мы не добавляем вообще
    if np.any(quads_mask == 0):
        # Можно просто оставить черный фон в пустотах
        pass

    return result