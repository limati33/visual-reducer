import cv2
import numpy as np

def apply_zdepth(img, w=None, h=None, out_dir=None, base_name=None):
    # --- Resize ---
    h0, w0 = img.shape[:2]
    if w and h and (w0 != w or h0 != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. Сохранение краев и сглаживание текстур (Bilateral) ---
    # Сохраняет резкие переходы, но убирает мелкий шум
    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # --- 2. Выделение формы и объема объекта (Distance Transform) ---
    # Находим границы Canny
    edges = cv2.Canny(smooth, 50, 150)
    
    # Инвертируем: белое — фон, черное — границы
    inv_edges = cv2.bitwise_not(edges)
    
    # Считаем расстояние от каждого пикселя до ближайшей границы.
    # Это создает эффекты «объемных» выступов внутри контуров.
    dist_transform = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 3. Локальный контраст и формы (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    volume_map = clahe.apply(smooth)

    # --- 4. Пространственный передний/задний план (Геометрия) ---
    # Предполагаем, что передний план ближе к центру/низу кадра
    Y, X = np.ogrid[:h_img, :w_img]
    cy, cx = h_img * 0.5, w_img * 0.5
    max_dist = np.sqrt(cx**2 + cy**2)
    
    # Виньетка/Затемнение заднего плана по краям
    spatial_mask = 1.0 - (np.sqrt((X - cx)**2 + (Y - cy)**2) / max_dist)
    spatial_mask = np.clip(spatial_mask, 0, 1)

    # --- 5. Сборка всех слоев ---
    # 40% объема от формата объекта + 40% геометрии расстояний + 20% пространственной маски
    combined = (volume_map.astype(np.float32) * 0.4) + \
               (dist_norm.astype(np.float32) * 0.4) + \
               (spatial_mask * 255.0 * 0.2)

    # --- 6. Финальная стилизация под Z-Depth ---
    # Нормализация
    depth = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Легкий мягкий блюр для гладкости градиентов без потери жесткости границ
    depth = cv2.edgePreservingFilter(depth, flags=1, sigma_s=60, sigma_r=0.4)

    # Приводим к 3 каналам BGR
    result = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    return result