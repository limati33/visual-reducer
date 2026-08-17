import cv2
import numpy as np

def _shift_image(img, dx, dy):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def _add_texture(img, strength=0.08):
    h, w = img.shape[:2]

    noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 3)

    noise3 = cv2.merge([noise, noise, noise])

    tex = img.astype(np.float32) / 255.0
    tex = np.clip(tex + noise3 * strength, 0, 1)

    return (tex * 255).astype(np.uint8)

def _make_drips(mask, max_drips=180):
    """Подтёки вниз от нижних границ маски."""
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros_like(mask)

    # Берём точки у нижней части объекта
    y_threshold = np.percentile(ys, 70)
    candidate_idx = np.where(ys >= y_threshold)[0]
    if len(candidate_idx) == 0:
        candidate_idx = np.arange(len(xs))

    drip_layer = np.zeros((h, w), dtype=np.uint8)
    n = min(max_drips, len(candidate_idx))

    chosen = np.random.choice(candidate_idx, size=n, replace=False)
    for i in chosen:
        x = int(xs[i])
        y = int(ys[i])

        length = np.random.randint(h // 40, h // 10)
        width = np.random.randint(1, 4)
        intensity = np.random.randint(110, 230)

        # Вертикальный подтёк
        y2 = min(h - 1, y + length)
        cv2.line(drip_layer, (x, y), (x, y2), intensity, width)

        # Капля внизу
        if np.random.rand() < 0.7:
            r = np.random.randint(1, 4)
            cv2.circle(drip_layer, (x, y2), r, 255, -1)

    drip_layer = cv2.GaussianBlur(drip_layer, (5, 5), 0)
    return drip_layer

def _make_splatter(edges, amount=1200):
    """Брызги краски вокруг контуров."""
    h, w = edges.shape[:2]
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    splatter = np.zeros((h, w), dtype=np.uint8)
    n = min(amount, len(xs) * 2)

    # Чуть расширим область, откуда летят капли
    idx = np.random.choice(len(xs), size=n, replace=True)
    for i in idx:
        x = int(xs[i] + np.random.randint(-12, 13))
        y = int(ys[i] + np.random.randint(-12, 13))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        # Мелкие и крупные капли
        if np.random.rand() < 0.85:
            r = np.random.randint(1, 3)
        else:
            r = np.random.randint(3, 7)

        val = np.random.randint(120, 255)
        cv2.circle(splatter, (x, y), r, val, -1)

        # Иногда "хвостик" от брызги
        if np.random.rand() < 0.25:
            dx = np.random.randint(-6, 7)
            dy = np.random.randint(2, 12)
            cv2.line(splatter, (x, y), (x + dx, y + dy), val, 1)

    splatter = cv2.GaussianBlur(splatter, (3, 3), 0)
    return splatter

def apply_graffiti(img, w=None, h=None, out_dir=None, base_name=None):
    """
    Mode 41: Граффити / стрит-арт.
    Добавляет: слоистую заливку, spray-outline, брызги, подтёки и текстуру стены.
    """
    # --- Resize ---
    h0, w0 = img.shape[:2]
    if w and h and (w0 != w or h0 != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]

    # --- 1. Сглаживание + квантование цветов ---
    blurred = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    Z = blurred.reshape((-1, 3)).astype(np.float32)
    K = 8
    _, labels, centers = cv2.kmeans(
        Z, K, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    flat = centers[labels.flatten()].reshape(img.shape)

    # --- 2. Эджи для граффити-контуров ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    kernel = np.ones((3, 3), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=2)
    soft_edges = cv2.GaussianBlur(thick_edges, (5, 5), 0)

    # --- 3. Основные слои краски ---
    # Несколько слегка смещённых слоёв, чтобы было ощущение нанесения краски
    layer1 = flat.copy()
    layer2 = _shift_image(flat, np.random.randint(-2, 3), np.random.randint(-2, 3))
    layer3 = _shift_image(flat, np.random.randint(-3, 4), np.random.randint(-3, 4))

    # Лёгкая вариация яркости по слоям
    layer2 = cv2.convertScaleAbs(layer2, alpha=1.03, beta=np.random.randint(-8, 9))
    layer3 = cv2.convertScaleAbs(layer3, alpha=0.97, beta=np.random.randint(-10, 11))

    # Мягко смешиваем слои
    paint = cv2.addWeighted(layer1, 0.62, layer2, 0.23, 0)
    paint = cv2.addWeighted(paint, 0.80, layer3, 0.20, 0)

    # --- 4. Контур spray-paint ---
    noise = np.random.randint(0, 255, soft_edges.shape, dtype=np.uint8)
    spray_mask = np.where(soft_edges > noise, 255, 0).astype(np.uint8)

    # Добавим чуть более плотную внутреннюю линию, чтобы контур читался
    core = cv2.dilate(edges, kernel, iterations=1)
    core = cv2.GaussianBlur(core, (3, 3), 0)

    outline = np.zeros_like(paint)
    outline[spray_mask > 0] = (20, 20, 20)
    outline[core > 50] = (10, 10, 10)

    result = np.where(spray_mask[:, :, None] > 0, outline, paint).astype(np.uint8)

    # --- 5. Брызги краски ---
    splatter = _make_splatter(edges, amount=max(800, (w * h) // 250))
    splatter_3c = cv2.cvtColor(splatter, cv2.COLOR_GRAY2BGR)

    # Разные оттенки брызг: часть тёмные, часть цветные
    splatter_color = result.copy().astype(np.float32)
    random_tint = np.random.randint(-20, 21, size=splatter_3c.shape).astype(np.float32)
    splatter_color = np.clip(splatter_color + random_tint * (splatter_3c > 0), 0, 255)

    alpha = (splatter_3c.astype(np.float32) / 255.0) * 0.8
    result = (result.astype(np.float32) * (1 - alpha) + splatter_color * alpha).astype(np.uint8)

    # --- 6. Подтёки вниз ---
    drips = _make_drips(thick_edges, max_drips=max(100, (w * h) // 5000))
    drips = cv2.cvtColor(drips, cv2.COLOR_GRAY2BGR)

    # Подтёки делаем более тёмными и слегка цветными
    drip_color = result.copy().astype(np.float32)
    drip_color = np.clip(drip_color * 0.65 + np.random.randint(-10, 11, drip_color.shape), 0, 255)
    alpha = (drips.astype(np.float32) / 255.0) * 0.9
    result = (result.astype(np.float32) * (1 - alpha) + drip_color * alpha).astype(np.uint8)

    # --- 7. Текстура стены / зерно ---
    result = _add_texture(result, strength=0.045)

    # --- 8. Финальная смычка ---
    result = cv2.medianBlur(result, 3)

    return result