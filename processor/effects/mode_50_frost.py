# processor/effects/mode_50_frost.py
import cv2
import numpy as np

def apply_frost(img, w=None, h=None, out_dir=None, base_name=None,
                 branch_density=0.35, tint=(210, 235, 255)):
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    growth_zone = np.clip(1.0 - dist / (dist.max() + 1e-6), 0, 1)  # ближе к краям — выше шанс роста льда

    # шумовое поле для "веток" кристаллов
    small = np.random.rand(h_img // 6 + 1, w_img // 6 + 1).astype(np.float32)
    field = cv2.resize(small, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=1.2)
    field_u8 = cv2.normalize(field, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # тонкие ветвящиеся линии — не заливка, а именно контурная сеть
    branches = cv2.Canny(field_u8, 80, 160).astype(np.float32) / 255.0
    branches *= growth_zone  # ветки гуще у краёв объектов, реже в глубине фона
    branches = cv2.GaussianBlur(branches, (3, 3), 0)
    branches = (branches > 0.15).astype(np.float32) * branch_density + branches * 0.3

    tint_layer = np.full_like(img, tint, dtype=np.float32)
    mask3 = np.clip(branches, 0, 1)[..., None]

    # screen-blend: осветляет, не просто подмешивает цвет — видно белые "прожилки", а не тонировку
    base = img.astype(np.float32)
    screened = 255 - (255 - base) * (255 - tint_layer) / 255
    out = base * (1 - mask3) + screened * mask3
    return np.clip(out, 0, 255).astype(np.uint8)