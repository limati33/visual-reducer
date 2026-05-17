import cv2
import numpy as np


def apply_fire_v2(img, w=None, h=None, out_dir=None, base_name=None):

    # =========================================
    # PREP
    # =========================================

    img_f = img.astype(np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    # =========================================
    # 1. FIRE MASK
    # =========================================

    # Вместо жесткого threshold —
    # плавная температурная маска

    fire_mask = cv2.pow(gray_f, 1.8)

    fire_mask = cv2.GaussianBlur(
        fire_mask,
        (41, 41),
        0
    )

    fire_mask_3 = fire_mask[:, :, None]

    # =========================================
    # 2. TEMPERATURE MAP
    # =========================================

    # Уводим картинку в heat map

    heat_input = np.clip(
        gray_f * 255 * 1.2,
        0,
        255
    ).astype(np.uint8)

    fire_colors = cv2.applyColorMap(
        heat_input,
        cv2.COLORMAP_HOT
    ).astype(np.float32)

    # =========================================
    # 3. REMOVE BLUE / COLD
    # =========================================

    # Самая важная часть.
    # Иначе будет "кислота".

    cooled = img_f.copy()

    # подавляем синий
    cooled[:, :, 0] *= 0.25

    # немного зелёный
    cooled[:, :, 1] *= 0.65

    # красный усиливаем
    cooled[:, :, 2] *= 1.15

    # =========================================
    # 4. FIRE BLEND
    # =========================================

    # Screen-like blend

    fire_layer = (
        cooled * (1.0 - fire_mask_3 * 0.7)
        + fire_colors * fire_mask_3 * 1.6
    )

    # =========================================
    # 5. GLOW
    # =========================================

    glow = cv2.GaussianBlur(
        fire_layer,
        (0, 0),
        sigmaX=12,
        sigmaY=12
    )

    fire_layer = cv2.addWeighted(
        fire_layer,
        1.0,
        glow,
        0.35,
        0
    )

    # =========================================
    # 6. EMBERS / SPARKS
    # =========================================

    sparks = (
        np.random.rand(h or img.shape[0], w or img.shape[1])
        > 0.997
    ).astype(np.float32)

    sparks = cv2.GaussianBlur(
        sparks,
        (3, 3),
        0
    )

    sparks_rgb = np.zeros_like(img_f)

    # оранжевые искры
    sparks_rgb[:, :, 1] = sparks * 180
    sparks_rgb[:, :, 2] = sparks * 255

    # =========================================
    # 7. FINAL COMPOSITE
    # =========================================

    result = cv2.add(
        fire_layer.astype(np.float32),
        sparks_rgb.astype(np.float32)
    )

    # =========================================
    # 8. CONTRAST
    # =========================================

    result = np.clip(result, 0, 255)

    result = cv2.convertScaleAbs(
        result,
        alpha=1.08,
        beta=4
    )

    return result