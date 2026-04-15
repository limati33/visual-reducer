import cv2
import numpy as np
import re

# Кэши для карт / масок по размерам
_MAP_CACHE = {}
_MASK_CACHE = {}

def _extract_frame_idx_from_basename(base_name):
    if not base_name:
        return 0
    m = re.search(r'frame[_\-]?(\d+)', base_name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0

def _get_maps(w, h):
    key = (w, h)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    _MAP_CACHE[key] = (map_x, map_y)
    return map_x, map_y

def _get_radial_mask(w, h):
    key = (w, h)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), int(np.sqrt(w * w + h * h) // 2), 1, -1)
    mask = cv2.GaussianBlur(mask, (max(1, (w // 3) | 1), max(1, (w // 3) | 1)), 0)
    _MASK_CACHE[key] = mask
    return mask

def apply_underwater(img, w=None, h=None, out_dir=None, base_name=None):

    # Сохраняем исходник, чтобы при желании можно было смешать обратно
    original = img.copy()

    # Resize
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    frame_idx = _extract_frame_idx_from_basename(base_name)
    phase = frame_idx * 0.12

    # 1) Деформация
    base_map_x, base_map_y = _get_maps(w, h)

    shift_x = (
        8.0 * np.sin(2 * np.pi * base_map_y / 150.0 + phase)
        + 4.0 * np.sin(2 * np.pi * base_map_y / 70.0 + phase * 1.4)
        + 2.0 * np.sin(2 * np.pi * base_map_x / 400.0 + phase * 0.6)
    )
    shift_y = (
        5.0 * np.cos(2 * np.pi * base_map_x / 200.0 + phase * 1.1)
        + 2.0 * np.sin(2 * np.pi * base_map_y / 300.0 + phase * 0.9)
    )

    map_x = (base_map_x + shift_x).astype(np.float32)
    map_y = (base_map_y + shift_y).astype(np.float32)

    submerged = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )

    # 2) Подводный оттенок (RGB)
    submerged = submerged.astype(np.float32)

    blue_boost = 1.10 + 0.04 * np.sin(phase * 1.3)
    green_mul = 0.88 + 0.02 * np.sin(phase * 1.1 + 1.0)
    red_mul = 0.62 + 0.02 * np.sin(phase * 0.9 + 2.0)

    # RGB: 0=R, 1=G, 2=B
    submerged[:, :, 0] *= red_mul
    submerged[:, :, 1] *= green_mul
    submerged[:, :, 2] *= blue_boost

    submerged = np.clip(submerged, 0, 255).astype(np.uint8)

    # 3) Каустика — только в яркость
    caustic_pattern = (
        np.sin((base_map_x * 0.018 + base_map_y * 0.022) + phase * 2.2)
        + 0.5 * np.sin((base_map_x * 0.035 - base_map_y * 0.012) + phase * 1.6)
    )

    caustic_norm = (caustic_pattern - caustic_pattern.min()) / (
        np.ptp(caustic_pattern) + 1e-8
    )
    caustic = (caustic_norm * 255).astype(np.uint8)

    k = max(3, (w // 20) | 1)
    caustic = cv2.GaussianBlur(caustic, (k, k), 0)
    caustic = cv2.normalize(caustic, None, 0, 45, cv2.NORM_MINMAX)

    # Работаем в LAB через RGB-конвертацию
    lab = cv2.cvtColor(submerged, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.add(l, caustic)

    lab = cv2.merge([l, a, b])
    submerged = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 4) Виньетка / глубина
    mask = _get_radial_mask(w, h)
    submerged = (
        submerged.astype(np.float32) * mask[:, :, np.newaxis]
    ).astype(np.uint8)

    # 5) Лёгкое возвращение оригинальных цветов (по желанию)
    # submerged = cv2.addWeighted(original, 0.10, submerged, 0.90, 0)

    return submerged
