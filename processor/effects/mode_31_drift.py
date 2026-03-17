# processor/effects/mode_31_drift.py
import cv2
import numpy as np
import os
import random

def apply_drift(img, w=None, h=None, out_dir=None, base_name=None):
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    result = img.copy().astype(np.float32)

    # Маска яркости — чтобы локально контролировать силу смазывания
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # лёгкое размытие маски, чтобы смягчить переходы
    mask_smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=6, sigmaY=6)

    # Параметры "кина"
    n_strips = max(3, int(w / 220))  # число крупных полос, зависит от ширины
    max_width = max(12, int(w * 0.08))
    min_width = max(6, int(w * 0.03))

    # базовые карты координат
    ys = np.arange(h, dtype=np.float32)

    for si in range(n_strips):
        # случайная позиция и ширина полосы
        cx = random.randint(0, w - 1)
        sw = random.randint(min_width, max_width)
        x0 = max(0, cx - sw // 2)
        x1 = min(w, x0 + sw)

        # амплитуда смещения и частота волн — больше для драматичности
        amp = random.uniform(h * 0.02, h * 0.12)
        freq = random.uniform(1.0 / 180.0, 1.0 / 80.0)  # волн по высоте
        phase = random.uniform(0, 2 * np.pi)
        tilt = random.uniform(-0.6, 0.6)  # небольшое наклонное смещение

        # envelope: сверху сильнее, снизу плавно затухает (как "сдвиг пленки")
        env = 1.0 - (ys / (h * (0.9)))  # 1..0
        env = np.clip(env, 0.0, 1.0)
        # более кинематографическое затухание
        env = 0.3 + 0.7 * (env ** 1.6)

        # волновой профиль по Y
        displacement = amp * np.sin(2.0 * np.pi * (ys * freq) + phase) * env

        # добавим небольшой линейный наклон
        displacement += (ys - h/2) * (tilt * 0.002 * amp)

        # создаём карты для remap (для этой полосы)
        strip_w = x1 - x0
        if strip_w <= 0:
            continue

        map_x = np.tile(np.arange(0, strip_w, dtype=np.float32), (h, 1))
        # map_y: для каждой строке прибавляем displacement
        map_y = np.tile(ys.reshape(h, 1), (1, strip_w)).astype(np.float32) - np.tile(displacement.reshape(h, 1), (1, strip_w))

        # remap требует координат относительно исходного strip; используем strip_src = img[:, x0:x1]
        strip_src = img[:, x0:x1]
        # приведение карт в относительную систему координат для strip_src:
        # map_x_rel = map_x (already 0..strip_w-1)
        # map_y_rel may go out of bounds — remap с borderMode=cv2.BORDER_REFLECT обрабатывает это
        remapped = cv2.remap(strip_src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # alpha маска для плавного смешивания: комбинируем env и локальную яркость
        # возьмём среднюю яркость в полосе (по исходному серому)
        local_brightness = cv2.mean(gray[:, x0:x1])[0]
        alpha_base = 0.15 + 0.55 * (1.0 - local_brightness)  # темные зоны — сильнее эффект
        # по-строчно модулируем alpha через env (чтобы сверху было сильнее)
        alpha = (0.6 * env).reshape(h, 1) * alpha_base
        alpha = np.clip(alpha, 0.0, 1.0)

        # смешиваем
        result[:, x0:x1, :] = (result[:, x0:x1, :] * (1.0 - alpha[..., None]) + remapped.astype(np.float32) * (alpha[..., None]))

    # После полос — добавим вертикальный motion blur (имитация длинной выдержки)
    # kernel длиной пропорционален высоте, но не больше 2% ширины
    mblur_len = max(3, min(45, int(h * 0.025)))
    kernel = np.zeros((mblur_len, 1), dtype=np.float32)
    # создадим 'успокаивающий' профиль: пиковая экспозиция вверху -> понижение вниз
    weights = np.linspace(1.0, 0.4, mblur_len)
    kernel[:, 0] = weights
    kernel /= kernel.sum()
    result = cv2.filter2D(result, -1, kernel)

    # Хроматическая аберрация: сдвигаем каналы по X слегка (края больше)
    shift_px = max(1, int(w * 0.0025))
    # мягкая маска краёв (чтобы аберрация сильнее по краям кадра)
    xv = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    edge_mask = np.clip(np.abs(xv) ** 1.2, 0.0, 1.0)
    edge_mask = np.tile(edge_mask.reshape(1, w), (h, 1))
    bch, gch, rch = cv2.split(result.astype(np.uint8))
    # сдвигаем R влево, B вправо (типичное поведение линз)
    r_shift = np.roll(rch, -shift_px, axis=1)
    b_shift = np.roll(bch, shift_px, axis=1)
    # interpolate original and shifted by edge_mask
    rch = (rch.astype(np.float32) * (1.0 - 0.6 * edge_mask) + r_shift.astype(np.float32) * (0.6 * edge_mask)).astype(np.uint8)
    bch = (bch.astype(np.float32) * (1.0 - 0.6 * edge_mask) + b_shift.astype(np.float32) * (0.6 * edge_mask)).astype(np.uint8)
    result = cv2.merge([bch, gch, rch]).astype(np.float32)

    # Лёгкая пленочная зернистость (grain)
    grain_strength = 6.0  # 0..20
    noise = np.random.normal(0.0, grain_strength, result.shape).astype(np.float32)
    result = np.clip(result + noise, 0, 255)

    # Небольшая кинематографическая тонировка: чуть теплее в тенях, холоднее в светах
    lab = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    Lch, ach, bch = cv2.split(lab)
    # смещаем b-channel (сине-жёлтый) в тенях
    bch = bch + (20.0 * (1.0 - (Lch / 255.0)) )
    lab = cv2.merge([Lch, ach, bch])
    result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    # Финальная легкая контрастная корректировка и коррекция насыщенности
    result = np.clip(result * 1.02 + 2.0, 0, 255)
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.03, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result.astype(np.uint8)
