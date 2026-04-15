import cv2
import numpy as np
import re

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

def apply_liquidmetal(img, w=None, h=None, out_dir=None, base_name=None):
    if img is None:
        return None

    # Resize при необходимости
    img_h, img_w = img.shape[:2]
    if w and h and (img_w != w or img_h != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    frame_idx = _extract_frame_idx_from_basename(base_name)
    phase = frame_idx * 0.15  # Скорость течения металла

    # ------------------------------------------------------------------
    # 1) ИСКАЖЕНИЕ: "Плавление" изображения (Displacement / Remap)
    # ------------------------------------------------------------------
    # Создаем сетку координат
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Генерируем волны для сдвига пикселей
    disp_x = np.sin(Y / 35.0 + phase * 1.5) * 8.0 + np.cos((X + Y) / 50.0 + phase) * 6.0
    disp_y = np.cos(X / 35.0 + phase * 1.5) * 8.0 + np.sin((X - Y) / 50.0 + phase) * 6.0

    map_x = np.clip(X + disp_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(Y + disp_y, 0, h - 1).astype(np.float32)

    # Искажаем оригинальное изображение (эффект текучести)
    liquid_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # Переводим в ЧБ и слегка размываем для гладкости "литой" поверхности
    gray = cv2.cvtColor(liquid_img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # ------------------------------------------------------------------
    # 2) ПСЕВДО-3D (Карта нормалей)
    #    Используем оператор Собеля для нахождения уклона поверхности
    # ------------------------------------------------------------------
    # Градиенты по осям X и Y (глубина рельефа)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5) * 2.5
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5) * 2.5
    
    # Виртуальная ось Z (направлена на нас), чтобы плоскость не была плоской
    grad_z = np.full((h, w), 0.5, dtype=np.float32)

    # Нормализация векторов (nx, ny, nz) от -1 до 1
    norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2) + 1e-5
    nx = grad_x / norm
    ny = grad_y / norm
    nz = grad_z / norm

    # ------------------------------------------------------------------
    # 3) СИНТЕТИЧЕСКОЕ ОТРАЖЕНИЕ (Chrome Environment Mapping)
    #    Хром выглядит как хром только из-за резкого "горизонта" в отражении.
    # ------------------------------------------------------------------
    # Имитируем горизонт: если нормаль смотрит вверх (ny > 0) — это небо.
    # Если вниз (ny < 0) — это темная земля.
    
    # Резкий переход (горизонт)
    horizon = np.clip((ny + 0.1) * 6.0, 0, 1)  
    
    # Яркое "небо" (сверху) и темная "земля" (снизу)
    sky = 0.7 + 0.3 * ny 
    ground = 0.1 + 0.1 * np.abs(ny)
    
    # Базовое отражение
    base_reflection = ground * (1 - horizon) + sky * horizon
    
    # Добавляем жесткие блики от источника света
    # Свет падает примерно сверху-слева
    light_dir_x, light_dir_y = 0.5, 0.5
    specular = np.clip((nx * light_dir_x + ny * light_dir_y + nz * 0.7), 0, 1)
    specular = np.power(specular, 8.0) * 1.5 # Жесткий блик
    
    # Эффект Френеля: края объектов отражают сильнее, чем центр
    fresnel = np.power(1.0 - np.clip(nz, 0, 1), 3.0) * 0.6

    # Собираем яркость хрома
    chrome_intensity = base_reflection + specular + fresnel

    # ------------------------------------------------------------------
    # 4) ТОНИРОВКА ФИНАЛЬНОГО РЕЗУЛЬТАТА
    # ------------------------------------------------------------------
    chrome = np.zeros((h, w, 3), dtype=np.float32)
    
    # Делаем металл холодным (стальным/синеватым)
    chrome[:, :, 0] = chrome_intensity * 0.85  # R
    chrome[:, :, 1] = chrome_intensity * 0.95  # G
    chrome[:, :, 2] = chrome_intensity * 1.10  # B

    # Поднимаем контраст с помощью S-образной кривой
    chrome = np.clip(chrome, 0, 1)
    chrome = chrome * chrome * (3.0 - 2.0 * chrome) # Smoothstep для контраста
    
    chrome = np.clip(chrome * 255.0, 0, 255).astype(np.uint8)

    # Легкое сглаживание, чтобы убрать возможный пиксельный шум от Собеля
    chrome = cv2.bilateralFilter(chrome, d=5, sigmaColor=30, sigmaSpace=30)

    return chrome