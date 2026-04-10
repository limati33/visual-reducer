# processor/effects/mode_34_ledscreen.py
import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _is_video_source(source_path):
    if not source_path:
        return False
    return Path(str(source_path)).suffix.lower() in VIDEO_EXTS


def _ensure_bgr(img):
    if img is None:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _to_rgb(img):
    if img is None:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _make_cell_mask(cell, led_size, border_softness=1.5):
    """
    Круглая/округлая светящаяся область внутри ячейки.
    Возвращает маску shape=(cell, cell), значения 0..1.
    """
    yy, xx = np.indices((cell, cell), dtype=np.float32)
    cx = (led_size - 1) * 0.5
    cy = (led_size - 1) * 0.5

    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)

    # радиус основной лампочки
    r = max(1.0, (led_size - 2) * 0.5)
    main = np.clip(1.0 - (dist / (r + 1e-6)) ** 1.8, 0.0, 1.0)

    # мягкое свечение вокруг
    glow_r = r + 2.5
    glow = np.clip(1.0 - (dist / (glow_r + 1e-6)) ** 2.2, 0.0, 1.0)

    # немного затемняем края самой лампочки
    edge = np.clip((main - 0.12) / 0.88, 0.0, 1.0)

    mask = (edge ** 1.35) * 0.82 + (glow ** 2.0) * 0.18
    mask = np.clip(mask, 0.0, 1.0)

    # Если led_size меньше cell, остальное останется тёмным
    if led_size < cell:
        pad = cell - led_size
        # Лампа занимает верхнюю левую часть внутри cell? Нет — центрируем:
        out = np.zeros((cell, cell), dtype=np.float32)
        start = pad // 2
        end = start + led_size
        core = mask[:led_size, :led_size]
        out[start:end, start:end] = core
        mask = out

    return mask


def _apply_led_screen(
    img,
    w=None,
    h=None,
    seed=42,
    led_size=8,
    led_gap=4,
    brightness=1.12,
    contrast=1.08,
    panel_darkness=0.16,
    scanline_strength=0.06,
    vignette_strength=0.10,
    cell_variation=0.06,
    glow_boost=0.22,
):
    """
    LED-экран без уменьшения итогового размера:
    - изображение превращается в крупную сетку светодиодов;
    - между светодиодами есть тёмные зазоры;
    - добавляется мягкое свечение, scanlines и лёгкая виньетка.
    """
    if img is None:
        return img

    bgr = _ensure_bgr(img.copy())
    ih, iw = bgr.shape[:2]

    if w is not None and h is not None and (iw != int(w) or ih != int(h)):
        bgr = cv2.resize(bgr, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        ih, iw = bgr.shape[:2]

    rng = np.random.default_rng(seed)

    led_size = max(3, int(led_size))
    led_gap = max(1, int(led_gap))
    cell = led_size + led_gap

    # Сетка по экрану
    cols = int(np.ceil(iw / cell))
    rows = int(np.ceil(ih / cell))

    # Берём средний цвет на ячейку и растягиваем обратно блоками
    small = cv2.resize(bgr, (cols, rows), interpolation=cv2.INTER_AREA)
    block = cv2.resize(small, (cols * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
    block = block[:ih, :iw].astype(np.float32)

    # Индивидуальная вариация яркости у каждой LED-ячейки
    cell_noise = rng.normal(1.0, cell_variation, (rows, cols)).astype(np.float32)
    cell_noise = np.clip(cell_noise, 0.82, 1.20)
    cell_noise = cv2.resize(cell_noise, (cols * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
    cell_noise = cell_noise[:ih, :iw]

    # Маска самой LED-лампы внутри ячейки
    cell_mask = _make_cell_mask(cell, led_size)

    # Повторяем маску по всей площади
    mask = np.tile(cell_mask, (rows, cols))
    mask = mask[:ih, :iw].astype(np.float32)

    # Небольшая “пружина” яркости по центру экрана
    yy, xx = np.indices((ih, iw), dtype=np.float32)
    cx = iw * 0.5
    cy = ih * 0.5
    dx = (xx - cx) / max(1.0, iw)
    dy = (yy - cy) / max(1.0, ih)
    dist = np.sqrt(dx * dx + dy * dy)
    vignette = 1.0 - vignette_strength * (dist ** 1.8)

    # Панельный тёмный фон
    panel = np.full((ih, iw, 3), int(255 * panel_darkness), dtype=np.float32)

    # Немного усиливаем контраст и яркость “ламп”
    lit = np.clip(block * contrast * brightness, 0, 255)

    # LED-лампа: в центре ярче, по краям темнее
    mask3 = mask[..., None] * cell_noise[..., None]
    result = panel * (1.0 - mask3) + lit * mask3

    # Дополнительное свечение: мягкий bloom на основе слегка размытых блоков
    if glow_boost > 0:
        glow = cv2.GaussianBlur(lit.astype(np.uint8), (0, 0), 1.8).astype(np.float32)
        result = cv2.addWeighted(result.astype(np.float32), 1.0, glow, glow_boost, 0)

    # Scanlines — очень умеренно, чтобы экран выглядел как настоящий LED
    if scanline_strength > 0:
        scan = np.ones((ih, 1, 1), dtype=np.float32)
        scan[1::2] = 1.0 - scanline_strength
        result *= scan

    # Лёгкая виньетка
    result *= vignette[..., None]

    # Чуть-чуть случайного шума по ячейкам, не по пикселям
    noise_cells = rng.normal(0.0, 4.0, (rows, cols)).astype(np.float32)
    noise_cells = cv2.resize(noise_cells, (cols * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
    noise_cells = noise_cells[:ih, :iw]
    result += noise_cells[..., None]

    # Очень лёгкая резкость по границам ячеек
    grid = np.zeros((ih, iw), dtype=np.float32)
    grid[:, ::cell] = 1.0
    grid[::cell, :] = 1.0
    grid = cv2.GaussianBlur(grid, (0, 0), 0.8)
    result = result * (1.0 - 0.05 * grid[..., None])

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Возвращаем RGB для PIL/save
    return _to_rgb(result)


def apply_ledscreen(
    img,
    w=None,
    h=None,
    out_dir=None,
    base_name=None,
    seed=42,
    led_size=8,
    led_gap=4,
    brightness=1.12,
    contrast=1.08,
    panel_darkness=0.16,
    scanline_strength=0.06,
    vignette_strength=0.10,
    cell_variation=0.06,
    glow_boost=0.22,
    source_path=None,
):
    _ = out_dir, base_name, source_path

    return _apply_led_screen(
        img,
        w=w,
        h=h,
        seed=seed,
        led_size=led_size,
        led_gap=led_gap,
        brightness=brightness,
        contrast=contrast,
        panel_darkness=panel_darkness,
        scanline_strength=scanline_strength,
        vignette_strength=vignette_strength,
        cell_variation=cell_variation,
        glow_boost=glow_boost,
    )
