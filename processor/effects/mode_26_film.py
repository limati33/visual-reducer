# processor/effects/mode_26_film.py
import cv2
import numpy as np

def apply_film(img, w, h, out_dir, base_name):
    # 1. Цветокоррекция (Kodak Warmth)
    # Смещаем баланс: чуть меньше синего, чуть больше красного
    img_f = img.astype(np.float32)
    img_f[:, :, 0] *= 0.9  # Blue (холод)
    img_f[:, :, 1] *= 1.02 # Green (для естественности кожи)
    img_f[:, :, 2] *= 1.1  # Red (тепло)
    
    # Слегка поднимаем контраст в тенях (характерно для пленки)
    img_f = 255 * (img_f / 255)**1.1 
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    h_img, w_img = img.shape[:2]

    # 2. Умный Bloom (только для светлых участков)
    # Выделяем яркие области
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Размываем их сильно
    glow = cv2.GaussianBlur(cv2.bitwise_and(img, img, mask=mask), (25, 25), 0)
    img = cv2.addWeighted(img, 1.0, glow, 0.4, 0)

    # 3. Органическое зерно (Film Grain)
    # Вместо чистого рандома создаем "хлопья"
    noise = np.random.normal(0, 12, (h_img // 2, w_img // 2, 3)).astype(np.float32)
    noise = cv2.resize(noise, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Мягкая Виньетка (Vignette)
    kernel_x = cv2.getGaussianKernel(w_img, w_img/1.2)
    kernel_y = cv2.getGaussianKernel(h_img, h_img/1.2)
    kernel = kernel_y * kernel_x.T
    v_mask = kernel / kernel.max()
    # Делаем падение яркости не ниже 0.75 (чтобы не было черных углов)
    vignette = 0.75 + 0.25 * v_mask
    img = (img * vignette[:, :, np.newaxis]).astype(np.uint8)

    return img