# processor/effects/mode_7_chalk.py
import cv2
import numpy as np

def apply_chalk(img, w, h, out_dir, base_name):
    # Ч/б основа
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Чуть сглаживаем, чтобы штрихи были мягче
    blur = cv2.GaussianBlur(gray, (0, 0), 1.6)

    # Контур для "меловых" линий
    edges = cv2.Canny(blur, 40, 110)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    # Осветляем и слегка выцветаем картинку
    base = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    base = cv2.convertScaleAbs(base, alpha=0.88, beta=18)
    base = cv2.GaussianBlur(base, (3, 3), 0)

    # Убираем насыщенность, делая эффект ближе к мелу
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.35, 0, 255)
    base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Текстура бумаги / мела
    grain = np.random.normal(0, 10, img.shape).astype(np.int16)
    chalk = np.clip(base.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # Меловые линии — светлые края и чуть затемнённые тени
    chalk = cv2.subtract(chalk, np.full_like(chalk, 8))
    chalk = cv2.add(chalk, np.repeat(edges[:, :, None], 3, axis=2))

    return cv2.cvtColor(np.clip(chalk, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
