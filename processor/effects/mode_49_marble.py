# processor/effects/mode_49_marble.py
import cv2
import numpy as np

def apply_marble(img, w=None, h=None, out_dir=None, base_name=None,
                  strength=25, vein_intensity=0.3):
    h_img, w_img = img.shape[:2]

    def field():
        n = np.random.randn(max(h_img // 8, 1), max(w_img // 8, 1)).astype(np.float32)
        n = cv2.resize(n, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
        return cv2.GaussianBlur(n, (0, 0), sigmaX=15)

    noise_x, noise_y = field(), field()
    map_x, map_y = np.meshgrid(np.arange(w_img, dtype=np.float32),
                                np.arange(h_img, dtype=np.float32))
    map_x += noise_x * strength
    map_y += noise_y * strength
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    noise_norm = cv2.normalize(noise_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    veins = cv2.Canny(noise_norm, 50, 100)
    veins_rgb = cv2.cvtColor(veins, cv2.COLOR_GRAY2RGB).astype(np.float32) * vein_intensity
    out = np.clip(warped.astype(np.float32) + veins_rgb, 0, 255)
    return out.astype(np.uint8)