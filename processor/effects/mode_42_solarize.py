# processor/effects/mode_42_solarize.py
import cv2
import numpy as np

def apply_solarize(img, w=None, h=None, out_dir=None, base_name=None,
                    threshold=128, blend_width=30):
    img_f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inverted = 255 - img_f
    mask = np.clip((gray - threshold) / blend_width + 0.5, 0, 1)[..., None]
    out = img_f * (1 - mask) + inverted * mask
    return out.astype(np.uint8)