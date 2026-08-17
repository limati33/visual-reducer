# processor/effects/mode_44_chromaticaberration.py
import cv2
import numpy as np

def apply_chromaticaberration(img, w=None, h=None, out_dir=None, base_name=None,
                               strength=0.015):
    h_img, w_img = img.shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    Y, X = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    dx, dy = X - cx, Y - cy

    def remap_channel(channel, scale):
        map_x = cx + dx * (1 + scale)
        map_y = cy + dy * (1 + scale)
        return cv2.remap(channel, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

    r, g, b = cv2.split(img)
    r_out = remap_channel(r, strength)      # красный "распухает" наружу
    b_out = remap_channel(b, -strength)     # синий стягивается к центру
    return cv2.merge([r_out, g, b_out])