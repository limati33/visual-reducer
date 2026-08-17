# processor/effects/mode_45_topolines.py
import cv2
import numpy as np

def apply_topolines(img, w=None, h=None, out_dir=None, base_name=None,
                     levels=8, line_thickness=1):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape
    step = 256 // levels

    # закрашенные "слои высот" — терраса из полос, не просто линии
    band_idx = (gray // step)
    lut = cv2.applyColorMap(
        np.linspace(0, 255, levels).astype(np.uint8).reshape(1, -1),
        cv2.COLORMAP_SUMMER
    )[0]
    canvas = lut[band_idx]

    # линии-изолинии между полосами поверх заливки
    for level in range(step, 256, step):
        _, band = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (30, 30, 30), line_thickness, cv2.LINE_AA)

    return canvas.astype(np.uint8)