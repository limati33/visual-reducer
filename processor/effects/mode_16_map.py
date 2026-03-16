# processor/effects/mode_16_map.py
import cv2
from pathlib import Path

def apply_map(img, w, h, out_dir, base_name):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray8 = cv2.convertScaleAbs(gray)
    cmap = cv2.__dict__.get('COLORMAP_TERRAIN', None)
    if cmap is None:
        cmap = cv2.COLORMAP_JET
    out_main = cv2.applyColorMap(gray8, cmap)
    out_swapped = out_main[:, :, ::-1]
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        swap_path = Path(out_dir) / f"{Path(base_name).stem}_map_swapped.png"
        cv2.imwrite(str(swap_path), cv2.cvtColor(out_swapped, cv2.COLOR_RGB2BGR))
    return out_main