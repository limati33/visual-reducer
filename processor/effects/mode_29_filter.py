import os
import math
import cv2
import numpy as np

def apply_filter(
    img,
    w=None, h=None,
    out_dir=None, base_name=None,
    n_seeds=120,
    downscale=2,
    cell_edge_blur=3,
    seed_jitter=0.12,
    color_smooth=0.85,
    radial_strength=0.7,
    shift_amount=6,
    paper_noise=6,
    return_multiple=False,
    save_debug=False
):
    if img is None:
        return None

    ih0, iw0 = img.shape[:2]
    if w and h and (iw0 != w or ih0 != h):
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    ih, iw = img.shape[:2]

    # downscale
    ds = max(1, int(downscale))
    ws, hs = max(1, iw // ds), max(1, ih // ds)
    # make sure we have at least 1x1
    ws = max(1, ws); hs = max(1, hs)
    small = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
    small_f = small.astype(np.float32)

    # grey normalized
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # good corners
    max_corners = max(20, int(n_seeds * 0.6))
    corners = cv2.goodFeaturesToTrack(gray_norm, maxCorners=max_corners, qualityLevel=0.01, minDistance=8)
    seeds = []
    if corners is not None:
        for c in corners.reshape(-1,2):
            seeds.append((int(c[0]), int(c[1])))

    # RNG
    rng = np.random.default_rng(12345)
    while len(seeds) < n_seeds:
        x = int(rng.integers(0, ws))
        y = int(rng.integers(0, hs))
        prob = (gray_norm[y, x] / 255.0) * 0.6 + 0.2
        if rng.random() < prob:
            seeds.append((x, y))
    seeds = seeds[:n_seeds]

    # jitter
    pts = np.array(seeds, dtype=np.float32)
    if len(pts) > 1 and seed_jitter > 0:
        # compute avg_dist on some neighbour pairs
        dists = []
        ncheck = min(200, len(pts)-1)
        for i in range(ncheck):
            a = pts[i]
            b = pts[(i+1) % len(pts)]
            dists.append(np.hypot(a[0]-b[0], a[1]-b[1]))
        avg_dist = max(1.0, float(np.mean(dists)) if dists else math.hypot(ws, hs)/10.0)
        jitter_px = seed_jitter * avg_dist
        jitter = rng.normal(0, jitter_px, size=pts.shape).astype(np.float32)
        pts = np.clip(pts + jitter, [0,0], [ws-1, hs-1])

    k = len(pts)

    # ===== memory saver: compute Voronoi assignment incrementally (no big dist2) =====
    xs, ys = np.meshgrid(np.arange(ws, dtype=np.float32), np.arange(hs, dtype=np.float32))
    # min distance initialized to +inf
    min_dist = np.full((hs, ws), np.inf, dtype=np.float32)
    idx = np.full((hs, ws), -1, dtype=np.int32)

    for i in range(k):
        sx = float(pts[i,0]); sy = float(pts[i,1])
        dx = xs - sx
        dy = ys - sy
        d2 = dx*dx + dy*dy  # (hs,ws) float32
        mask_update = d2 < min_dist
        if np.any(mask_update):
            # update where this seed is closer
            min_dist[mask_update] = d2[mask_update]
            idx[mask_update] = i

    # pre-split channels once
    bch = cv2.split(small)  # uint8 channels
    # prepare accumulators in float32
    result_small = np.zeros_like(small_f, dtype=np.float32)  # what we add as coloured blocks
    blended_small = small_f.copy()  # original reduced (will be multiplied-down in-place)

    # Precompute noise (float32)
    noise = rng.normal(size=(hs, ws)).astype(np.float32)
    # Blur noise
    sigma_noise = max(0.5, ws * 0.02)
    noise = cv2.GaussianBlur(noise, (0,0), sigma_noise)
    # normalize to -1..1
    nmin, nmax = float(noise.min()), float(noise.max())
    if nmax - nmin > 1e-6:
        noise = 2.0*(noise - nmin)/(nmax - nmin) - 1.0
    else:
        noise.fill(0.0)

    # For each cell: compute mask, dominant color, radial, and accumulate using in-place ops
    for i in range(k):
        mask_bool = (idx == i)
        if not mask_bool.any():
            continue
        # minimal area check
        if mask_bool.sum() < 4:
            continue
        # mask blurred for soft edge
        if cell_edge_blur > 0:
            # convert to float32 then blur
            mask_f = cv2.GaussianBlur(mask_bool.astype(np.float32), (cell_edge_blur*2+1, cell_edge_blur*2+1), 0)
        else:
            mask_f = mask_bool.astype(np.float32)

        sx = int(round(pts[i,0])); sy = int(round(pts[i,1]))
        sx = max(0, min(ws-1, sx)); sy = max(0, min(hs-1, sy))

        # dominant color (median per channel) — operate on original uint8 channels
        dom = []
        for ch in bch:
            vals = ch[mask_bool]
            if vals.size == 0:
                # fallback to channel median
                dom.append(float(np.median(ch)))
            else:
                dom.append(float(np.median(vals)))
        dom = np.array(dom, dtype=np.float32)  # B,G,R

        # radial vignette (float32)
        dx = xs - float(sx)
        dy = ys - float(sy)
        r = np.sqrt(dx*dx + dy*dy)
        rmax = max(1.0, math.sqrt((ws**2 + hs**2)) * 0.035)
        radial = np.exp(- (r / (rmax*(1.0 + radial_strength)))**2).astype(np.float32)

        # factor per-pixel for this cell (scalar mask_f * radial * radial_strength)
        factor = mask_f * radial * float(radial_strength)  # (hs,ws) float32

        # add color contribution in-place: result_small += factor[:,:,None] * dom[None,None,:]
        # prepare small 3-channel factor broadcasted without allocating huge temp:
        # we'll compute per-channel in a small loop to reduce temporaries
        for c in range(3):
            # result_small[:,:,c] += factor * dom[c]
            np.add(result_small[:,:,c], factor * dom[c], out=result_small[:,:,c])

        # reduce original contribution inside this mask (in-place)
        # alpha = mask_f[:,:,None] * (1.0 - color_smooth)
        alpha = (mask_f * (1.0 - color_smooth)).astype(np.float32)  # (hs,ws)
        # multiply blended_small by (1 - alpha) per channel, in-place
        one_minus_alpha = (1.0 - alpha)
        for c in range(3):
            np.multiply(blended_small[:,:,c], one_minus_alpha, out=blended_small[:,:,c])

    # compose
    composed = blended_small + result_small
    # clip and to uint8
    composed = np.clip(composed, 0, 255).astype(np.uint8)

    # local shifts / remap using noise
    shift_map_x = (noise * shift_amount).astype(np.float32)
    shift_map_y = (cv2.GaussianBlur(noise, (0,0), max(0.5, ws*0.01)) * shift_amount * 0.6).astype(np.float32)
    xs_f, ys_f = np.meshgrid(np.arange(ws, dtype=np.float32), np.arange(hs, dtype=np.float32))
    map_x = (xs_f + shift_map_x).astype(np.float32)
    map_y = (ys_f + shift_map_y).astype(np.float32)
    remapped = cv2.remap(composed, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # paper texture
    paper = (rng.normal(loc=0.0, scale=max(0.0001, paper_noise), size=remapped.shape).astype(np.float32))
    paper = cv2.GaussianBlur(paper, (0,0), max(0.5, ws*0.005))
    textured = np.clip(remapped.astype(np.float32) + paper, 0, 255).astype(np.uint8)

    # upscale
    if ds != 1:
        final = cv2.resize(textured, (iw, ih), interpolation=cv2.INTER_LINEAR)
    else:
        final = textured

    # final color tweak (CLAHE)
    lab = cv2.cvtColor(final, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8))
    l = clahe.apply(np.clip(l,0,255).astype(np.uint8)).astype(np.float32)
    lab = cv2.merge([l, a, b]).astype(np.uint8)
    final = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if save_debug and out_dir and base_name:
        try:
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{base_name}_mode29_orgmosaic.png"), final)
        except Exception:
            pass

    return final
