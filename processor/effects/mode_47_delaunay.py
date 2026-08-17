# processor/effects/mode_47_delaunay.py
import cv2
import numpy as np

def apply_delaunay(img, w=None, h=None, out_dir=None, base_name=None,
                    n_points=800):
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)

    n_edge = min(len(xs), n_points // 2)
    if n_edge > 0:
        idx = np.random.choice(len(xs), n_edge, replace=False)
        edge_points = list(zip(xs[idx].tolist(), ys[idx].tolist()))
    else:
        edge_points = []
    random_points = [(np.random.randint(0, w_img), np.random.randint(0, h_img))
                      for _ in range(n_points - len(edge_points))]

    # гарантируем покрытие краёв и углов — вот тут была дыра
    border_points = [(0, 0), (w_img - 1, 0), (0, h_img - 1), (w_img - 1, h_img - 1)]
    for t in np.linspace(0, 1, 8):
        border_points += [
            (int(t * (w_img - 1)), 0), (int(t * (w_img - 1)), h_img - 1),
            (0, int(t * (h_img - 1))), (w_img - 1, int(t * (h_img - 1))),
        ]
    points = edge_points + random_points + border_points

    subdiv = cv2.Subdiv2D((0, 0, w_img, h_img))
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    # фон — размытая версия фото, а не чёрный: если где-то останется зазор, будет незаметно
    canvas = cv2.GaussianBlur(img, (15, 15), 0)

    for t in subdiv.getTriangleList():
        pts = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)
        pts = pts.astype(np.int32)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        color = img[cy, cx].tolist()
        cv2.fillConvexPoly(canvas, pts, color)
        cv2.polylines(canvas, [pts], True, tuple(int(c * 0.7) for c in color), 1, cv2.LINE_AA)

    return canvas