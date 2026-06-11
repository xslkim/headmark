# -*- coding: utf-8 -*-
"""实验3：黑帽响应 + 端点锚定最小路径（Dijkstra），处理鬓角陡降段"""
import cv2
import numpy as np
from skimage.graph import route_through_array

DIR = r"D:\headmark\test_image"
CASES = [("input1.png", "output1.png"), ("input3.JPG", "output3.png")]


def load_case(inp, out):
    img = cv2.imread(f"{DIR}\\{inp}")
    gt = cv2.imread(f"{DIR}\\{out}")
    if gt.shape[:2] != img.shape[:2]:
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    b, g, r = cv2.split(gt.astype(np.int16))
    gt_mask = ((g > 150) & (r < 120) & (b < 120)).astype(np.uint8) * 255
    return img, gt_mask


def forehead_roi(gt_mask, img_shape):
    ys, xs = np.where(gt_mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h, w = img_shape[:2]
    px = int((x1 - x0) * 0.15)
    py = int((y1 - y0) * 0.8 + 20)
    return (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))


def gt_endpoints(gt_mask):
    """GT 线最左/最右点的中心，模拟由 landmark 提供的鬓角锚点"""
    xs = np.where(gt_mask.any(axis=0))[0]
    pts = []
    for cx in (xs.min(), xs.max()):
        rows = np.where(gt_mask[:, cx] > 0)[0]
        pts.append((int(rows.mean()), int(cx)))  # (row, col)
    return pts


for inp, out in CASES:
    img, gt_mask = load_case(inp, out)
    x0, y0, x1, y1 = forehead_roi(gt_mask, img.shape)
    crop = img[y0:y1, x0:x1]
    gt_crop = gt_mask[y0:y1, x0:x1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    w_full = img.shape[1]
    k = max(15, int(w_full * 0.025) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

    # 代价图：响应越强代价越低；加 eps 防零
    cost = (bh.max() - bh) + 1.0
    (r0, c0), (r1, c1) = gt_endpoints(gt_crop)
    path, _ = route_through_array(cost, (r0, c0), (r1, c1),
                                  fully_connected=True, geometric=True)
    path = np.array(path)  # (N, 2) row,col

    # 评估：路径点到 GT 骨架的距离（用 GT mask 距离变换）
    inv = (gt_crop == 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    d = dist_map[path[:, 0], path[:, 1]]
    print(f"{inp}: 最小路径 vs GT 平均 {d.mean():.1f}px, 中位 {np.median(d):.1f}px, 最大 {d.max():.1f}px, P95 {np.percentile(d,95):.1f}px")

    vis = crop.copy()
    cnts, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    cv2.polylines(vis, [path[:, ::-1].reshape(-1, 1, 2)], False, (0, 0, 255), 2)
    scale = min(1.0, 800 / vis.shape[1])
    vis = cv2.resize(vis, None, fx=scale, fy=scale)
    out_path = f"{DIR}\\dijkstra_{inp.split('.')[0]}.jpg"
    cv2.imwrite(out_path, vis)
    print(f"  saved {out_path}")
