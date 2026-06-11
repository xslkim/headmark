# -*- coding: utf-8 -*-
"""实验2：额头 ROI 内放大对比 + 黑帽响应上做动态规划单曲线提取"""
import cv2
import numpy as np

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


def forehead_roi(gt_mask, img_shape, pad_frac=0.35):
    """用 GT 包围盒外扩出额头 ROI（实际系统里由 step1-3 提供）"""
    ys, xs = np.where(gt_mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h, w = img_shape[:2]
    px = int((x1 - x0) * pad_frac)
    py = int((y1 - y0) * 1.2 + 20)
    return (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))


def blackhat_resp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = img.shape[1]
    k = max(15, int(w * 0.025) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def dp_trace(resp):
    """每列选一行，最大化累计响应，相邻列行号变化有平滑惩罚"""
    resp = resp.astype(np.float32)
    H, W = resp.shape
    smooth = 1.2  # 每偏移1像素的代价
    max_jump = 4
    cost = np.full((W, H), -1e9, dtype=np.float32)
    parent = np.zeros((W, H), dtype=np.int32)
    cost[0] = resp[:, 0]
    for x in range(1, W):
        best = np.full(H, -1e9, dtype=np.float32)
        arg = np.zeros(H, dtype=np.int32)
        for d in range(-max_jump, max_jump + 1):
            shifted = np.roll(cost[x - 1], d)
            if d > 0:
                shifted[:d] = -1e9
            elif d < 0:
                shifted[d:] = -1e9
            cand = shifted - smooth * abs(d)
            upd = cand > best
            best[upd] = cand[upd]
            arg[upd] = d
        cost[x] = best + resp[:, x]
        parent[x] = arg
    ys = np.zeros(W, dtype=np.int32)
    ys[-1] = int(np.argmax(cost[-1]))
    for x in range(W - 1, 0, -1):
        ys[x - 1] = ys[x] - parent[x][ys[x]]
    return ys


for inp, out in CASES:
    img, gt_mask = load_case(inp, out)
    x0, y0, x1, y1 = forehead_roi(gt_mask, img.shape)
    crop = img[y0:y1, x0:x1]
    gt_crop = gt_mask[y0:y1, x0:x1]

    bh = blackhat_resp(img)[y0:y1, x0:x1]
    # 限制 DP 搜索在 GT 横向范围内（实际系统中用发际线 landmark 横向范围）
    gxs = np.where(gt_crop.any(axis=0))[0]
    gx0, gx1 = gxs.min(), gxs.max()
    ys = dp_trace(bh[:, gx0:gx1 + 1])

    # 评估：DP 曲线到 GT 中心线的平均距离
    gt_centers = {}
    for cx in range(gx0, gx1 + 1):
        rows = np.where(gt_crop[:, cx] > 0)[0]
        if len(rows):
            gt_centers[cx] = rows.mean()
    dists = [abs(ys[cx - gx0] - gt_centers[cx]) for cx in gt_centers]
    print(f"{inp}: DP曲线 vs GT 平均距离 {np.mean(dists):.1f}px, 中位数 {np.median(dists):.1f}px, 最大 {np.max(dists):.1f}px (ROI宽{gx1-gx0}px)")

    # 可视化：黑帽响应热力图 + GT绿 + DP红
    bh_vis = cv2.applyColorMap(cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    over = crop.copy()
    cnts, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for vis_img in (bh_vis, over):
        cv2.drawContours(vis_img, cnts, -1, (0, 255, 0), 2)
        pts = np.array([[cx + gx0, ys[cx]] for cx in range(0, gx1 - gx0 + 1)])
        cv2.polylines(vis_img, [pts], False, (0, 0, 255), 2)
    grid = np.vstack([over, bh_vis])
    scale = min(1.0, 800 / grid.shape[1])
    grid = cv2.resize(grid, None, fx=scale, fy=scale)
    out_path = f"{DIR}\\dp_{inp.split('.')[0]}.jpg"
    cv2.imwrite(out_path, grid)
    print(f"  saved {out_path}")
