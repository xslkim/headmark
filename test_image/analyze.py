# -*- coding: utf-8 -*-
"""对比不同画线检测方案的实验脚本（不依赖 mediapipe，直接全图跑后看额头区域）"""
import cv2
import numpy as np
from skimage.filters import sato

DIR = r"D:\headmark\test_image"
CASES = [("input1.png", "output1.png"), ("input3.JPG", "output3.png")]


def load_case(inp, out):
    img = cv2.imread(f"{DIR}\\{inp}")
    gt = cv2.imread(f"{DIR}\\{out}")
    if gt.shape[:2] != img.shape[:2]:
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # GT 绿线 mask
    b, g, r = cv2.split(gt.astype(np.int16))
    gt_mask = ((g > 150) & (r < 120) & (b < 120)).astype(np.uint8) * 255
    return img, gt_mask


def analyze_ink_color(img, gt_mask):
    """统计 GT 线下的像素 vs 周边皮肤像素的颜色特性"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = gt_mask > 0
    # 周边皮肤：GT 线膨胀 25px 的环带，排除线本身
    big = cv2.dilate(gt_mask, np.ones((51, 51), np.uint8)) > 0
    ring = big & ~cv2.dilate(gt_mask, np.ones((15, 15), np.uint8)).astype(bool)
    for name, region in [("ink ", ink), ("skin", ring)]:
        L, A, B = lab[..., 0][region], lab[..., 1][region], lab[..., 2][region]
        S = hsv[..., 1][region]
        g_ = gray[region]
        print(f"  {name}: gray={g_.mean():6.1f}±{g_.std():5.1f}  "
              f"L={L.mean()*100/255:5.1f} a={A.mean()-128:+6.1f} b={B.mean()-128:+6.1f}  "
              f"S={S.mean():5.1f}")


def method_global_thresh(img, threshold=115):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray < threshold).astype(np.uint8) * 255


def method_blackhat(img):
    """黑帽变换：突出比局部背景暗的细线状结构，再做相对阈值"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = img.shape[1]
    k = max(15, int(w * 0.025) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, m = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m, bh


def method_ridge(img):
    """Sato 脊线滤波（black_ridges）：响应细长暗线"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    w = img.shape[1]
    sigmas = [w * s for s in (0.002, 0.004, 0.006)]
    resp = sato(gray, sigmas=sigmas, black_ridges=True)
    resp = (resp / (resp.max() + 1e-9) * 255).astype(np.uint8)
    _, m = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m, resp


def method_adaptive(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = img.shape[1]
    blk = max(15, int(w * 0.03) | 1)
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, blk, 10)
    return m


def overlay(img, mask, gt_mask):
    """检测结果红色，GT 绿色描边"""
    vis = img.copy()
    vis[mask > 0] = (0, 0, 255)
    cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return vis


for inp, out in CASES:
    img, gt_mask = load_case(inp, out)
    print(f"\n=== {inp}  {img.shape[1]}x{img.shape[0]} ===")
    analyze_ink_color(img, gt_mask)

    m1 = method_global_thresh(img)
    m2, bh = method_blackhat(img)
    m3, ridge = method_ridge(img)
    m4 = method_adaptive(img)

    panels = [
        overlay(img, m1, gt_mask),
        overlay(img, m2, gt_mask),
        overlay(img, m3, gt_mask),
        overlay(img, m4, gt_mask),
    ]
    h, w = img.shape[:2]
    scale = 700 / w
    panels = [cv2.resize(p, (int(w * scale), int(h * scale))) for p in panels]
    labels = ["1 global<115 (current)", "2 black-hat", "3 sato ridge", "4 adaptive"]
    for p, t in zip(panels, labels):
        cv2.putText(p, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    out_path = f"{DIR}\\cmp_{inp.split('.')[0]}.jpg"
    cv2.imwrite(out_path, grid)
    print(f"  saved {out_path}")
