"""
Baseline test: run the processor on image/input1.png and compare with image/output1.png.
Outputs image/result_step4.jpg, image/result_contour.png, image/result_hull.png.
"""

import os
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cv2
import numpy as np
from processor import HeadmarkProcessor

INPUT = "image/input1.png"
GT = "image/output1.png"


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def green_to_binary(img_bgr: np.ndarray) -> np.ndarray:
    """Convert the green-painted ground truth to a binary mask."""
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    return ((g > 200) & (b < 100) & (r < 100)).astype(np.uint8) * 255


def load_api_key() -> str:
    key = os.environ.get("DOUBAO_API_KEY", "").strip()
    if key:
        return key
    f = pathlib.Path(__file__).parent / "apikey.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


def main():
    proc = HeadmarkProcessor()
    image_bytes = pathlib.Path(INPUT).read_bytes()
    result = proc.process(
        image_bytes,
        threshold=115,
        dilation_pct=1.0,
        doubao_api_key=load_api_key() or None,
    )
    print(f"VLM status: {result.get('vlm_status')}")

    gt_img = cv2.imread(GT)
    gt_mask_thin = green_to_binary(gt_img)

    # Dilate GT line so it can be compared against filled region masks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    gt_mask_dilated = cv2.dilate(gt_mask_thin, k)

    print(
        f"Contour mask IoU (vs dilated GT): {iou(proc._contour_mask, gt_mask_dilated):.3f}"
    )
    print(
        f"Hull mask IoU    (vs dilated GT): {iou(proc._hull_mask, gt_mask_dilated):.3f}"
    )

    # Save outputs for visual inspection
    cv2.imwrite("image/result_contour.png", proc._contour_mask)
    cv2.imwrite("image/result_hull.png", proc._hull_mask)
    cv2.imwrite("image/result_step4.jpg", cv2.imread("image/input1.png"))  # placeholder

    # Overlay contour mask on original
    orig = cv2.imread(INPUT)
    overlay = orig.copy()
    overlay[proc._contour_mask > 0] = [0, 255, 255]
    cv2.addWeighted(orig, 0.5, overlay, 0.5, 0, overlay)
    cv2.imwrite("image/result_overlay.jpg", overlay)

    print("Saved: image/result_contour.png, result_hull.png, result_overlay.jpg")


if __name__ == "__main__":
    main()
