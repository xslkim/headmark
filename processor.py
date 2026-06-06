import os
from typing import Optional

import base64
import io
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# MediaPipe face mesh landmark indices for forehead boundary
LANDMARK_INDICES = [21, 68, 104, 69, 108, 151, 337, 299, 333, 298, 251]

# Half-height of the DP search band around the VLM polyline, as a fraction of image
# height. The marker line is followed within ±this many rows of the VLM prior.
_VLM_BAND_FRAC = 0.030

# Black-hat structuring-element size (px). Highlights dark lines thinner than this,
# regardless of whether the local background is light skin or dark hair.
_BLACKHAT_KSIZE = 15

# Black-hat response above this value is shown as detected ink in the step-4 overlay.
_BLACKHAT_THRESH = 20

# DP ridge-follow weights: deviation-from-VLM-prior cost, and roughness (smoothness)
# cost per row of vertical step between adjacent columns. Tuned on植发 sample images.
_DP_PRIOR_WEIGHT = 0.8
_DP_SMOOTH_WEIGHT = 3.0

# Moving-average half-window (px) used to smooth the raw DP path.
_DP_SMOOTH_WINDOW = 9

# Minimum local-darkness difference for the no-VLM fallback detector.
_LOCAL_DARK_DELTA = 20

# Gaussian blur kernel for the no-VLM fallback local-darkness detector.
_LOCAL_BLUR_KSIZE = 25


class HeadmarkProcessor:
    def __init__(self) -> None:
        face_base = mp.tasks.BaseOptions(
            model_asset_path=os.path.join(MODEL_DIR, "face_landmarker.task")
        )
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=face_base,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            face_options
        )

        seg_base = mp.tasks.BaseOptions(
            model_asset_path=os.path.join(MODEL_DIR, "selfie_segmenter.tflite")
        )
        seg_options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=seg_base,
            output_confidence_masks=True,
        )
        self.selfie_seg = mp.tasks.vision.ImageSegmenter.create_from_options(
            seg_options
        )

        self.use_head_seg = self._check_head_seg_available()
        if self.use_head_seg:
            print("[INFO] head-segmentation 可用，将通过子进程调用")
        else:
            print("[INFO] head-segmentation 不可用，使用 MediaPipe Selfie Segmentation")

        self._image: Optional[np.ndarray] = None
        self._image_rgb: Optional[np.ndarray] = None
        self._upper_mask: Optional[np.ndarray] = None
        self._head_mask: Optional[np.ndarray] = None
        self._intersection: Optional[np.ndarray] = None
        self._black_mask: Optional[np.ndarray] = None
        self._contour_mask: Optional[np.ndarray] = None
        self._hull_mask: Optional[np.ndarray] = None
        # Cached VLM polyline: list of [x, y] in normalized [0, 1] coords, left → right
        self._vlm_points: Optional[list] = None
        # Fitted marker curve (Nx2 int32 px points) when VLM-guided path runs
        self._fitted_curve: Optional[np.ndarray] = None

    def process(
        self,
        image_bytes: bytes,
        threshold: int = 115,
        dilation_pct: float = 1.0,
        doubao_api_key: Optional[str] = None,
    ) -> dict:
        """Process uploaded image through all 5 steps."""
        self._load_image(image_bytes)

        vis1 = self._step1()
        vis2 = self._step2()
        vis3 = self._step3()

        # Optional VLM-guided ROI refinement (called once per image upload)
        vlm_status = self._run_vlm(doubao_api_key)

        threshold_results = self._process_threshold(threshold, dilation_pct)

        w = self._image.shape[1]
        dilation_px = max(1, int(w * dilation_pct / 100))

        return {
            "original": self._encode(self._image),
            "step1": self._encode(vis1),
            "step2": self._encode(vis2),
            "step3": self._encode(vis3),
            "seg_mode": (
                "head-segmentation" if self.use_head_seg else "selfie-segmentation"
            ),
            "vlm_status": vlm_status,
            "image_width": w,
            "dilation_px": dilation_px,
            **threshold_results,
        }

    def adjust(self, threshold: int, dilation_pct: float = 1.0) -> dict:
        """Re-process steps 4–5 with new threshold/dilation (reuses cached steps 1–3 and VLM bbox)."""
        if self._image is None:
            raise ValueError("请先上传图片")
        return self._process_threshold(threshold, dilation_pct)

    def get_mask(self, mask_type: str) -> bytes:
        mask = self._contour_mask if mask_type == "contour" else self._hull_mask
        if mask is None:
            raise ValueError("请先处理图片")
        _, buffer = cv2.imencode(".png", mask)
        return buffer.tobytes()

    # ── Internal methods ──

    def _run_vlm(self, api_key: Optional[str]) -> str:
        """Call Doubao Vision to trace the marker line as a polyline.
        Caches result in self._vlm_points. Returns a status string for the UI."""
        self._vlm_points = None
        if not api_key:
            return "disabled"

        try:
            from doubao_vision import detect_marker
        except ImportError:
            return "module_missing"

        result = detect_marker(self._image, api_key)
        if result is None:
            return "api_error"
        if not result.get("has_marker"):
            return "no_marker_detected"

        self._vlm_points = result.get("points")
        return "ok" if self._vlm_points else "no_points"

    @staticmethod
    def _check_head_seg_available() -> bool:
        try:
            import importlib.util

            spec = importlib.util.find_spec("head_segmentation")
            if spec is None:
                return False
            pkg_dir = None
            if spec.submodule_search_locations:
                pkg_dir = spec.submodule_search_locations[0]
            elif spec.origin:
                pkg_dir = os.path.dirname(spec.origin)
            if pkg_dir is None:
                return False
            return os.path.exists(
                os.path.join(pkg_dir, "model", "head_segmentation.ckpt")
            )
        except Exception:
            return False

    @staticmethod
    def _run_head_seg_subprocess(image_rgb: np.ndarray) -> np.ndarray:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            input_path = f.name
            np.save(f, image_rgb)

        output_path = input_path + ".out.npy"
        script = f"""
import numpy as np
from head_segmentation.segmentation_pipeline import HumanHeadSegmentationPipeline
img = np.load("{input_path}")
pipeline = HumanHeadSegmentationPipeline()
seg_map = pipeline.predict(img)
np.save("{output_path}", seg_map)
"""
        try:
            result = subprocess.run(
                ["python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"head-segmentation subprocess failed: {result.stderr}"
                )
            return np.load(output_path)
        finally:
            for p in [input_path, output_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def _load_image(self, image_bytes: bytes) -> None:
        pil = Image.open(io.BytesIO(image_bytes))
        pil = ImageOps.exif_transpose(pil)
        self._image_rgb = np.array(pil.convert("RGB"))
        self._image = cv2.cvtColor(self._image_rgb, cv2.COLOR_RGB2BGR)

    def _step1(self) -> np.ndarray:
        """MediaPipe face landmarks → upper-forehead region mask."""
        h, w = self._image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._image_rgb)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise ValueError("未检测到人脸，请上传包含清晰人脸的照片")

        face_lms = result.face_landmarks[0]
        points = [
            (int(face_lms[idx].x * w), int(face_lms[idx].y * h))
            for idx in LANDMARK_INDICES
        ]

        left_ext = (0, points[0][1])
        right_ext = (w - 1, points[-1][1])
        polygon = np.array(
            [left_ext] + points + [right_ext, (w - 1, 0), (0, 0)],
            dtype=np.int32,
        )

        self._upper_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._upper_mask, [polygon], 255)

        vis = self._image.copy().astype(np.float32)
        vis[self._upper_mask == 0] *= 0.3
        green = np.zeros_like(vis)
        green[self._upper_mask > 0] = [0, 60, 0]
        vis = np.clip(vis + green, 0, 255).astype(np.uint8)

        line_pts = [left_ext] + points + [right_ext]
        for i in range(len(line_pts) - 1):
            cv2.line(vis, line_pts[i], line_pts[i + 1], (0, 255, 0), 2)
        for i, pt in enumerate(points):
            cv2.circle(vis, pt, 5, (0, 0, 255), -1)
            cv2.putText(
                vis,
                str(LANDMARK_INDICES[i]),
                (pt[0] + 7, pt[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
        return vis

    def _step2(self) -> np.ndarray:
        """Head/body segmentation."""
        if self.use_head_seg:
            seg_map = self._run_head_seg_subprocess(self._image_rgb)
            self._head_mask = ((seg_map > 0) * 255).astype(np.uint8)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._image_rgb)
            result = self.selfie_seg.segment(mp_image)
            confidence = np.squeeze(result.confidence_masks[0].numpy_view())
            self._head_mask = ((confidence > 0.5) * 255).astype(np.uint8)

        vis = self._image.copy().astype(np.float32)
        vis[self._head_mask == 0] *= 0.3
        tint = np.zeros_like(vis)
        tint[self._head_mask > 0] = [0, 0, 60]
        vis = np.clip(vis + tint, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            self._head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
        return vis

    def _step3(self) -> np.ndarray:
        """Intersection of upper region and head segmentation."""
        self._intersection = cv2.bitwise_and(self._upper_mask, self._head_mask)

        vis = self._image.copy().astype(np.float32)
        vis[self._intersection == 0] *= 0.3
        tint = np.zeros_like(vis)
        tint[self._intersection > 0] = [60, 60, 0]
        vis = np.clip(vis + tint, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            self._intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
        return vis

    def _process_threshold(self, threshold: int, dilation_pct: float) -> dict:
        vis4 = self._step4(threshold, dilation_pct)
        vis5a, vis5b = self._step5()
        return {
            "step4": self._encode(vis4),
            "step5a": self._encode(vis5a),
            "step5b": self._encode(vis5b),
            "contour_mask": self._encode(self._contour_mask, fmt=".png"),
            "hull_mask": self._encode(self._hull_mask, fmt=".png"),
        }

    def _step4(self, threshold: int, dilation_pct: float = 1.0) -> np.ndarray:
        """Detect the marker line.

        Preferred path (VLM available): snap a smooth curve to the actual ink
        pixels inside a narrow band around the Doubao polyline. Falls back to a
        local-darkness threshold detector when no VLM polyline is available.
        """
        self._fitted_curve = None
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

        if self._vlm_points and len(self._vlm_points) >= 2:
            vis = self._step4_vlm_guided(gray, dilation_pct)
            if vis is not None:
                return vis

        return self._step4_fallback(gray, threshold, dilation_pct)

    def _step4_vlm_guided(
        self, gray: np.ndarray, dilation_pct: float
    ) -> Optional[np.ndarray]:
        """Follow the marker ink ridge within a band around the VLM polyline.

        The VLM polyline localizes the line semantically but jitters vertically
        by ±15–20 px between calls, so we don't trust its y directly. Instead a
        black-hat transform lights up the dark stroke (on skin *or* hair), and a
        dynamic-programming pass finds the lowest-cost continuous path through
        that ink — rewarding strong ink, lightly penalising deviation from the
        VLM prior and vertical roughness. This locks onto the real stroke and is
        robust to both hair-edge noise and VLM jitter.

        Returns the step-4 visualization, or None to signal a fall back (e.g. the
        polyline degenerated to a single column)."""
        h, w = gray.shape
        px = np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in self._vlm_points],
            dtype=np.int32,
        )
        x_lo, x_hi = int(px[:, 0].min()), int(px[:, 0].max())
        if x_hi - x_lo < 10:
            return None

        # Black-hat highlights dark lines thinner than the kernel, on light skin
        # OR dark hair alike — the key to "marker is the same grey as background".
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_BLACKHAT_KSIZE, _BLACKHAT_KSIZE)
        )
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se).astype(np.float32)

        xf = np.arange(x_lo, x_hi)
        prior_y = np.interp(xf, px[:, 0], px[:, 1]).astype(np.int32)
        band_half = max(10, int(h * _VLM_BAND_FRAC))

        yf = self._dp_follow_ridge(blackhat, xf, prior_y, band_half, h)
        yf = self._smooth_path(yf, _DP_SMOOTH_WINDOW)
        yf = np.clip(yf, 0, h - 1)

        curve = np.stack([xf, yf], axis=1).astype(np.int32)
        self._fitted_curve = curve

        # Region mask = curve dilated to the requested thickness.
        radius = max(1, int(w * dilation_pct / 100))
        self._black_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(self._black_mask, [curve], False, 255, thickness=radius * 2 + 1)

        # Visualization: dim band, show ink in amber, fitted curve in red.
        band = np.zeros((h, w), dtype=np.uint8)
        prior_curve = np.stack([xf, prior_y], axis=1).astype(np.int32)
        cv2.polylines(band, [prior_curve], False, 255, thickness=band_half * 2)
        ink = cv2.bitwise_and(
            (blackhat > _BLACKHAT_THRESH).astype(np.uint8) * 255, band
        )
        vis = self._image.copy()
        vis[band > 0] = (vis[band > 0].astype(np.float32) * 0.5).astype(np.uint8)
        vis[ink > 0] = [0, 200, 255]
        cv2.polylines(vis, [curve], False, (0, 0, 255), 2)
        return vis

    @staticmethod
    def _dp_follow_ridge(
        blackhat: np.ndarray,
        xf: np.ndarray,
        prior_y: np.ndarray,
        band_half: int,
        h: int,
    ) -> np.ndarray:
        """Min-cost path (left→right) through the black-hat ink near the prior.

        Cost per pixel = -ink_strength + prior_weight*|offset|; transitions add
        smooth_weight*|Δrow|. Returns absolute y per column in xf."""
        offsets = np.arange(-band_half, band_half + 1)
        n_col, n_row = len(xf), len(offsets)

        # Per-column node cost: low where ink is strong and near the prior.
        node_cost = np.empty((n_col, n_row), dtype=np.float32)
        for j in range(n_col):
            yy = prior_y[j] + offsets
            valid = (yy >= 0) & (yy < h)
            strength = np.where(valid, blackhat[np.clip(yy, 0, h - 1), xf[j]], 0.0)
            node_cost[j] = -strength + _DP_PRIOR_WEIGHT * np.abs(offsets)

        # Transition penalty matrix: smooth_weight * |row_i - row_k|.
        step_pen = _DP_SMOOTH_WEIGHT * np.abs(offsets[:, None] - offsets[None, :])

        dp = node_cost[0].copy()
        back = np.zeros((n_col, n_row), dtype=np.int32)
        for j in range(1, n_col):
            total = dp[None, :] + step_pen  # (row_k, row_prev)
            best_prev = np.argmin(total, axis=1)
            dp = node_cost[j] + total[np.arange(n_row), best_prev]
            back[j] = best_prev

        path = np.empty(n_col, dtype=np.int32)
        path[-1] = int(np.argmin(dp))
        for j in range(n_col - 1, 0, -1):
            path[j - 1] = back[j, path[j]]
        return prior_y + offsets[path]

    @staticmethod
    def _smooth_path(y: np.ndarray, half_window: int) -> np.ndarray:
        """Moving-average smoothing that preserves the curve's overall shape."""
        k = half_window * 2 + 1
        if len(y) < k:
            return y.astype(np.float64)
        kernel = np.ones(k) / k
        padded = np.pad(y.astype(np.float64), half_window, mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    def _step4_fallback(
        self, gray: np.ndarray, threshold: int, dilation_pct: float = 1.0
    ) -> np.ndarray:
        """No-VLM detector: adaptive threshold + local-darkness within the head ROI."""
        h, w = gray.shape
        roi_mask = self._intersection.copy()
        roi = roi_mask > 0

        # Adaptive absolute threshold (marker on skin).
        roi_pixels = gray[roi]
        if len(roi_pixels) > 100:
            skin_level = float(np.percentile(roi_pixels, 75))
            adaptive_thresh = int(skin_level * 0.58)
            effective_threshold = max(min(threshold, adaptive_thresh), 40)
        else:
            effective_threshold = threshold
        dark_abs = roi & (gray < effective_threshold)

        # Local-darkness (marker on dark hair).
        ksize = _LOCAL_BLUR_KSIZE | 1
        gray_blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        diff = gray_blur.astype(np.int16) - gray.astype(np.int16)
        dark_local = roi & (diff > _LOCAL_DARK_DELTA)

        combined = (dark_abs | dark_local).astype(np.uint8) * 255

        dilation_radius = max(1, int(w * dilation_pct / 100))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_radius * 2 + 1, dilation_radius * 2 + 1),
        )
        self._black_mask = cv2.dilate(combined, kernel)
        self._black_mask = cv2.bitwise_and(self._black_mask, roi_mask)

        vis = self._image.copy()
        vis[roi_mask == 0] = (vis[roi_mask == 0].astype(np.float32) * 0.3).astype(
            np.uint8
        )
        vis[self._black_mask > 0] = [0, 0, 255]
        return vis

    def _step5(self) -> tuple[np.ndarray, np.ndarray]:
        """Build contour and convex-hull masks with shape-aware component selection."""
        h, w = self._image.shape[:2]

        if np.sum(self._black_mask > 0) < 50:
            self._contour_mask = np.zeros((h, w), dtype=np.uint8)
            self._hull_mask = np.zeros((h, w), dtype=np.uint8)
            blank = (self._image.astype(np.float32) * 0.3).astype(np.uint8)
            return blank.copy(), blank.copy()

        # VLM-guided path: the marker band is already a clean single curve.
        if self._fitted_curve is not None:
            self._contour_mask = self._black_mask.copy()
            ys, xs = np.where(self._black_mask > 0)
            self._hull_mask = np.zeros((h, w), dtype=np.uint8)
            hull = None
            if len(xs) > 2:
                hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
                cv2.fillConvexPoly(self._hull_mask, hull, 255)
            curve_outline = self._fitted_curve.reshape(-1, 1, 2)
            vis_contour = self._make_mask_vis(
                self._contour_mask, [curve_outline], (0, 255, 255)
            )
            vis_hull = self._make_mask_vis(
                self._hull_mask, [hull] if hull is not None else [], (0, 255, 255)
            )
            return vis_contour, vis_hull

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(self._black_mask, cv2.MORPH_OPEN, kernel_open)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self._contour_mask = np.zeros((h, w), dtype=np.uint8)
        selected: list = []

        if contours:
            scored = sorted(
                contours,
                key=lambda c: self._score_contour(c),
                reverse=True,
            )

            best = scored[0]
            best_score = self._score_contour(best)
            _, by, _, bh = cv2.boundingRect(best)
            best_y_mid = by + bh // 2

            for cnt in scored:
                if self._score_contour(cnt) < best_score * 0.25:
                    break
                _, cy, _, ch = cv2.boundingRect(cnt)
                cnt_y_mid = cy + ch // 2
                # Accept companion fragments within ±10 % image height of the best contour
                if abs(cnt_y_mid - best_y_mid) <= h * 0.10:
                    selected.append(cnt)

            cv2.drawContours(self._contour_mask, selected, -1, 255, -1)

        # Convex hull over all cleaned pixels
        ys, xs = np.where(cleaned > 0)
        hull = None
        self._hull_mask = np.zeros((h, w), dtype=np.uint8)
        if len(xs) > 2:
            pts = np.column_stack((xs, ys)).astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(self._hull_mask, hull, 255)

        vis_contour = self._make_mask_vis(self._contour_mask, selected, (0, 255, 255))
        vis_hull = self._make_mask_vis(
            self._hull_mask, [hull] if hull is not None else [], (0, 255, 255)
        )
        return vis_contour, vis_hull

    @staticmethod
    def _score_contour(cnt: np.ndarray) -> float:
        """Score a contour for likelihood of being a marker stroke.

        Rewards large, horizontally-elongated shapes (marker lines).
        Penalises small blobs (hair-root noise, shadows).
        """
        area = float(cv2.contourArea(cnt))
        if area < 20:
            return 0.0
        _, _, cw, ch = cv2.boundingRect(cnt)
        elongation = max(cw, ch) / (min(cw, ch) + 1)
        # Extra reward for horizontal orientation (marker follows hairline)
        orientation_bonus = 1.5 if cw >= ch else 1.0
        return area * min(elongation, 10.0) * orientation_bonus

    def _make_mask_vis(
        self, mask: np.ndarray, contours: list, outline_color: tuple
    ) -> np.ndarray:
        vis = self._image.copy().astype(np.float32)
        vis[mask == 0] *= 0.3
        tint = np.zeros_like(vis)
        tint[mask > 0] = [80, 40, 0]
        vis = np.clip(vis + tint, 0, 255).astype(np.uint8)
        for cnt in contours:
            cv2.drawContours(vis, [cnt], -1, outline_color, 2)
        return vis

    @staticmethod
    def _encode(img: Optional[np.ndarray], fmt: str = ".jpg") -> str:
        if img is None:
            img = np.zeros((1, 1, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        params = [cv2.IMWRITE_JPEG_QUALITY, 90] if fmt == ".jpg" else []
        _, buf = cv2.imencode(fmt, img, params)
        b64 = base64.b64encode(buf).decode()
        mime = "image/jpeg" if fmt == ".jpg" else "image/png"
        return f"data:{mime};base64,{b64}"
