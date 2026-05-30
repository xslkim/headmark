import os
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from PIL import Image, ImageOps

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# MediaPipe face mesh landmark indices for forehead boundary
LANDMARK_INDICES = [21, 68, 104, 69, 108, 151, 337, 299, 333, 298, 251]


class HeadmarkProcessor:
    def __init__(self):
        # FaceLandmarker (replaces FaceMesh in new MediaPipe)
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

        # Selfie Segmenter (fallback)
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

        # head-segmentation runs in a subprocess to avoid PyTorch/MediaPipe GPU conflict
        self.use_head_seg = self._check_head_seg_available()
        if self.use_head_seg:
            print("[INFO] head-segmentation 可用，将通过子进程调用")
        else:
            print("[INFO] head-segmentation 不可用，使用 MediaPipe Selfie Segmentation")

        # Cached intermediate results
        self._image = None
        self._image_rgb = None
        self._upper_mask = None
        self._head_mask = None
        self._intersection = None
        self._black_mask = None
        self._contour_mask = None
        self._hull_mask = None

    def process(
        self, image_bytes: bytes, threshold: int = 115, dilation_pct: float = 1.0
    ) -> dict:
        """Process uploaded image through all 5 steps."""
        self._load_image(image_bytes)

        vis1 = self._step1()
        vis2 = self._step2()
        vis3 = self._step3()

        threshold_results = self._process_threshold(threshold, dilation_pct)

        # Return default dilation radius in pixels for frontend display
        w = self._image.shape[1]
        dilation_px = max(1, int(w * dilation_pct / 100))

        return {
            "original": self._encode(self._image),
            "step1": self._encode(vis1),
            "step2": self._encode(vis2),
            "step3": self._encode(vis3),
            "seg_mode": "head-segmentation"
            if self.use_head_seg
            else "selfie-segmentation",
            "image_width": w,
            "dilation_px": dilation_px,
            **threshold_results,
        }

    def adjust(self, threshold: int, dilation_pct: float = 1.0) -> dict:
        """Re-process steps 4-5 with new threshold/dilation (uses cached steps 1-3)."""
        if self._image is None:
            raise ValueError("请先上传图片")
        return self._process_threshold(threshold, dilation_pct)

    def get_mask(self, mask_type: str) -> bytes:
        """Get binary mask as PNG bytes for download."""
        mask = self._contour_mask if mask_type == "contour" else self._hull_mask
        if mask is None:
            raise ValueError("请先处理图片")
        _, buffer = cv2.imencode(".png", mask)
        return buffer.tobytes()

    # ── Internal methods ──

    @staticmethod
    def _check_head_seg_available() -> bool:
        """Check if head-segmentation package is installed and model exists.
        Uses importlib.util.find_spec to avoid importing PyTorch (which conflicts
        with MediaPipe's GPU libraries in the same process).
        """
        try:
            import importlib.util

            spec = importlib.util.find_spec("head_segmentation")
            if spec is None:
                return False
            # Find the package directory without importing it
            pkg_dir = None
            if spec.submodule_search_locations:
                pkg_dir = spec.submodule_search_locations[0]
            elif spec.origin:
                pkg_dir = os.path.dirname(spec.origin)
            if pkg_dir is None:
                return False
            model_path = os.path.join(pkg_dir, "model", "head_segmentation.ckpt")
            return os.path.exists(model_path)
        except Exception:
            return False

    @staticmethod
    def _run_head_seg_subprocess(image_rgb: np.ndarray) -> np.ndarray:
        """Run head segmentation in a subprocess to avoid GPU library conflicts."""
        import subprocess
        import tempfile

        h, w = image_rgb.shape[:2]

        # Save image to temp file
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
                raise RuntimeError(f"head-segmentation subprocess failed: {result.stderr}")

            seg_map = np.load(output_path)
            return seg_map
        finally:
            for p in [input_path, output_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def _load_image(self, image_bytes: bytes):
        pil = Image.open(io.BytesIO(image_bytes))
        pil = ImageOps.exif_transpose(pil)
        self._image_rgb = np.array(pil.convert("RGB"))
        self._image = cv2.cvtColor(self._image_rgb, cv2.COLOR_RGB2BGR)

    def _step1(self):
        """Step 1: MediaPipe face landmarks → upper region mask."""
        h, w = self._image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._image_rgb)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise ValueError("未检测到人脸，请上传包含清晰人脸的照片")

        face_lms = result.face_landmarks[0]
        points = []
        for idx in LANDMARK_INDICES:
            lm = face_lms[idx]
            points.append((int(lm.x * w), int(lm.y * h)))

        # Build upper region polygon
        # Left extension: horizontal line from image left edge at point 21's y
        # Right extension: horizontal line from point 251 to image right edge
        left_ext = (0, points[0][1])
        right_ext = (w - 1, points[-1][1])

        polygon = np.array(
            [left_ext] + points + [right_ext, (w - 1, 0), (0, 0)],
            dtype=np.int32,
        )

        self._upper_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._upper_mask, [polygon], 255)

        # Visualization
        vis = self._image.copy().astype(np.float32)
        # Dim lower region
        vis[self._upper_mask == 0] *= 0.3
        # Green tint on upper region
        green = np.zeros_like(vis)
        green[self._upper_mask > 0] = [0, 60, 0]
        vis = np.clip(vis + green, 0, 255).astype(np.uint8)

        # Draw boundary line
        line_pts = [left_ext] + points + [right_ext]
        for i in range(len(line_pts) - 1):
            cv2.line(vis, line_pts[i], line_pts[i + 1], (0, 255, 0), 2)

        # Draw landmark points with labels
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

    def _step2(self):
        """Step 2: Head/body segmentation."""
        if self.use_head_seg:
            seg_map = self._run_head_seg_subprocess(self._image_rgb)
            self._head_mask = ((seg_map > 0) * 255).astype(np.uint8)
        else:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=self._image_rgb
            )
            result = self.selfie_seg.segment(mp_image)
            confidence_mask = np.squeeze(result.confidence_masks[0].numpy_view())
            self._head_mask = ((confidence_mask > 0.5) * 255).astype(np.uint8)

        # Visualization
        vis = self._image.copy().astype(np.float32)
        vis[self._head_mask == 0] *= 0.3
        tint = np.zeros_like(vis)
        tint[self._head_mask > 0] = [0, 0, 60]
        vis = np.clip(vis + tint, 0, 255).astype(np.uint8)

        # Draw contour of segmented region
        contours, _ = cv2.findContours(
            self._head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

        return vis

    def _step3(self):
        """Step 3: Intersection of upper region and head segmentation."""
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

    def _process_threshold(self, threshold: int, dilation_pct: float = 1.0) -> dict:
        vis4 = self._step4(threshold, dilation_pct)
        vis5a, vis5b = self._step5()

        return {
            "step4": self._encode(vis4),
            "step5a": self._encode(vis5a),
            "step5b": self._encode(vis5b),
            "contour_mask": self._encode(self._contour_mask, fmt=".png"),
            "hull_mask": self._encode(self._hull_mask, fmt=".png"),
        }

    def _step4(self, threshold: int, dilation_pct: float = 1.0):
        """Step 4: Detect black/near-black pixels, then dilate."""
        h, w = self._image.shape[:2]
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        self._black_mask = np.zeros_like(gray)
        roi = self._intersection > 0
        self._black_mask[roi & (gray < threshold)] = 255

        # Dilate detected pixels by radius = image_width * dilation_pct / 100
        dilation_radius = max(1, int(w * dilation_pct / 100))
        kernel_size = dilation_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self._black_mask = cv2.dilate(self._black_mask, kernel)
        # Clip dilated mask to intersection region
        self._black_mask = cv2.bitwise_and(self._black_mask, self._intersection)

        # Visualization: show detected+dilated pixels in red
        vis = self._image.copy()
        vis[self._intersection == 0] = (
            vis[self._intersection == 0].astype(np.float32) * 0.3
        ).astype(np.uint8)
        vis[self._black_mask > 0] = [0, 0, 255]

        return vis

    def _step5(self):
        """Step 5: Create contour and convex hull masks. Keep only the largest region."""
        h, w = self._image.shape[:2]

        if np.sum(self._black_mask > 0) < 50:
            self._contour_mask = np.zeros((h, w), dtype=np.uint8)
            self._hull_mask = np.zeros((h, w), dtype=np.uint8)
            vis = (self._image.astype(np.float32) * 0.3).astype(np.uint8)
            return vis.copy(), vis.copy()

        # Remove small noise (morphological opening)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(self._black_mask, cv2.MORPH_OPEN, kernel_open)

        # ── Contour mask (keep only largest contour) ──
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self._contour_mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            # Keep only the largest contour
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(self._contour_mask, [largest], -1, 255, -1)
            contours = [largest]

        # ── Convex hull mask (single region by definition) ──
        ys, xs = np.where(cleaned > 0)
        hull = None
        if len(xs) > 2:
            points = np.column_stack((xs, ys)).astype(np.int32)
            hull = cv2.convexHull(points)
            self._hull_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(self._hull_mask, hull, 255)
        else:
            self._hull_mask = np.zeros((h, w), dtype=np.uint8)

        # Visualizations
        vis_contour = self._make_mask_vis(
            self._contour_mask, contours, (0, 255, 255)
        )
        vis_hull = self._make_mask_vis(
            self._hull_mask, [hull] if hull is not None else [], (0, 255, 255)
        )

        return vis_contour, vis_hull

    def _make_mask_vis(self, mask, contours, outline_color):
        """Create visualization with mask overlay and contour outline."""
        vis = self._image.copy().astype(np.float32)
        vis[mask == 0] *= 0.3
        tint = np.zeros_like(vis)
        tint[mask > 0] = [80, 40, 0]
        vis = np.clip(vis + tint, 0, 255).astype(np.uint8)
        for cnt in contours:
            cv2.drawContours(vis, [cnt], -1, outline_color, 2)
        return vis

    @staticmethod
    def _encode(img, fmt=".jpg"):
        """Encode image as base64 data URI."""
        if img is None:
            img = np.zeros((1, 1, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            # Grayscale mask - convert to 3-channel for display
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        params = [cv2.IMWRITE_JPEG_QUALITY, 90] if fmt == ".jpg" else []
        _, buf = cv2.imencode(fmt, img, params)
        b64 = base64.b64encode(buf).decode()
        mime = "image/jpeg" if fmt == ".jpg" else "image/png"
        return f"data:{mime};base64,{b64}"
