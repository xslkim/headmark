# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HeadMark (发际线蒙板生成工具) is a web tool that generates a hairline/forehead mask from a portrait photo, intended as input for diffusion models. A user uploads a face photo and the app produces two downloadable binary masks (`contour` and `hull`) over the detected hairline region, with interactive threshold/dilation tuning.

## Commands

```bash
./run.sh          # Create venv if missing, install deps, start server on http://0.0.0.0:8888
python app.py     # Run directly (assumes venv already active + deps installed)
```

There is no test suite, linter, or build step. `run.sh` installs everything except `head-segmentation` from `requirements.txt`, then attempts `pip install head-segmentation` separately (it's allowed to fail — see fallback below).

## Architecture

Two files do all the work:

- **`app.py`** — FastAPI server. Endpoints: `GET /` (renders `templates/index.html`), `POST /upload` (multipart image → full pipeline), `POST /adjust` (JSON `{threshold, dilation_pct}` → re-runs only the threshold-dependent steps), `GET /download_mask?type=contour|hull`.
- **`processor.py`** — `HeadmarkProcessor`, the entire image pipeline.

**Single stateful processor instance.** One global `HeadmarkProcessor` is created at startup and holds the last-uploaded image plus all intermediate masks as instance attributes (`self._image`, `self._intersection`, `self._black_mask`, etc.). This is what makes `/adjust` and `/download_mask` cheap — they reuse cached state from the last `/upload`. The consequence: **the server is effectively single-user / single-image at a time.** Concurrent uploads clobber each other's state. Keep this in mind before adding sessions or parallelism.

**The 5-step pipeline** (`process()` runs all; `adjust()` re-runs only steps 4–5):
1. `_step1` — MediaPipe FaceLandmarker finds forehead-boundary landmarks (`LANDMARK_INDICES`), builds an "upper region" polygon (everything above the hairline boundary line, extended to image edges). Raises `ValueError` if no face is detected.
2. `_step2` — Head/body segmentation → `_head_mask` (see backend note below).
3. `_step3` — `_intersection` = upper-region AND head-mask. This is the ROI for all later steps.
4. `_step4` — Within the intersection, mark near-black pixels (`gray < threshold`) as hair, then dilate by `radius = image_width * dilation_pct / 100` and clip back to the intersection.
5. `_step5` — Produce two final masks from the black-pixel mask: `_contour_mask` (morphological close, keep only the **largest** contour, fill) and `_hull_mask` (convex hull of all hair pixels).

Each step also returns a BGR visualization image; all images are returned to the frontend as base64 data URIs via `_encode` (JPEG for visualizations, PNG for downloadable masks).

**Segmentation backend selection (important gotcha).** Step 2 has two backends:
- Preferred: the `head-segmentation` PyTorch package. It is **run in a subprocess** (`_run_head_seg_subprocess`, via `python3 -c` with numpy temp files) on purpose — importing PyTorch in the same process as MediaPipe causes a GPU library conflict. Availability is checked with `importlib.util.find_spec` so PyTorch is *never imported in the main process* (`_check_head_seg_available`).
- Fallback: MediaPipe Selfie Segmenter (`selfie_segmenter.tflite`), used automatically if `head-segmentation` or its checkpoint isn't present.

`self.use_head_seg` records which is active; it's surfaced to the UI as `seg_mode`.

**Models** live in `models/` (`face_landmarker.task`, `selfie_segmenter.tflite`), loaded by absolute path. The `head-segmentation` checkpoint lives inside that pip package, not in `models/`.

## Notes

- UI strings, comments, and error messages are in Chinese; match that when editing user-facing text.
- `static/` and `uploads/` exist but are currently unused (uploads are processed in memory, never written to disk).
- `app.py` runs uvicorn with `reload=True` and binds `0.0.0.0:8888`.
