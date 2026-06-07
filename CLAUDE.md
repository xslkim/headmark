# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HeadMark is a FastAPI web tool that generates hairline/forehead masks for diffusion models. It runs a 5-step pipeline that combines MediaPipe face landmarks, head segmentation, and a VLM (Doubao Vision) that semantically traces the marker line, then snaps a smooth curve onto the real ink with a black-hat + dynamic-programming ridge follower.

See `ALGORITHM.md` for the full algorithm writeup.

## Running the Server

```bash
# Activate venv first
source venv/bin/activate

# Start dev server (hot-reload on, port 8002)
python3 app.py
```

The app is available at `http://localhost:8002`.

## Virtual Environment

This project uses a standard Python venv. Set it up once:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`head-segmentation` (PyTorch-based) is the preferred segmentation backend but optional — the app falls back to MediaPipe Selfie Segmentation automatically if it's not installed or fails to load.

## Doubao Vision API Key

The VLM-guided detection path needs an Ark API Key (`ark-` prefix). The app reads `env DOUBAO_API_KEY` first, then `apikey.txt` in the repo root. Without a key, `/upload` still works — it degrades to the local fallback detector. The model ID must include the version suffix (e.g. `doubao-seed-1-6-vision-250815`); the un-versioned ID returns 404.

## Architecture

- `app.py` — FastAPI app with 4 endpoints: `GET /`, `POST /upload`, `POST /adjust`, `GET /download_mask`
- `processor.py` — `HeadmarkProcessor` class, 5-step pipeline:
  1. MediaPipe face landmarks → upper forehead polygon
  2. Head/body segmentation (head-segmentation preferred, MediaPipe fallback)
  3. Intersection of upper region + head mask → ROI (cached — steps 1–3 and the VLM call are not re-run on `/adjust`)
  - VLM: Doubao Vision traces the marker line as a normalized polyline (semantic prior; called once per upload). Cached in `_vlm_points`.
  4. Marker-line detection, two paths:
     - **VLM-guided (preferred)**: black-hat transform highlights the dark stroke, then a dynamic-programming ridge follower locks a smooth curve onto the real ink within a narrow band around the VLM prior (the VLM y-coords jitter ±15–20 px and are not trusted directly).
     - **Fallback (no VLM)**: adaptive absolute threshold (marker on skin) OR'd with local-darkness detection (marker on dark hair), within the ROI.
  5. Contour and convex-hull mask generation (VLM path uses the fitted curve directly; fallback path does morphological cleanup + shape-aware contour scoring)
- `doubao_vision.py` — `detect_marker()`: calls the Ark Vision API (`doubao-seed-1-6-vision-250815`, version suffix required) and returns the marker polyline. Returns `None` on failure so the processor degrades to the fallback detector.
- `templates/index.html` — single-page frontend (vanilla JS, no build step)
- `models/` — bundled `.task` and `.tflite` model files; do not delete

**Critical gotcha — GPU library conflict**: `head-segmentation` (PyTorch) and MediaPipe both compete for the GPU. The processor runs `head-segmentation` in an isolated subprocess to prevent crashes. Temp `.npy` files are used for inter-process I/O and cleaned up automatically.

## Code Style

- PEP 8 + type hints on all function signatures
- Match existing Chinese-language UI text and error messages in `app.py` and `templates/index.html`
- No external build or lint tools configured — match the style of existing code

## Git Workflow

- Work on `feature/<name>` branches
- Open a PR to `main` when ready
