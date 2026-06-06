# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HeadMark is a FastAPI web tool that generates hairline/forehead masks for diffusion models. It runs a 5-step image processing pipeline using MediaPipe face landmarks and head segmentation.

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

## Architecture

- `app.py` — FastAPI app with 4 endpoints: `GET /`, `POST /upload`, `POST /adjust`, `GET /download_mask`
- `processor.py` — `HeadmarkProcessor` class, 5-step pipeline:
  1. MediaPipe face landmarks → upper forehead polygon
  2. Head/body segmentation (head-segmentation preferred, MediaPipe fallback)
  3. Intersection of upper region + head mask (cached — steps 1–3 are not re-run on `/adjust`)
  4. Near-black pixel detection + dilation
  5. Contour and convex hull mask generation
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
