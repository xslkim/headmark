---
name: verify
description: Verify the HeadMark app is working correctly by exercising the upload and adjust endpoints with a test image.
disable-model-invocation: true
---

1. Confirm the server is running on port 8002: `curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/`. If it returns anything other than `200`, run the `/run` skill first.
2. Find a test image: look for any `.jpg` or `.png` file in the project directory (excluding `models/`). If none exists, ask the user to provide a path to a face image.
3. Upload the image to `/upload`:
   ```bash
   curl -s -X POST http://localhost:8002/upload \
     -F "file=@<path-to-image>" | python3 -m json.tool
   ```
   Check that the response contains `step1` through `step5` keys and no `detail` error field.
4. Test the `/adjust` endpoint with a modified threshold:
   ```bash
   curl -s -X POST http://localhost:8002/adjust \
     -H "Content-Type: application/json" \
     -d '{"threshold": 80, "dilation_pct": 2.0}' | python3 -m json.tool
   ```
   Check that it returns updated `step4` and `step5` keys.
5. Test the mask download endpoint:
   ```bash
   curl -s -o /tmp/mask_contour.png "http://localhost:8002/download_mask?type=contour"
   file /tmp/mask_contour.png
   ```
   Confirm the output is a valid PNG file.
6. Report: which steps passed, any error details, and the active segmentation mode (visible in the `/upload` response or by checking `processor.use_head_seg` from the server logs).
