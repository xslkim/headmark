"""
Doubao Vision integration for marker pen detection.
使用火山引擎方舟视觉大模型语义定位发际线马克笔标记线。

认证：
  API Key（方舟控制台 → API Key 管理，ark- 前缀）作为 Bearer Token。
  env DOUBAO_API_KEY，或调用时传 api_key 参数。

模型 ID 必须带版本号，例如 doubao-seed-1-6-vision-250815
（不带版本的 doubao-seed-1-6-vision 会返回 404）。

detect_marker() 返回一条折线（沿马克笔线采样的点），用于约束本地像素检测的 ROI。
"""

import base64
import json
import os
from typing import Optional

import cv2
import numpy as np

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = os.environ.get("DOUBAO_MODEL", "doubao-seed-1-6-vision-250815")

_PROMPT = (
    "这张植发机构的照片里，有人用黑色马克笔（记号笔）在额头或头皮上"
    "画了一条发际线标记线，用于标注预期的新发际线位置。\n\n"
    "请沿着这条马克笔线，从最左端到最右端，均匀采样大约 15 个点，描述线的走向。\n"
    "严格只返回如下 JSON（不要包含任何其他文字或 Markdown 代码块）：\n"
    '{"has_marker": true, "points": [[x, y], [x, y], ...]}\n'
    "其中 x、y 为 0.0–1.0 之间的小数（占图片宽/高的比例），按从左到右排序。\n"
    "如果图中没有明显的马克笔标记线，返回：\n"
    '{"has_marker": false}\n\n'
    "注意：只标注人工绘制的马克笔线条，不要把自然头发边缘、眉毛或皮肤阴影当作标记线。"
)


def detect_marker(
    image: np.ndarray,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[dict]:
    """
    调用 Doubao Vision 定位马克笔线，返回采样折线。

    Args:
        image:   BGR image (numpy array)
        api_key: Ark API Key；不传则读环境变量 DOUBAO_API_KEY
        model:   带版本号的模型 ID 或接入点 ID

    Returns:
        {"has_marker": True, "points": [[x, y], ...]}  # x, y ∈ [0, 1]
        {"has_marker": False}
        None — API/解析失败（调用方降级到本地算法）
    """
    key = (api_key or os.environ.get("DOUBAO_API_KEY", "")).strip()
    if not key:
        print("[WARN] No DOUBAO_API_KEY provided")
        return None

    try:
        from volcenginesdkarkruntime import Ark

        client = Ark(api_key=key, base_url=ARK_BASE_URL)
    except Exception as exc:
        print(f"[WARN] Ark client init failed: {exc}")
        return None

    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
            max_tokens=600,
            temperature=0,
        )
        content: str = resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[WARN] Doubao Vision API call failed: {exc}")
        return None

    # Strip markdown fences if the model wraps the JSON
    if content.startswith("```"):
        body = content.splitlines()[1:]
        content = "\n".join(body).replace("```", "").strip()

    try:
        result: dict = json.loads(content)
    except json.JSONDecodeError:
        print(f"[WARN] Doubao returned non-JSON: {content[:200]}")
        return None

    if not result.get("has_marker"):
        return {"has_marker": False}

    raw_pts = result.get("points") or []
    points = []
    for p in raw_pts:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            x = max(0.0, min(1.0, float(p[0])))
            y = max(0.0, min(1.0, float(p[1])))
            points.append([x, y])

    if len(points) < 2:
        return {"has_marker": False}

    points.sort(key=lambda p: p[0])  # left → right
    return {"has_marker": True, "points": points}
