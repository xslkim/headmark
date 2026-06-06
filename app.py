import os

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from processor import HeadmarkProcessor

_APIKEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apikey.txt")


def _load_api_key() -> str:
    """Doubao Ark API Key: env DOUBAO_API_KEY first, else apikey.txt."""
    key = os.environ.get("DOUBAO_API_KEY", "").strip()
    if key:
        return key
    try:
        with open(_APIKEY_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


DOUBAO_API_KEY: str = _load_api_key()

app = FastAPI(title="HeadMark - 发际线蒙板生成工具")
templates = Jinja2Templates(directory="templates")

processor: HeadmarkProcessor = None


@app.on_event("startup")
async def startup():
    global processor
    processor = HeadmarkProcessor()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"seg_mode": processor.use_head_seg},
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    try:
        result = processor.process(
            contents,
            threshold=115,
            dilation_pct=1.0,
            doubao_api_key=DOUBAO_API_KEY or None,
        )
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理出错: {str(e)}")


@app.post("/adjust")
async def adjust(request: Request):
    data = await request.json()
    threshold = int(data.get("threshold", 115))
    threshold = max(0, min(255, threshold))
    dilation_pct = float(data.get("dilation_pct", 1.0))
    dilation_pct = max(0, min(5.0, dilation_pct))

    try:
        result = processor.adjust(threshold, dilation_pct)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理出错: {str(e)}")


@app.get("/download_mask")
async def download_mask(type: str = "contour"):
    if type not in ("contour", "hull"):
        raise HTTPException(status_code=400, detail="type 必须是 contour 或 hull")
    try:
        png_bytes = processor.get_mask(type)
        filename = f"mask_{type}.png"
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
