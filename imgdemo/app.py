"""极简图像分类演示服务（CPU + ONNX Runtime）。

接口：
  GET  /              演示页面
  POST /api/predict   上传图片，返回 top-5 类别
  GET  /api/health    健康检查
  GET  /api/samples   内置示例图列表

可选：设置环境变量 DEMO_BACKEND=remote 与 DEMO_REMOTE_URL，把推理转发到
AutoDL 等外部实例（该实例需接受 multipart 的 file 字段并返回同结构 JSON）。
"""

import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "mobilenet_v2.onnx"
LABELS_PATH = BASE / "labels.json"
STATIC_DIR = BASE / "static"
SAMPLES_DIR = BASE / "samples"

MAX_BYTES = 25 * 1024 * 1024
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BACKEND = os.environ.get("DEMO_BACKEND", "local").strip().lower()
REMOTE_URL = os.environ.get("DEMO_REMOTE_URL", "").strip()

LABELS = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

_session = None
_input_name = None


def _get_session() -> tuple[ort.InferenceSession, str]:
    global _session, _input_name
    if _session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            str(MODEL_PATH), opts, providers=["CPUExecutionProvider"]
        )
        _input_name = _session.get_inputs()[0].name
    return _session, _input_name


def preprocess(raw: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """短边缩放到 256，中心裁剪 224，再按 ImageNet 均值方差归一化。"""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    nw, nh = max(SIZE, round(w * scale)), max(SIZE, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - SIZE) // 2, (nh - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr, dtype=np.float32), (w, h)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def infer_local(raw: bytes) -> dict:
    session, input_name = _get_session()
    tensor, (w, h) = preprocess(raw)
    logits = session.run(None, {input_name: tensor})[0].reshape(-1).astype(np.float32)

    # 该导出把 index 0 作为 background，1..1000 才是 ImageNet 类别。
    if logits.shape[0] == len(LABELS) + 1:
        names = ["背景 / 其他"] + list(LABELS)
    elif logits.shape[0] == len(LABELS):
        names = list(LABELS)
    else:
        raise RuntimeError(f"模型输出 {logits.shape[0]} 类，标签表 {len(LABELS)} 条，不匹配")

    probs = softmax(logits)
    top = np.argsort(-probs)[:5]
    return {
        "ok": True,
        "backend": "MobileNetV2 · CPU",
        "image": {"width": w, "height": h},
        "top": [
            {"label": names[int(i)], "prob": round(float(probs[int(i)]), 4)}
            for i in top
        ],
    }


def infer_remote(raw: bytes, filename: str, content_type: str) -> dict:
    """转发到外部推理实例（如 AutoDL），期望对方返回同结构 JSON。"""
    import httpx

    files = {"file": (filename or "upload.jpg", raw, content_type or "image/jpeg")}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(REMOTE_URL, files=files)
    resp.raise_for_status()
    data = resp.json()
    data.setdefault("ok", True)
    data["backend"] = data.get("backend", "远程实例")
    return data


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """预热一次，避免首张图把模型加载时间算进推理耗时。"""
    if BACKEND == "local":
        try:
            session, input_name = _get_session()
            session.run(None, {input_name: np.zeros((1, 3, SIZE, SIZE), np.float32)})
        except Exception as exc:  # 预热失败不应阻止启动
            print(f"[warn] warmup failed: {exc}")
    yield


app = FastAPI(
    title="Image Demo", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "backend": BACKEND, "labels": len(LABELS)})


@app.get("/api/samples")
def samples() -> JSONResponse:
    items = []
    if SAMPLES_DIR.is_dir():
        for p in sorted(SAMPLES_DIR.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                items.append({"name": p.stem, "url": f"samples/{p.name}"})
    return JSONResponse({"items": items})


@app.get("/samples/{name}")
def sample_file(name: str) -> FileResponse:
    path = (SAMPLES_DIR / name).resolve()
    if not SAMPLES_DIR.resolve() in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="图片超过 25 MB")

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="无法识别为图片，请上传 JPG/PNG/WebP")

    try:
        if BACKEND == "remote" and REMOTE_URL:
            result = infer_remote(raw, file.filename or "", file.content_type or "")
        else:
            result = infer_local(raw)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[error] inference failed: {exc!r}")
        raise HTTPException(status_code=500, detail="推理失败，请重试")

    result["infer_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["bytes"] = len(raw)
    return JSONResponse(result)
