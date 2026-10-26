"""AutoDL 侧 YOLO 手术器械检测服务。

POST /api/predict  multipart file -> 检测框（像素坐标，基于原图）
GET  /api/health   模型、类别、设备、验证集指标

启动：
  /root/autodl-tmp/envs/yolo/bin/python -m uvicorn worker:app --host 127.0.0.1 --port 8010
"""

import asyncio
import csv
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
WEIGHTS = os.environ.get(
    "YOLO_WEIGHTS", "/root/YOLOv11/runs/surgical_yolo11n/weights/best_cn.pt"
)
RESULTS_CSV = os.environ.get(
    "YOLO_RESULTS", "/root/YOLOv11/runs/surgical_yolo11n/results.csv"
)
CONF = float(os.environ.get("YOLO_CONF", "0.25"))
IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
MAX_BYTES = 25 * 1024 * 1024

DEVICE = 0 if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU"
MODEL = YOLO(WEIGHTS)
NAMES = [MODEL.names[i] for i in sorted(MODEL.names)]


def read_metrics() -> dict:
    """从训练日志取最后一轮的 mAP，用作页面展示。"""
    try:
        with open(RESULTS_CSV, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return {}
        last = rows[-1]
        out = {}
        for key, alias in (
            ("metrics/mAP50(B)", "map50"),
            ("metrics/mAP50-95(B)", "map50_95"),
            ("metrics/precision(B)", "precision"),
            ("metrics/recall(B)", "recall"),
            ("epoch", "epoch"),
        ):
            if key in last and last[key]:
                out[alias] = round(float(last[key]), 4)
        out["epochs"] = len(rows)
        return out
    except Exception:
        return {}


METRICS = read_metrics()

# PyTorch 每个线程的 CUDA 流首次推理都有 ~2s 的一次性开销，
# 因此推理固定交给这唯一的线程，并在启动时先把它焐热。
_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")


def _predict_sync(img):
    return MODEL.predict(
        img, imgsz=IMGSZ, conf=CONF, device=DEVICE, half=(DEVICE == 0), verbose=False
    )[0]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """用真实图片预热专用推理线程，把首图的一次性开销提前吃掉。"""

    def _warm():
        t0 = time.perf_counter()
        warm_img = BASE / "warmup.jpg"
        try:
            img = (
                Image.open(warm_img).convert("RGB")
                if warm_img.exists()
                else Image.new("RGB", (640, 480), (128, 128, 128))
            )
            for _ in range(2):
                _predict_sync(img)
            print(f"[warmup] done in {time.perf_counter() - t0:.2f}s", flush=True)
        except Exception as exc:
            print(f"[warmup] failed: {exc!r}", flush=True)

    await asyncio.get_running_loop().run_in_executor(_EXEC, _warm)
    yield


app = FastAPI(
    title="Surgical YOLO Worker",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/api/health")
def health():
    return JSONResponse(
        {
            "ok": True,
            "backend": f"YOLO11n · {GPU_NAME}",
            "device": str(DEVICE),
            "gpu": GPU_NAME,
            "classes": len(NAMES),
            "names": NAMES,
            "conf_threshold": CONF,
            "imgsz": IMGSZ,
            "weights": Path(WEIGHTS).name,
            "metrics": METRICS,
        }
    )


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "文件为空")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "图片超过 25 MB")

    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "无法识别为图片")

    w, h = img.size
    t0 = time.perf_counter()
    try:
        result = await asyncio.get_running_loop().run_in_executor(_EXEC, _predict_sync, img)
    except Exception as exc:
        print(f"[error] {exc!r}", flush=True)
        raise HTTPException(500, "推理失败")
    model_ms = (time.perf_counter() - t0) * 1000

    dets = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        dets.append(
            {
                "label": NAMES[cls_id],
                "class_id": cls_id,
                "conf": round(float(box.conf.item()), 4),
                "box": [x1, y1, x2, y2],
            }
        )
    dets.sort(key=lambda d: -d["conf"])

    counts = {}
    for d in dets:
        counts[d["label"]] = counts.get(d["label"], 0) + 1

    return JSONResponse(
        {
            "ok": True,
            "backend": f"YOLO11n · {GPU_NAME}",
            "device": str(DEVICE),
            "classes": len(NAMES),
            "image": {"width": w, "height": h},
            "detections": dets,
            "counts": counts,
            "model_ms": round(model_ms, 1),
            "infer_ms": round((time.perf_counter() - t0) * 1000, 1),
            "bytes": len(raw),
        }
    )
