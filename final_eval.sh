#!/bin/bash
PY=/root/autodl-tmp/envs/yolo/bin/python
CSV=/root/YOLOv11/runs/surgical_yolo11n/results.csv
echo "=== 最优 epoch (按 mAP50-95) ==="
$PY - <<'PY'
import csv
rows = list(csv.DictReader(open("/root/YOLOv11/runs/surgical_yolo11n/results.csv")))
best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
print("epoch", best["epoch"], "| time", round(float(best["time"]),1), "s")
print("P", round(float(best["metrics/precision(B)"]),4),
      "R", round(float(best["metrics/recall(B)"]),4),
      "mAP50", round(float(best["metrics/mAP50(B)"]),4),
      "mAP50-95", round(float(best["metrics/mAP50-95(B)"]),4))
last = rows[-1]
print("last epoch", last["epoch"], "mAP50", round(float(last["metrics/mAP50(B)"]),4),
      "mAP50-95", round(float(last["metrics/mAP50-95(B)"]),4))
PY

echo "=== 每类指标 (best.pt 验证) ==="
cd /root/YOLOv11 || exit 1
$PY - <<'PY' 2>&1 | grep -vE "^\s*$" | tail -20
from ultralytics import YOLO
m = YOLO("/root/YOLOv11/runs/surgical_yolo11n/weights/best.pt")
r = m.val(data="/root/autodl-tmp/surgical-yolo/data.yaml", split="val",
          imgsz=640, batch=32, plots=True, project="/root/YOLOv11/runs",
          name="surgical_val", exist_ok=True, verbose=True)
print("mAP50", round(r.box.map50, 4), "| mAP50-95", round(r.box.map, 4))
print("speed ms/img", r.speed)
PY
echo "=== 产物 ==="
ls -lh /root/YOLOv11/runs/surgical_yolo11n/weights/
ls /root/YOLOv11/runs/surgical_val/ 2>/dev/null | head -20
