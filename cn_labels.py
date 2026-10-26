#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rename YOLO11 surgical classes to Chinese and re-run predict + val with Chinese labels."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_FILE = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT_FILE)
CN = font_manager.FontProperties(fname=FONT_FILE).get_name()
print("registered font family:", CN)
plt.rcParams["font.family"] = CN
plt.rcParams["font.sans-serif"] = [CN, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from ultralytics import YOLO

NAMES = {0: "手术刀", 1: "直分离钳", 2: "直梅奥剪", 3: "弯梅奥剪"}
W = "/root/YOLOv11/runs/surgical_yolo11n/weights/best.pt"
DATA_CN = "/root/autodl-tmp/surgical-yolo/data_cn.yaml"

# ---------- 1. write data_cn.yaml ----------
with open(DATA_CN, "w", encoding="utf-8") as f:
    f.write("path: /root/autodl-tmp/surgical-yolo\ntrain: images/train\nval: images/val\n\n")
    f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in NAMES.items()))
    f.write("\nnc: 4\n")
print("wrote", DATA_CN)
print(open(DATA_CN, encoding="utf-8").read())

# ---------- 2. predict with Chinese labels ----------
model = YOLO(W)
model.model.names = NAMES

imgs = sorted(glob.glob("/root/autodl-tmp/surgical-yolo/images/val/*.jpg"))[:8]
results = model.predict(
    imgs, imgsz=640, conf=0.25, iou=0.7,
    save=True, project="/root/YOLOv11/runs", name="predict_cn", exist_ok=True,
    line_width=3, show_labels=True, show_conf=True, verbose=False,
)
print("predict saved to:", results[0].save_dir)
for f in sorted(os.listdir(results[0].save_dir))[:10]:
    print("  ", f)

# ---------- 3. val with Chinese labels ----------
m = YOLO(W)
m.model.names = NAMES
r = m.val(data=DATA_CN, imgsz=640, batch=32, plots=True,
          project="/root/YOLOv11/runs", name="val_cn", exist_ok=True, verbose=True)
print("mAP50", round(r.box.map50, 4), "| mAP50-95", round(r.box.map, 4))
print("per-class:")
for i, name in NAMES.items():
    print(f"  {name}: AP50={r.box.ap50[i]:.4f} AP={r.box.ap[i]:.4f}")
print("val dir:", r.save_dir)
