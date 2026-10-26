#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-run detection with the Chinese-label weights and build a contact sheet."""
import os, glob
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

FONT_FILE = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_manager.fontManager.addfont(FONT_FILE)
CN = font_manager.FontProperties(fname=FONT_FILE).get_name()
plt.rcParams["font.family"] = CN
plt.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, "/root/YOLOv11")
from ultralytics import YOLO

W = "/root/YOLOv11/runs/surgical_yolo11n/weights/best_cn.pt"
OUT = "/root/YOLOv11/runs/detect_cn"
model = YOLO(W)
print("names:", model.names)

imgs = sorted(glob.glob("/root/autodl-tmp/surgical-yolo/images/val/*.jpg"))[:12]
res = model.predict(imgs, imgsz=640, conf=0.25, iou=0.7, save=True,
                    project="/root/YOLOv11/runs", name="detect_cn",
                    exist_ok=True, line_width=3, verbose=False)

names = model.names
total = 0
from collections import Counter
cls_counter = Counter()
conf_all = []
per_image = []
for r in res:
    boxes = r.boxes
    n = len(boxes)
    total += n
    items = []
    for c, cf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        cls_counter[names[int(c)]] += 1
        conf_all.append(cf)
        items.append(f"{names[int(c)]} {cf:.2f}")
    per_image.append((os.path.basename(r.path), items))

print(f"\n图片 {len(res)} 张 | 检出目标 {total} 个")
for fn, items in per_image:
    print(f"  {fn}: {', '.join(items) if items else '(无检出)'}")
if conf_all:
    print(f"平均置信度 {sum(conf_all)/len(conf_all):.3f} | 类别分布 {dict(cls_counter)}")

# contact sheet 2x3
saved = [os.path.join(OUT, os.path.basename(p)) for p in imgs]
saved = [p for p in saved if os.path.exists(p)][:6]
if saved:
    W_, H_ = 640, 480
    sheet = Image.new("RGB", (W_ * 2, H_ * 3), "white")
    f = ImageFont.truetype(FONT_FILE, 20)
    for idx, p in enumerate(saved):
        im = Image.open(p).convert("RGB").resize((W_, H_))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 632, 28], fill=(0, 0, 0))
        d.text((6, 3), os.path.basename(p), font=f, fill=(255, 255, 0))
        sheet.paste(im, ((idx % 2) * W_, (idx // 2) * H_))
    sheet.save(os.path.join(OUT, "contact_sheet.jpg"), quality=90)
    print("contact sheet:", os.path.join(OUT, "contact_sheet.jpg"))
print("output dir:", OUT)
