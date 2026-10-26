#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-run detection on a stratified sample (3 images per class) and check pred vs GT."""
import os, glob, sys, random
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

sys.path.insert(0, "/root/YOLOv11")
from ultralytics import YOLO

W = "/root/YOLOv11/runs/surgical_yolo11n/weights/best_cn.pt"
OUT = "/root/YOLOv11/runs/detect_cn2"
IMG_DIR = "/root/autodl-tmp/surgical-yolo/images/val"
LBL_DIR = "/root/autodl-tmp/surgical-yolo/labels/val"
NAMES = {0: "手术刀", 1: "直分离钳", 2: "直梅奥剪", 3: "弯梅奥剪"}

# ground truth per image
gt = {}
for lf in glob.glob(LBL_DIR + "/*.txt"):
    stem = os.path.splitext(os.path.basename(lf))[0]
    ids = set()
    with open(lf) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                ids.add(int(p[0]))
    gt[stem] = sorted(ids)

# stratified sample: 3 per class
random.seed(0)
picks = []
for cid in NAMES:
    pool = [s for s, ids in gt.items() if cid in ids]
    picks += sorted(random.sample(pool, min(3, len(pool))))
imgs = [os.path.join(IMG_DIR, s + ".jpg") for s in picks]

model = YOLO(W)
res = model.predict(imgs, imgsz=640, conf=0.25, iou=0.7, save=True,
                    project="/root/YOLOv11/runs", name="detect_cn2",
                    exist_ok=True, line_width=3, verbose=False)

print(f"{'图片':<16}{'标注':<22}{'预测':<34}一致")
ok = 0
conf_all = []
for r in res:
    stem = os.path.splitext(os.path.basename(r.path))[0]
    g = gt.get(stem, [])
    preds = [(NAMES[int(c)], float(cf)) for c, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    conf_all += [p[1] for p in preds]
    pstr = ", ".join(f"{n} {c:.2f}" for n, c in preds) or "(无检出)"
    gstr = ", ".join(NAMES[i] for i in g)
    match = sorted(n for n, _ in preds) == sorted(NAMES[i] for i in g)
    ok += int(match)
    print(f"{stem:<16}{gstr:<22}{pstr:<34}{'✓' if match else '✗'}")

print(f"\n完全一致 {ok}/{len(res)} 张 | 平均置信度 {sum(conf_all)/max(len(conf_all),1):.3f}")

# contact sheet 3x2
saved = [os.path.join(OUT, os.path.basename(p)) for p in imgs]
saved = [p for p in saved if os.path.exists(p)][:6]
W_, H_ = 640, 480
sheet = Image.new("RGB", (W_ * 2, H_ * 3), "white")
f = ImageFont.truetype(FONT_FILE, 20)
for idx, p in enumerate(saved):
    im = Image.open(p).convert("RGB").resize((W_, H_))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 640, 28], fill=(0, 0, 0))
    d.text((6, 3), os.path.basename(p), font=f, fill=(255, 255, 0))
    sheet.paste(im, ((idx % 2) * W_, (idx // 2) * H_))
sheet.save(os.path.join(OUT, "contact_sheet.jpg"), quality=90)
print("contact sheet:", os.path.join(OUT, "contact_sheet.jpg"))
