#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert Surgical-Dataset (Roboflow-style folders) into YOLO detection layout."""
import os, shutil, random
from collections import Counter

D = "/root/YOLOv11/datasets/surgical-dataset/Surgical-Dataset"
IMG_SRC = os.path.join(D, "Images/All/images")
LBL_SRC = os.path.join(D, "Labels/label object names")
GRP = os.path.join(D, "Test-Train Groups")
OUT = "/root/autodl-tmp/surgical-yolo"
NAMES = ["Scalpel", "Straight Dissection Clamp", "Straight Mayo Scissor", "Curved Mayo Scissor"]

def stem(p):
    return os.path.splitext(os.path.basename(p.strip()))[0]

def read_group(fn):
    with open(os.path.join(GRP, fn)) as f:
        return [stem(l) for l in f if l.strip()]

imgs = {stem(f): f for f in os.listdir(IMG_SRC) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
labs = {stem(f): f for f in os.listdir(LBL_SRC) if f.lower().endswith(".txt")}

train = read_group("train-obj_detector.txt")
val = read_group("test-obj_detector.txt")
extra = sorted(set(imgs) - set(train) - set(val))

splits = {"train": train + extra, "val": val}

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

stats = {}
for split, names in splits.items():
    n_img = n_lab = n_box = 0
    cls = Counter()
    for n in names:
        if n not in imgs:
            continue
        shutil.copy2(os.path.join(IMG_SRC, imgs[n]), os.path.join(OUT, f"images/{split}", imgs[n]))
        n_img += 1
        if n in labs:
            dst = os.path.join(OUT, f"labels/{split}", n + ".txt")
            shutil.copy2(os.path.join(LBL_SRC, labs[n]), dst)
            n_lab += 1
            with open(dst) as f:
                for line in f:
                    p = line.split()
                    if len(p) == 5:
                        cls[int(p[0])] += 1
                        n_box += 1
    stats[split] = (n_img, n_lab, n_box, dict(sorted(cls.items())))

yaml_path = os.path.join(OUT, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n\n")
    f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    f.write(f"\nnc: {len(NAMES)}\n")

for split, (ni, nl, nb, c) in stats.items():
    print(f"{split}: images={ni} labels={nl} boxes={nb} avg_boxes={nb/max(ni,1):.2f} classes={c}")
print("extras added to train:", len(extra))
print("OUT =", OUT)
print(open(yaml_path).read())
