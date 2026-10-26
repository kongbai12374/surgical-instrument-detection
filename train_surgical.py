#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train YOLO11 on the surgical instrument detection dataset."""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/surgical-yolo/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--weights", default="", help="optional weights to load on top of --model")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--cache", default="ram", help="ram | disk | empty")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--project", default="/root/YOLOv11/runs")
    ap.add_argument("--name", default="surgical_yolo11n")
    ap.add_argument("--rect", action="store_true", help="rectangular training (no padding to square)")
    ap.add_argument("--deterministic", action="store_true", help="reproducible but slower")
    ap.add_argument("--plots", action="store_true", help="save per-epoch plots")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--resume", action="store_true")
    o = ap.parse_args()

    model = YOLO(o.model)
    if o.weights:
        model = model.load(o.weights)
    model.info()
    model.train(
        data=o.data,
        epochs=o.epochs,
        imgsz=o.imgsz,
        batch=o.batch,
        workers=o.workers,
        device=o.device,
        patience=o.patience,
        cache=(o.cache or None),
        fraction=o.fraction,
        rect=o.rect,
        deterministic=o.deterministic,
        plots=o.plots,
        project=o.project,
        name=o.name,
        val=not o.no_val,
        exist_ok=True,
        resume=o.resume,
    )


if __name__ == "__main__":
    main()
