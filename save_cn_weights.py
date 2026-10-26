import sys
sys.path.insert(0, "/root/YOLOv11")
from ultralytics import YOLO

NAMES = {0: "手术刀", 1: "直分离钳", 2: "直梅奥剪", 3: "弯梅奥剪"}
src = "/root/YOLOv11/runs/surgical_yolo11n/weights/best.pt"
dst = "/root/YOLOv11/runs/surgical_yolo11n/weights/best_cn.pt"

print("has save:", hasattr(YOLO, "save"))
m = YOLO(src)
m.model.names = NAMES
m.save(dst)

m2 = YOLO(dst)
print("reloaded names:", m2.names)
print("nc:", m2.model.nc)
