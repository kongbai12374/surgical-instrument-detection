#!/bin/bash
set -e
echo "=== install CJK font ==="
apt-get update -qq 2>&1 | tail -2
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq fonts-wqy-zenhei 2>&1 | tail -3
echo "=== verify ==="
fc-list :lang=zh | head -3

echo "=== wire font into ultralytics ==="
F=$(fc-list :lang=zh file | head -1 | cut -d: -f1)
echo "picked: $F"
mkdir -p /root/.config/Ultralytics
cp "$F" /root/.config/Ultralytics/Arial.ttf
cp "$F" /root/.config/Ultralytics/Arial.Unicode.ttf
ls -lh /root/.config/Ultralytics/

echo "=== render test ==="
/root/autodl-tmp/envs/yolo/bin/python - <<'PY'
from PIL import Image, ImageDraw, ImageFont
img = Image.new("RGB", (620, 110), "white")
d = ImageDraw.Draw(img)
f = ImageFont.truetype("/root/.config/Ultralytics/Arial.ttf", 34)
d.text((10, 10), "手术刀 0.96  直分离钳 0.94", font=f, fill="black")
d.text((10, 58), "直梅奥剪 0.98  弯梅奥剪 0.95", font=f, fill=(200, 0, 0))
img.save("/root/font_test.png")
print("saved /root/font_test.png")
PY
