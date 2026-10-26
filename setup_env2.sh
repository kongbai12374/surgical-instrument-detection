#!/bin/bash
# Install a sm_120-capable torch on the cloned instance.
ENV=/root/autodl-tmp/envs/yolo
PY=$ENV/bin/python
PIP=$ENV/bin/pip
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== step1: conda create py3.11 (tuna) ==="
mkdir -p /root/autodl-tmp/envs
/root/miniconda3/bin/conda create -y -p "$ENV" python=3.11 2>&1 | tail -4
"$PY" -V || { echo "PYTHON MISSING"; }

echo "=== step2: torch 2.8.0 cu128 (academic proxy) ==="
(
  source /etc/network_turbo 2>/dev/null
  echo "proxy: ${https_proxy:-none}"
  "$PIP" install --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url "$MIRROR" torch==2.8.0 torchvision==0.23.0 2>&1 | tail -10
)

echo "=== step3: verify gpu ==="
"$PY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
    a = torch.randn(1024, 1024, device="cuda")
    print("matmul OK ->", float((a @ a).sum()))
PY

echo "=== step4: ultralytics 8.3.13 (tuna) ==="
"$PIP" install -q "ultralytics==8.3.13" -i "$MIRROR" 2>&1 | tail -5
"$PY" -c "import ultralytics, cv2, numpy; print('ultralytics', ultralytics.__version__, '| cv2', cv2.__version__, '| numpy', numpy.__version__)" 2>&1 | tail -2

echo "=== step5: sizes ==="
du -sh "$ENV"; df -h /root/autodl-tmp | tail -1
echo "=== SETUP2 DONE ==="
