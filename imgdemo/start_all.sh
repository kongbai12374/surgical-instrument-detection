#!/bin/bash
# 一键恢复：启动 YOLO worker + 反向隧道（AutoDL 容器重启后执行）
# 用法：bash /root/imgdemo-worker/start_all.sh

V=/root/autodl-tmp/envs/yolo/bin/python
cd /root/imgdemo-worker || exit 1

if pgrep -f "uvicorn worker:app" >/dev/null; then
  echo "worker: already running"
else
  nohup "$V" -m uvicorn worker:app --host 127.0.0.1 --port 8010 --workers 1 \
    >>/root/imgdemo-worker/worker.log 2>&1 &
  echo "worker: starting..."
  for i in $(seq 1 30); do
    sleep 1
    if curl -s --max-time 2 http://127.0.0.1:8010/api/health >/dev/null 2>&1; then
      echo "worker: ready after ${i}s"
      break
    fi
  done
fi

if pgrep -f "bash tunnel.sh" >/dev/null; then
  echo "tunnel: already running"
elif [ -z "${TUNNEL_HOST}" ]; then
  echo "tunnel: 跳过（未设置 TUNNEL_HOST；需要时 export TUNNEL_HOST=<服务器IP> 后重跑）"
else
  setsid nohup bash tunnel.sh >>/root/imgdemo-worker/tunnel.log 2>&1 &
  sleep 3
  echo "tunnel: started -> ${TUNNEL_HOST}"
fi

echo "--- status ---"
curl -s --max-time 5 http://127.0.0.1:8010/api/health | head -c 300
echo
echo "worker procs: $(pgrep -cf 'uvicorn worker:app')  tunnel procs: $(pgrep -cf 'bash tunnel.sh')"
