#!/bin/bash
# 反向隧道：把服务器的 127.0.0.1:8020 转发到本实例的 127.0.0.1:8010
# 断线自动重连；日志在 /root/imgdemo-worker/tunnel.log

SERVER=admin@SERVER_IP
KEY=/root/.ssh/id_tunnel
RPORT=8020
LPORT=8010

while true; do
  ssh -N \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    -o ConnectTimeout=15 \
    -o StrictHostKeyChecking=accept-new \
    -i "$KEY" \
    -R "127.0.0.1:${RPORT}:127.0.0.1:${LPORT}" \
    "$SERVER"
  echo "[tunnel] disconnected at $(date '+%F %T'), retry in 5s" >>/root/imgdemo-worker/tunnel.log
  sleep 5
done
