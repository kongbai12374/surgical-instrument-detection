# 手术器械检测演示站

上传一张器械照片，模型框出每件器械的位置并给出类别。推理跑在 AutoDL 的 GPU 实例上，
网页由阿里云这台机器提供。

## 地址

```
http://SERVER_IP/demo/
```

免操作演示（自动载入“四件同框”示例并出结果）：

```
http://SERVER_IP/demo/?demo=1
```

## 能检测几类

**4 类**，检测（不是整图分类），一张图里可以同时框出多件：

| id | 中文 | 英文（data.yaml） | 颜色 |
|---|---|---|---|
| 0 | 手术刀 | Scalpel | 蓝 |
| 1 | 直分离钳 | Straight Dissection Clamp | 绿 |
| 2 | 直梅奥剪 | Straight Mayo Scissor | 橙 |
| 3 | 弯梅奥剪 | Curved Mayo Scissor | 紫 |

模型与指标（训练日志最后一轮，共 100 epoch）：

- YOLO11n，权重 `best_cn.pt`（5.3 MB），输入 640，置信度阈值 0.25
- mAP50 **98.6%**、mAP50-95 **92.0%**、precision 95.6%、recall 94.8%
- 验证集 786 张图片，4 类各约 1:1

## 架构

```
浏览器 ──► 阿里云 SERVER_IP:80
                │  /demo/          → /opt/imgdemo/static/  (静态页)
                │  /demo/samples/  → /opt/imgdemo/samples/ (示例图)
                │  /demo/api/      → 127.0.0.1:8020
                ▼
        SSH 反向隧道（AutoDL 主动连出，无需入站端口）
                ▼
        AutoDL RTX 5090 · 127.0.0.1:8010 · YOLO11n
```

选这个结构的原因：AutoDL 实例没有公网入站端口，只有出网和它自己的 6006 代理；
反向隧道让服务器“借用”一个本地端口访问 GPU 服务，实例侧不需要开放任何端口。

实测：模型 10–35 ms/张，公网端到端 0.21–0.34 s（含上传、隧道、回传）。

## 文件位置

AutoDL（`ssh autodl`，`connect.REGION.seetacloud.com:PORT`，root）：

| 路径 | 说明 |
|---|---|
| `/root/imgdemo-worker/worker.py` | YOLO 推理服务（FastAPI + ultralytics） |
| `/root/imgdemo-worker/start_all.sh` | **一键恢复**：起 worker + 起隧道 |
| `/root/imgdemo-worker/tunnel.sh` | 反向隧道，断线自动重连（5 s） |
| `/root/imgdemo-worker/warmup.jpg` | 启动预热用图，必须有检测框，否则首图会慢 2 s |
| `/root/imgdemo-worker/worker.log`、`tunnel.log` | 日志 |
| `/root/autodl-tmp/envs/yolo/bin/python` | 运行环境（torch 2.8.0+cu128，ultralytics 8.3.13） |

阿里云（`ssh SERVER_IP`，admin）：

| 路径 | 说明 |
|---|---|
| `/opt/imgdemo/static/index.html` | 单页前端，全内联，无 CDN |
| `/opt/imgdemo/samples/*.jpg` | 4 张示例图 |
| `/etc/nginx/sites-available/default` | `/demo/` 三个 location（改动前的备份 `.pre-detect`、`.bak.*`） |
| `/home/admin/.ssh/authorized_keys` | 含 `autodl-tunnel` 公钥（隧道用） |
| `/opt/imgdemo/app.py` | **已停用的**旧 CPU 分类演示，仅留档（服务 `imgdemo` 已 disable） |

## 常用运维

```bash
# AutoDL：容器重启后（实例开机、镜像重置后都必须执行）
ssh autodl
bash /root/imgdemo-worker/start_all.sh

# 看状态
curl -s http://127.0.0.1:8010/api/health        # AutoDL 本地
curl -s http://127.0.0.1:8020/api/health        # 阿里云经隧道

# 阿里云：确认 nginx 与路由
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1/demo/
```

## 已知限制

- **AutoDL 实例关机 / 释放后，网页会报 502**（前端已提示“推理实例暂时离线”）。
  演示前先确认实例在运行，并执行一次 `start_all.sh`。
- 隧道依赖 AutoDL 能出网连到 `SERVER_IP:22`；若阿里云换了 IP 或改端口，要同步改 `tunnel.sh`。
- 推理固定跑在**单线程**上（`ThreadPoolExecutor(max_workers=1)`）。这是刻意的：
  PyTorch 每个新线程的 CUDA 流首次推理要重付约 2 s，串行化后每张都稳定在 10–35 ms。
  代价是不支持并发，演示场景够用。
- 置信度固定 0.25，阈值以下的目标不显示。要更灵敏就调 `YOLO_CONF` 环境变量（见下）。
- 单张上限 25 MB；支持 JPG/PNG/WebP；HEIC 不支持。
- 无鉴权，拿到地址即可用。

## 换模型 / 调参

在 `start_all.sh` 里给 uvicorn 加环境变量，或直接改 worker 的默认值：

```bash
YOLO_WEIGHTS=/root/YOLOv11/runs/surgical_yolo11n/weights/best.pt   # 英文标签版
YOLO_CONF=0.4        # 提高阈值，减少低置信度框
YOLO_IMGSZ=960       # 小目标更准，速度略降
```

改完重启 worker（`start_all.sh` 会跳过已在跑的进程，需先 `pkill -f 'uvicorn worker:app'`）。

## 演示结束下线

```bash
# 只停网页（GPU 实例保持原样）
sudo systemctl stop nginx            # 或把 default 里的 /demo 三个 location 删掉后 reload
# AutoDL 侧停止推理与隧道
pkill -f 'uvicorn worker:app'; pkill -f 'bash tunnel.sh'
```

最省事的做法是**在 AutoDL 控制台关机**：实例停掉后网页自然 502，GPU 也不再计费。
