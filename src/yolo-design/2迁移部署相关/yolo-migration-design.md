# YOLO 检测服务迁移设计方案 — 本机 → 5060Ti

**日期：** 2026-05-02
**目标：** 将 YOLO 检测服务 + API 接口 + Cloudflare 内网穿透从本机 (Ultra 7 155H, CPU) 迁移到 5060Ti 机器 (i5-14600KF, RTX 5060Ti 16GB, GPU)

---

## 1. 当前架构

```
Windows 11 (本机 — Redmibook 14pro)
├── .wslconfig: networkingMode=mirrored + autoProxy=true + dnsTunneling=true + firewall=true
├── /etc/wsl.conf: systemd=true, default=alice
├── VPN → 127.0.0.1:7897
├── WSL2 Ubuntu-24.04 (D:\WSL\Ubuntu-24.04)
│   ├── ~/yolo-service/          ← FastAPI + YOLOv11m 源码
│   │   ├── Dockerfile           ← python:3.11-slim, CPU-only torch
│   │   ├── requirements.txt
│   │   ├── src/yolo/infer.py    ← device="cpu"
│   │   ├── models/yolo11m.pt
│   │   └── data/{input,output}/
│   ├── ~/cloudflared/           ← 隧道配置文件
│   │   ├── config.yml
│   │   └── 6885fde8-....json    ← 凭证
│   └── Docker (CE, 非 Desktop)
│       ├── yolo_service (yolo-api:v2, port 8000)
│       └── cloudflare_tunnel (HTTP/2)
└── 公网域名: yolo.alice1.xyz → Cloudflare Tunnel → localhost:8000
```

## 2. 目标架构

```
Windows 11 (5060Ti — i5-14600KF)
├── .wslconfig: 同上 (mirror + autoProxy + dnsTunneling + firewall)
├── /etc/wsl.conf: 同上 (systemd=true, default=alice)
├── VPN → 127.0.0.1:7897 (同端口)
├── NVIDIA 驱动 (已装)
├── WSL2 Ubuntu-24.04 (D:\WSL\Ubuntu-24.04)
│   ├── ~/yolo-service/
│   │   ├── Dockerfile           ← ultralytics/ultralytics:latest (自带 CUDA + PyTorch)
│   │   ├── requirements.txt     ← 只装 FastAPI 栈
│   │   ├── src/yolo/infer.py    ← device=0 (GPU)
│   │   ├── models/yolo11m.pt    ← 新下载
│   │   └── data/{input,output}/
│   ├── ~/cloudflared/           ← 从本机拷贝
│   ├── Docker (CE)
│   └── NVIDIA Container Toolkit
│       ├── yolo_service (yolo-api:gpu, --gpus all, port 8000)
│       └── cloudflare_tunnel (复用本机配置)
└── 公网域名: yolo.alice1.xyz (不变, 关掉本机后自动走 5060Ti)
```

## 3. 迁移步骤见yolo-migration-plan.md

## 4. 差异对比

| 项目 | 本机 (当前) | 5060Ti (目标) |
|------|------------|---------------|
| CPU | Ultra 7 155H | i5-14600KF |
| GPU | 无 | RTX 5060Ti 16GB |
| 推理设备 | `device="cpu"` | `device=0` |
| Docker 基础镜像 | `python:3.11-slim` | `ultralytics/ultralytics:latest` |
| PyTorch | `pip install torch --index-url .../cpu` | 镜像自带 CUDA PyTorch |
| Docker 运行参数 | 无 | `--gpus all` |
| 其他 | 完全相同 | 完全相同 |

## 5. 验证清单

| # | 验证项 | 命令 | 预期 |
|---|--------|------|------|
| 1 | GPU 可用 | `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` | 5060Ti 信息 |
| 2 | YOLO 服务 | `docker logs yolo_service` | Uvicorn running |
| 3 | 本地上传 | `curl -X POST http://localhost:8000/upload -F "file=@test.jpg"` | 201 |
| 4 | GPU 推理 | `curl -X POST http://localhost:8000/detect -H "Content-Type: application/json" -d '{"image_path":"/data/input/test.jpg"}'` | 推理时间应 <100ms |
| 5 | 隧道连接 | `docker logs cloudflare_tunnel` | Registered tunnel connection |
| 6 | 公网检测 | `curl -X POST https://yolo.alice1.xyz/detect -H "Content-Type: application/json" -d '{"image_path":"/data/input/test.jpg"}'` | 同本机结果 |
| 7 | 本机关闭 | 停止本机 Docker | 公网仍可访问 |
| 8 | 文件类型校验 | 上传 .py 文件 | 400 |
| 9 | 路径穿越 | `curl /images/../../../etc/passwd` | 404 |

## 6. 注意事项

- **隧道冲突：** 两台机器不能同时跑 `cloudflare_tunnel`，迁移验证完成前本机隧道保持运行，5060Ti 上先不要启动隧道容器，或用 `docker run` 但不启动 (先创建好)。
- **模型下载：** `yolo11m.pt` 走 GitHub Release 直链下载，若直连慢可用 `curl -L` 或镜像。
- **代理：** WSL 的 `autoProxy=true` 会自动读取 Windows 代理设置，无需手动配环境变量。
- **WSL 重启：** `.wslconfig` 和 `/etc/wsl.conf` 修改后需 `wsl --shutdown` 再启动才能生效。
- **Docker 重启：** `nvidia-ctk runtime configure` 后必须 `sudo systemctl restart docker`。
- **端口冲突：** 5060Ti 上确保没有其他程序占用 8000 和 20241 端口。
