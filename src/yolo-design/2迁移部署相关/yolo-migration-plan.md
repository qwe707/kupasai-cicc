# YOLO 检测服务迁移实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 YOLO 检测服务 + API 接口 + Cloudflare 内网穿透从本机 (Ultra 7 155H, CPU) 迁移到 5060Ti (i5-14600KF, RTX 5060Ti 16GB, GPU)，切换为 GPU 推理。

**架构：** 5060Ti 上重新搭建 WSL2 + Docker Engine + NVIDIA Container Toolkit，基于 ultralytics/ultralytics:latest 构建 GPU 版 YOLO 服务镜像，复用本机的 Cloudflare Tunnel 配置。

**技术栈：** WSL2 / Docker CE / NVIDIA Container Toolkit / ultralytics:latest / FastAPI / Cloudflare Tunnel

**执行标记：**
- `[本机]` = 在当前笔记本上执行
- `[5060Ti]` = 在 5060Ti 新机器上执行
- `[通用]` = 在两台机器上任意一台执行（取决于你当前在哪台前操作）

---

## 任务 1：5060Ti — WSL2 安装与基础配置

**地点：** 5060Ti 机器, Windows PowerShell (管理员)

- [ ] **步骤 1：安装 WSL2 + Ubuntu-24.04 到 D 盘**

```powershell
# PowerShell (管理员)
New-Item -ItemType Directory -Force D:\WSL\Ubuntu-24.04
wsl --install -d Ubuntu-24.04 --location D:\WSL\Ubuntu-24.04
wsl --set-default-version 2
```

完成后 WSL 会自动启动进入 Ubuntu 首次设置界面，设好用户名和密码（注意：第一次启动要等几分钟）。

- [ ] **步骤 2：创建 `.wslconfig`**

```powershell
# PowerShell, 创建 C:\Users\<你的用户名>\.wslconfig
notepad $env:USERPROFILE\.wslconfig
```

内容：

```ini
[wsl2]
networkingMode=mirrored
autoProxy=true
dnsTunneling=true
firewall=true
```

- [ ] **步骤 3：创建 `/etc/wsl.conf`**

进入 WSL 后：

```bash
sudo tee /etc/wsl.conf << 'EOF'
[boot]
systemd=true
[user]
default=alice
EOF
```

- [ ] **步骤 4：重启 WSL 使配置生效**

```powershell
# PowerShell
wsl --shutdown
wsl
```

- [ ] **步骤 5：验证代理和网络**

```bash
# 确认代理自动生效
echo $http_proxy
# 预期输出: http://127.0.0.1:7897

# 确认外网可达
curl -s -o /dev/null -w "%{http_code}" https://www.google.com
# 预期: 200

# 确认 WSL 版本
wsl -l -v
# 预期: Ubuntu-24.04 running, version 2
```

---

## 任务 2：5060Ti — Docker Engine 安装

**地点：** 5060Ti WSL Ubuntu

- [ ] **步骤 1：安装 Docker CE**

```bash
curl -fsSL https://get.docker.com | sudo sh
```

- [ ] **步骤 2：将当前用户加入 docker 组**

```bash
sudo usermod -aG docker $USER
newgrp docker
```

- [ ] **步骤 3：验证 Docker 运行**

```bash
docker ps
# 预期: 不报错 (CONTAINER ID 列为空)
docker info | grep "Server Version"
# 预期: Server Version >= 24
```
#倘若失败（成功则直接执行任务3）

#Docker 官方脚本下载被代理重置了

#换 Ubuntu 自带源安装（apt 刚才已经跑通了）：

```bash
sudo apt install -y docker.io
sudo usermod -aG docker $USER
```
  退出再进 WSL：
```bash
wsl --shutdown
wsl
```
  验证：
```bash
docker ps
docker info | grep "Server Version"
```
---

## 任务 3：5060Ti — NVIDIA Container Toolkit 安装

**地点：** 5060Ti WSL Ubuntu

- [ ] **步骤 1：添加 NVIDIA apt 源**

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

- [ ] **步骤 2：安装 nvidia-container-toolkit**

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

- [ ] **步骤 3：配置 Docker NVIDIA 运行时**

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

- [ ] **步骤 4：验证 GPU 可访问**

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

预期输出类似：

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI ...  Driver Version: ...      CUDA Version: 12.8                           |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti ... |  ...  ...  ...  |
+-------------------------------+----------------------+----------------------+
```

> 如果这步失败，先确认 Windows 端 5060Ti 的 NVIDIA 驱动已装好（你的原话），且版本 >= 525.60.11（WSL2 CUDA 的最低要求）。

#代替方案：配 Docker 走代理
```bash
# 创建 Docker 代理配置
sudo mkdir -p /etc/systemd/system/docker.service.d

sudo tee /etc/systemd/system/docker.service.d/proxy.conf << 'EOF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7897"
Environment="HTTPS_PROXY=http://127.0.0.1:7897"
Environment="NO_PROXY=localhost,127.0.0.1"
EOF

# 重载配置并重启 Docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 任务 4：5060Ti — 项目目录与文件准备

**地点：** 5060Ti WSL Ubuntu

- [ ] **步骤 1：创建目录结构**

```bash
mkdir -p ~/yolo-service/src/yolo
mkdir -p ~/yolo-service/models
mkdir -p ~/yolo-service/data/input
mkdir -p ~/yolo-service/data/output
```

- [ ] **步骤 2：从本机拷贝 Cloudflare Tunnel 配置**

在 5060Ti 上：

```bash
mkdir -p ~/cloudflared
```

从本机拷贝 `~/cloudflared/` 目录下的 `config.yml` 和 `6885fde8-...json`（凭证文件）。方法：U 盘、scp、或同一局域网下用 `scp`：

```bash
# 示例：如果两台机器在同一局域网，从本机传
# scp alice@<本机IP>:~/cloudflared/* ~/cloudflared/
```

> 注意：**两台机器不能同时启动 cloudflare_tunnel 容器**。迁移完成前 5060Ti 只配置不启动隧道。

- [ ] **步骤 3：创建 `requirements.txt`**

```bash
cat > ~/yolo-service/requirements.txt << 'EOF'
fastapi==0.110.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
EOF
```

- [ ] **步骤 4：创建 `Dockerfile`**

```bash
cat > ~/yolo-service/Dockerfile << 'DOCKERFILE'
FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
EXPOSE 8000
CMD ["uvicorn", "src.yolo.infer:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
DOCKERFILE
```

- [ ] **步骤 5：创建 `src/yolo/infer.py`（GPU 版）**

```bash
cat > ~/yolo-service/src/yolo/infer.py << 'PYEOF'
# src/yolo/infer.py
import time
import os
import uuid
import mimetypes
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="YOLOv11m Detection Service")

MODEL_PATH = "/models/yolo11m.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"模型加载失败，请检查 {MODEL_PATH} 是否存在。详细信息: {str(e)}")

INPUT_DIR = "/data/input"
OUTPUT_DIR = "/data/output/predicts"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

class DetectionRequest(BaseModel):
    image_path: str
    conf_threshold: float = 0.25
    save_annotated: bool = True

class BatchDetectionRequest(BaseModel):
    image_filenames: List[str]
    conf_threshold: float = 0.25
    save_annotated: bool = True

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _validate_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _validate_image_content(file_bytes: bytes) -> bool:
    try:
        import io
        Image.open(io.BytesIO(file_bytes)).verify()
        return True
    except Exception:
        return False

def _sanitize_filename(name: str) -> str:
    name = name.replace("\\", "_").replace("/", "_").replace("\x00", "")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    safe = safe.lstrip(".")
    return safe or "unnamed"

def _unique_filename(original: str) -> str:
    return f"{uuid.uuid4().hex[:8]}_{_sanitize_filename(original)}"

def _image_meta(dir_path: str, filename: str, img_type: str) -> dict:
    full = os.path.join(dir_path, filename)
    st = os.stat(full)
    return {
        "filename": filename,
        "storage_path": full,
        "size_bytes": st.st_size,
        "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
        "type": img_type,
    }

def _scan_images(directory: str, img_type: str) -> List[dict]:
    if not os.path.isdir(directory):
        return []
    results = []
    for f in os.listdir(directory):
        fp = os.path.join(directory, f)
        if os.path.isfile(fp) and _validate_extension(f):
            results.append(_image_meta(directory, f, img_type))
    results.sort(key=lambda x: x["modified_utc"], reverse=True)
    return results

def _safe_resolve(relative_name: str, base_dir: str) -> str:
    if not relative_name or ".." in relative_name or "/" in relative_name or "\\" in relative_name:
        raise HTTPException(400, "Invalid filename.")
    resolved = os.path.abspath(os.path.join(base_dir, relative_name))
    if not resolved.startswith(os.path.abspath(base_dir)):
        raise HTTPException(400, "Path traversal denied.")
    return resolved

# ---------------------------------------------------------------------------
# Existing endpoint
# ---------------------------------------------------------------------------

@app.post("/detect")
async def detect_image(request: DetectionRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Input image not found.")

    start_time = time.time()
    results = model.predict(
        source=request.image_path,
        conf=request.conf_threshold,
        device=0,
        save=request.save_annotated,
        project="/data/output",
        name="predicts",
        exist_ok=True,
    )
    result = results[0]
    inference_ms = round((time.time() - start_time) * 1000, 2)

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            b = box.xyxy[0].tolist()
            c = int(box.cls[0].item())
            conf = round(box.conf[0].item(), 3)
            detections.append({
                "class_id": c,
                "class_name": model.names[c],
                "confidence": conf,
                "bbox_xyxy": [round(x, 1) for x in b],
            })

    response = {
        "input_path": request.image_path,
        "detections": detections,
        "runtime": {"device": "cuda", "inference_ms": inference_ms},
    }
    if request.save_annotated:
        response["annotated_path"] = os.path.join(result.save_dir, os.path.basename(request.image_path))
    return response

# ---------------------------------------------------------------------------
# Upload endpoints
# ---------------------------------------------------------------------------

@app.post("/upload", status_code=201)
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file provided.")
    if not _validate_extension(file.filename):
        raise HTTPException(400, f"File type not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)}MB.")
    if not _validate_image_content(content):
        raise HTTPException(400, "File content is not a valid image.")
    saved_name = _unique_filename(file.filename)
    dest = os.path.join(INPUT_DIR, saved_name)
    os.makedirs(INPUT_DIR, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(content)
    return {
        "filename": saved_name,
        "original_filename": _sanitize_filename(file.filename),
        "storage_path": dest,
        "size": len(content),
        "content_type": file.content_type,
    }

@app.post("/upload/batch", status_code=201)
async def upload_images(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files provided.")
    uploaded, failed = [], []
    for f in files:
        try:
            if not f.filename or not _validate_extension(f.filename):
                failed.append({"filename": f.filename or "unknown", "reason": "File type not allowed."})
                continue
            content = await f.read()
            if len(content) > MAX_UPLOAD_SIZE:
                failed.append({"filename": f.filename, "reason": "File too large."})
                continue
            if not _validate_image_content(content):
                failed.append({"filename": f.filename, "reason": "Not a valid image."})
                continue
            saved_name = _unique_filename(f.filename)
            dest = os.path.join(INPUT_DIR, saved_name)
            os.makedirs(INPUT_DIR, exist_ok=True)
            with open(dest, "wb") as fw:
                fw.write(content)
            uploaded.append({"filename": saved_name, "original_filename": _sanitize_filename(f.filename), "storage_path": dest, "size": len(content)})
        except Exception as e:
            failed.append({"filename": f.filename or "unknown", "reason": str(e)})
    return {"uploaded": uploaded, "failed": failed, "total": len(uploaded) + len(failed)}

# ---------------------------------------------------------------------------
# List images
# ---------------------------------------------------------------------------

@app.get("/images")
async def list_images(source: str = Query("input", pattern="^(input|annotated|all)$")):
    if source == "input":
        imgs = _scan_images(INPUT_DIR, "input")
    elif source == "annotated":
        imgs = _scan_images(OUTPUT_DIR, "annotated")
    else:
        imgs = _scan_images(INPUT_DIR, "input") + _scan_images(OUTPUT_DIR, "annotated")
    return {"images": imgs, "count": len(imgs)}

# ---------------------------------------------------------------------------
# Serve image file
# ---------------------------------------------------------------------------

@app.get("/images/{filename}")
async def get_image(filename: str, type: str = Query("input", pattern="^(input|annotated)$")):
    base = INPUT_DIR if type == "input" else OUTPUT_DIR
    safe_path = _safe_resolve(filename, base)
    if not os.path.isfile(safe_path):
        raise HTTPException(404, "Image not found.")
    media_type, _ = mimetypes.guess_type(safe_path)
    if media_type is None:
        media_type = "application/octet-stream"
    return FileResponse(safe_path, media_type=media_type, headers={
        "Content-Disposition": "inline",
        "Cache-Control": "public, max-age=3600",
    })

# ---------------------------------------------------------------------------
# Batch detection
# ---------------------------------------------------------------------------

@app.post("/detect/batch")
async def detect_batch(request: BatchDetectionRequest):
    if not request.image_filenames:
        raise HTTPException(400, "No image filenames provided.")
    valid_paths, errors = [], []
    for name in request.image_filenames:
        safe_name = _sanitize_filename(name)
        try:
            full = _safe_resolve(safe_name, INPUT_DIR)
            if not os.path.isfile(full):
                errors.append({"filename": name, "error": "File not found in input directory."})
            else:
                valid_paths.append(full)
        except HTTPException:
            errors.append({"filename": name, "error": "Invalid filename."})
    if not valid_paths:
        raise HTTPException(400, {"detail": "No valid images to process.", "errors": errors})
    total_start = time.time()
    results = model.predict(
        source=valid_paths,
        conf=request.conf_threshold,
        device=0,
        save=request.save_annotated,
        project="/data/output",
        name="predicts",
        exist_ok=True,
    )
    total_ms = round((time.time() - total_start) * 1000, 2)
    per_image = []
    for i, r in enumerate(results):
        inf_ms = round(r.speed.get("inference", 0), 2) if hasattr(r, "speed") else 0
        dets = []
        if r.boxes is not None:
            for box in r.boxes:
                b = box.xyxy[0].tolist()
                c = int(box.cls[0].item())
                conf = round(box.conf[0].item(), 3)
                dets.append({"class_id": c, "class_name": model.names[c], "confidence": conf, "bbox_xyxy": [round(x, 1) for x in b]})
        entry = {"input_path": valid_paths[i], "detections": dets, "runtime": {"device": "cuda", "inference_ms": inf_ms}}
        if request.save_annotated:
            entry["annotated_path"] = os.path.join(r.save_dir, os.path.basename(valid_paths[i]))
        per_image.append(entry)
    return {"results": per_image, "errors": errors if errors else None, "total_runtime_ms": total_ms, "image_count": len(per_image)}
PYEOF
```

> 和本机版本有两个区别：`device=0`（GPU 推理）和 `runtime.device` 返回 `"cuda"`。

- [ ] **步骤 6：下载模型权重**

```bash
wget -O ~/yolo-service/models/yolo11m.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
```

> 如果直连慢，用 `curl -L` 或镜像 `https://github.moeyy.xyz/https://github.com/...`。

---

## 任务 5：5060Ti — 构建并测试 YOLO 服务容器

- [ ] **步骤 1：构建 GPU 版镜像**

```bash
cd ~/yolo-service
docker build -t yolo-api:gpu .
```

> Ultralytics 官方镜像较大（~10GB），第一次构建需要下载，确保网络畅通。

- [ ] **步骤 2：启动 YOLO 服务**

```bash
docker run -d \
  --name yolo_service \
  --network host \
  --gpus all \
  --restart always \
  -v ~/yolo-service/models:/models \
  -v ~/yolo-service/data:/data \
  yolo-api:gpu
```

- [ ] **步骤 3：确认服务启动 + GPU 识别**

```bash
docker logs yolo_service
```

预期日志中应有 Ultralytics 加载模型的信息，且 `device=0` 应自动映射到 5060Ti。查看完整日志确认没有 CUDA 相关报错。

- [ ] **步骤 4：本机上测试 GPU 推理速度**（准备一张测试图）

```bash
# 先上传一张测试图
cp ~/yolo-service/data/input/test.jpg ~/test_infer.jpg 2>/dev/null || \
  wget -O ~/test_infer.jpg https://ultralytics.com/images/bus.jpg

curl -X POST http://127.0.0.1:8000/upload -F "file=@/home/alice/test_infer.jpg"

# 检测（注意文件夹下的bus.jpg应该换成真实图片名）
curl -s -X POST http://127.0.0.1:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/data/input/bus.jpg"}' | python3 -m json.tool
```

注意 `runtime.inference_ms` 应该明显快于本机的 ~150ms（CPU: Ultra 7 155H），预估 GPU (5060Ti) 推理时间在 **10-30ms**。

- [ ] **步骤 5：验证所有 API 端点**

```bash
# 上传
curl -s -X POST http://127.0.0.1:8000/upload -F "file=@~/test_infer.jpg" | python3 -m json.tool

# 列表
curl -s http://127.0.0.1:8000/images | python3 -m json.tool

# 批量检测
curl -s -X POST http://127.0.0.1:8000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"image_filenames":["bus.jpg"]}' | python3 -m json.tool

# 查看标注图
curl -s -o /tmp/annotated.jpg "http://127.0.0.1:8000/images/bus.jpg?type=annotated"
ls -la /tmp/annotated.jpg

# 路径穿越防护
curl -s "http://127.0.0.1:8000/images/../../../etc/passwd"

# 非图片文件拒绝
echo "not an image" > /tmp/fake.txt
curl -s -X POST http://127.0.0.1:8000/upload -F "file=@/tmp/fake.txt"
```

---

## 任务 6：5060Ti — 启动 Cloudflare Tunnel

> ⚠️ **重要：** 启动隧道前，先确认本机的 `cloudflare_tunnel` 容器已停止，否则隧道 UUID 冲突。

- [ ] **步骤 1：停止本机的隧道（在本机执行）**

```powershell
# [本机] PowerShell
wsl -d Ubuntu-24.04 bash -c 'docker stop cloudflare_tunnel'
```

- [ ] **步骤 2：在 5060Ti 上启动隧道**

```bash
docker run -d \
  --name cloudflare_tunnel \
  --network host \
  --restart always \
  -v ~/cloudflared:/etc/cloudflared/ \
  cloudflare/cloudflared:latest \
  tunnel --config /etc/cloudflared/config.yml run
```

- [ ] **步骤 3：确认隧道注册成功**

```bash
docker logs cloudflare_tunnel
```

预期输出中有 `Registered tunnel connection connIndex=0...` 字样。

#调大 UDP 缓冲区：
```bash
echo alice123 | sudo -S sysctl -w net.core.rmem_max=7500000
echo alice123 | sudo -S sysctl -w net.core.wmem_max=7500000
docker restart cloudflare_tunnel
sleep 15
docker logs cloudflare_tunnel --tail 5
```
#确认没有 failed to sufficiently increase receive buffer 警告后，直接从 5060Ti WSL 测试公网连通性：
```bash
curl -s -X POST "https://yolo.alice1.xyz/detect" \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/data/input/d370096b_test_infer.jpg"}' | python3 -m json.tool
```

- [ ] **步骤 4：公网验证**

```powershell
# [本机或5060Ti] Windows PowerShell 或其他联网设备
curl.exe -X POST "https://yolo.alice1.xyz/detect" ^
  -H "Content-Type: application/json" ^
  -d "{\"image_path\":\"/data/input/test.jpg\",\"conf_threshold\":0.25}"
```

预期：返回检测结果的 JSON，状态码 200。

---

## 任务 7：本机清理

- [ ] **步骤 1：停止并清理本机 Docker**

```powershell
# [本机] PowerShell
wsl -d Ubuntu-24.04 bash -c '
docker stop yolo_service cloudflare_tunnel 2>/dev/null
docker system prune -af
'
```

- [ ] **步骤 2：确认公网仍可访问**

```powershell
# [本机] 确认 5060Ti 上的服务正常
curl.exe -X POST "https://yolo.alice1.xyz/detect" ^
  -H "Content-Type: application/json" ^
  -d "{\"image_path\":\"/data/input/test.jpg\"}"
```

---

## 回滚方案

如果迁移后 5060Ti 上服务异常，回滚到本机的步骤：

### 回滚条件

- GPU 驱动/NVIDIA Container Toolkit 安装失败，`nvidia-smi` 容器内不可见
- YOLO 服务在 GPU 下推理结果异常（精度问题可能性低，但需确认）
- Cloudflare Tunnel 在 5060Ti 上无法注册

### 回滚步骤

- [ ] **快速回滚：启本机，停 5060Ti**

```powershell
# [5060Ti] 停止容器
wsl -d Ubuntu-24.04 bash -c 'docker stop yolo_service cloudflare_tunnel'

# [本机] 重新启动
wsl -d Ubuntu-24.04 bash -c 'docker start yolo_service cloudflare_tunnel'
```

- [ ] **完全回滚：彻底清理 5060Ti 环境**

```bash
# 清理所有容器/镜像
docker system prune -af

# 删除项目文件（可选）
rm -rf ~/yolo-service
rm -rf ~/cloudflared
```

---

## 迁移速度预期

| 场景 | 本机 (Ultra 7 155H CPU) | 5060Ti (RTX 5060Ti GPU) | 提升 |
|------|------------------------|------------------------|------|
| 单图推理 | ~150ms | 预估 10-30ms | 5-15x |
| 批量 4 图 | ~600ms | 预估 40-120ms | 5-15x |
| 上传速度 | 取决于网络，与本机一致 | 取决于网络，与本机一致 | — |
