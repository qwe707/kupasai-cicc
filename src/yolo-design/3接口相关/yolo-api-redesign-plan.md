# YOLO API 重构实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 5060Ti 上的 YOLO 检测服务 API 从单文件臃肿设计重构为异步任务体系：上传/本地路径双模式提交、后台批量 GPU 推理、进度查询、结果检索。

**架构：** 将 `infer.py` 拆为 `infer.py`（路由）+ `detector.py`（推理核心 + 后台任务）+ `schemas.py`（数据模型），删除旧接口，新增 `POST /detect/upload`、`POST /detect/local`、`GET /tasks` 三个端点。

**技术栈：** FastAPI / Ultralytics / CUDA / ThreadPoolExecutor

**执行地点：** 5060Ti WSL Ubuntu，`~/yolo-service/`

---

## 文件结构

修改前：
```
~/yolo-service/src/yolo/
├── infer.py   ← 所有代码混在一起（路由 + 推理 + 模型）
```

修改后：
```
~/yolo-service/src/yolo/
├── infer.py       ← 仅包含 FastAPI app + 3 个路由端点（~80 行）
├── detector.py    ← YOLO 推理 + 后台任务管理 + 进度表（~100 行）
└── schemas.py     ← Pydantic 请求/响应模型（~40 行）
```

---

## 任务 1：创建 `schemas.py` — 数据模型

**文件：** `~/yolo-service/src/yolo/schemas.py`（新建）

- [ ] **步骤 1：编写 schemas.py**

```python
from pydantic import BaseModel
from typing import List, Optional

class LocalTaskRequest(BaseModel):
    """POST /detect/local 的请求体"""
    task_id: str
    image_paths: List[str]

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str       # accepted | running | done | error
    progress: int = 0
    completed: int = 0
    total: int = 0

class TaskListResponse(BaseModel):
    tasks: List[str]
    count: int

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]

class RuntimeInfo(BaseModel):
    device: str
    inference_ms: float

class ImageResult(BaseModel):
    input_path: str
    detections: List[Detection]
    runtime: RuntimeInfo
    predicts_img_url: str
    predicts_info_url: str

class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    results: Optional[List[ImageResult]] = None
```

- [ ] **步骤 2：验证语法**

```bash
python3 -c "import ast; ast.parse(open('/home/alice/yolo-service/src/yolo/schemas.py').read()); print('OK')"
```

预期：`OK`

---

## 任务 2：创建 `detector.py` — 推理核心 + 后台任务

**文件：** `~/yolo-service/src/yolo/detector.py`（新建）

- [ ] **步骤 1：编写 detector.py**

```python
# src/yolo/detector.py
import os
import json
import time
import asyncio
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

MODEL_PATH = "/models/yolo11m.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"模型加载失败: {e}")

INPUT_DIR = "/data/input"
OUTPUT_IMG_DIR = "/data/output/predicts_img"
OUTPUT_INFO_DIR = "/data/output/predicts_info"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

_executor = ThreadPoolExecutor(max_workers=1)
_task_progress: dict[str, dict] = {}

def init_task(task_id: str, total: int):
    _task_progress[task_id] = {
        "status": "accepted", "total": total,
        "completed": 0, "progress": 0,
    }

def update_progress(task_id: str, completed: int):
    if task_id not in _task_progress:
        return
    total = _task_progress[task_id]["total"]
    _task_progress[task_id].update(
        completed=completed, progress=int(completed / total * 100),
    )

def get_task_status(task_id: str) -> Optional[dict]:
    return _task_progress.get(task_id)

def list_tasks() -> List[str]:
    return list(_task_progress.keys())

def set_task_error(task_id: str, msg: str):
    if task_id in _task_progress:
        _task_progress[task_id]["status"] = "error"
        _task_progress[task_id]["error"] = msg

def validate_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def validate_image_content(file_bytes: bytes) -> bool:
    try:
        import io
        Image.open(io.BytesIO(file_bytes)).verify()
        return True
    except Exception:
        return False

def sanitize_filename(name: str) -> str:
    name = name.replace("\\", "_").replace("/", "_").replace("\x00", "")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    safe = safe.lstrip(".")
    return safe or "unnamed"

def scan_input_images(task_id: str) -> List[str]:
    dir_path = os.path.join(INPUT_DIR, task_id)
    if not os.path.isdir(dir_path):
        return []
    result = []
    for f in sorted(os.listdir(dir_path)):
        fp = os.path.join(dir_path, f)
        if os.path.isfile(fp) and validate_extension(f):
            result.append(fp)
    return result

def scan_info_files(task_id: str) -> List[str]:
    dir_path = os.path.join(OUTPUT_INFO_DIR, task_id)
    if not os.path.isdir(dir_path):
        return []
    result = []
    for f in sorted(os.listdir(dir_path)):
        fp = os.path.join(dir_path, f)
        if os.path.isfile(fp) and f.endswith(".json"):
            result.append(fp)
    return result

def _build_public_url(task_id: str, filename: str, type: str) -> str:
    base = "https://yolo.alice1.xyz"
    if type == "img":
        return f"{base}/tasks?task_id={task_id}&file={filename}&type=annotated"
    return f"{base}/tasks?task_id={task_id}&file={filename}"

def run_detection(task_id: str):
    _task_progress[task_id]["status"] = "running"
    image_paths = scan_input_images(task_id)
    if not image_paths:
        set_task_error(task_id, "No images found.")
        return

    total = len(image_paths)
    img_out_dir = os.path.join(OUTPUT_IMG_DIR, task_id)
    info_out_dir = os.path.join(OUTPUT_INFO_DIR, task_id)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(info_out_dir, exist_ok=True)

    start_time = time.time()
    results = model.predict(
        source=image_paths, device=0, batch=16, imgsz=640,
        conf=0.25, save=False,
    )
    total_ms = round((time.time() - start_time) * 1000, 2)
    per_img_ms = round(total_ms / total, 2)

    for i, (img_path, r) in enumerate(zip(image_paths, results)):
        filename = os.path.basename(img_path)
        name_stem = Path(filename).stem

        dets = []
        if r.boxes is not None:
            for box in r.boxes:
                b = box.xyxy[0].tolist()
                c = int(box.cls[0].item())
                conf = round(box.conf[0].item(), 3)
                dets.append({
                    "class_id": c, "class_name": model.names[c],
                    "confidence": conf, "bbox_xyxy": [round(x, 1) for x in b],
                })

        # 手动保存标注图（r.plot() 返回 BGR numpy 数组）
        im_bgr = r.plot()
        out_img_path = os.path.join(img_out_dir, filename)
        cv2.imwrite(out_img_path, im_bgr)

        # 写 JSON
        result_json = {
            "input_path": img_path,
            "detections": dets,
            "runtime": {"device": "cuda", "inference_ms": per_img_ms},
            "predicts_img_url": _build_public_url(task_id, filename, "img"),
            "predicts_info_url": _build_public_url(task_id, filename, "info"),
        }

        json_path = os.path.join(info_out_dir, f"{name_stem}.json")
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=2)

        update_progress(task_id, i + 1)

    _task_progress[task_id]["status"] = "done"

def submit_task(task_id: str):
    asyncio.get_event_loop().run_in_executor(_executor, run_detection, task_id)
```

- [ ] **步骤 2：验证语法**

```bash
python3 -c "import ast; ast.parse(open('/home/alice/yolo-service/src/yolo/detector.py').read()); print('OK')"
```

预期：`OK`

---

## 任务 3：重写 `infer.py` — 仅保留路由

**文件：** `~/yolo-service/src/yolo/infer.py`（重写，原内容备份）

- [ ] **步骤 1：备份原文件**

```bash
cp ~/yolo-service/src/yolo/infer.py ~/yolo-service/src/yolo/infer.py.bak
```

- [ ] **步骤 2：重写 infer.py**

```python
# src/yolo/infer.py
import os
import json
import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from . import schemas
from . import detector

app = FastAPI(title="YOLOv11m Detection Service (Async)")

# ---------------------------------------------------------------------------
# POST /detect/upload — 上传文件提交任务
# ---------------------------------------------------------------------------

@app.post("/detect/upload")
async def detect_upload(task_id: str = Query(...), files: list[UploadFile] = File(...)):
    if not task_id:
        raise HTTPException(400, "task_id is required.")
    if not files:
        raise HTTPException(400, "At least one file is required.")

    # 创建任务输入目录
    input_dir = os.path.join(detector.INPUT_DIR, task_id)
    os.makedirs(input_dir, exist_ok=True)

    saved_count = 0
    for f in files:
        if not f.filename:
            continue
        if not detector.validate_extension(f.filename):
            continue
        content = await f.read()
        if len(content) > detector.MAX_UPLOAD_SIZE:
            continue
        if not detector.validate_image_content(content):
            continue
        dest = os.path.join(input_dir, detector.sanitize_filename(f.filename))
        with open(dest, "wb") as fw:
            fw.write(content)
        saved_count += 1

    if saved_count == 0:
        raise HTTPException(400, "No valid images were uploaded.")

    detector.init_task(task_id, saved_count)
    detector.submit_task(task_id)

    return {"task_id": task_id, "status": "accepted", "image_count": saved_count}

# ---------------------------------------------------------------------------
# POST /detect/local — 本地路径提交任务
# ---------------------------------------------------------------------------

@app.post("/detect/local")
async def detect_local(req: schemas.LocalTaskRequest):
    if not req.task_id:
        raise HTTPException(400, "task_id is required.")
    if not req.image_paths:
        raise HTTPException(400, "At least one image_path is required.")

    import shutil
    input_dir = os.path.join(detector.INPUT_DIR, req.task_id)
    os.makedirs(input_dir, exist_ok=True)

    copied = 0
    for src in req.image_paths:
        src = os.path.abspath(src)
        if not os.path.isfile(src):
            continue
        if not detector.validate_extension(src):
            continue
        dest = os.path.join(input_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception:
            continue

    if copied == 0:
        raise HTTPException(400, "No valid images could be copied.")

    detector.init_task(req.task_id, copied)
    detector.submit_task(req.task_id)

    return {"task_id": req.task_id, "status": "accepted", "image_count": copied}

# ---------------------------------------------------------------------------
# GET /tasks — 查进度 / 查结果
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def get_tasks(
    task_id: str = Query(None),
    file: str = Query(None),
    type: str = Query("info", regex="^(info|annotated)$"),
):
    # 查全部任务
    if not task_id and not file:
        tasks = detector.list_tasks()
        return {"tasks": tasks, "count": len(tasks)}

    # 查单张图片的 info JSON
    if task_id and file and type == "info":
        json_name = f"{Path(file).stem}.json"
        json_path = os.path.join(detector.OUTPUT_INFO_DIR, task_id, json_name)
        if not os.path.isfile(json_path):
            raise HTTPException(404, "Result not found.")
        with open(json_path) as f:
            return JSONResponse(content=json.load(f))

    # 查看标注图
    if task_id and file and type == "annotated":
        img_path = os.path.join(detector.OUTPUT_IMG_DIR, task_id, file)
        if not os.path.isfile(img_path):
            raise HTTPException(404, "Image not found.")
        media_type, _ = mimetypes.guess_type(img_path)
        if media_type is None:
            media_type = "application/octet-stream"
        return FileResponse(img_path, media_type=media_type, headers={
            "Content-Disposition": "inline",
            "Cache-Control": "public, max-age=3600",
        })

    # 查任务状态/结果
    status = detector.get_task_status(task_id)
    if not status:
        raise HTTPException(404, f"Task '{task_id}' not found.")

    if status["status"] != "done":
        return {
            "task_id": task_id,
            "status": status["status"],
            "progress": status["progress"],
            "completed": status["completed"],
            "total": status["total"],
        }

    # 返回完整结果
    info_files = sorted(detector.scan_info_files(task_id))
    results = []
    for fp in info_files:
        with open(fp) as f:
            results.append(json.load(f))

    return {
        "task_id": task_id,
        "status": "done",
        "results": results,
    }
```

- [ ] **步骤 3：验证语法**

```bash
python3 -c "import ast; ast.parse(open('/home/alice/yolo-service/src/yolo/infer.py').read()); print('OK')"
```

预期：`OK`

---

## 任务 4：构建镜像 + 重启服务

- [ ] **步骤 1：删除旧容器**

```bash
docker stop yolo_service cloudflare_tunnel
docker rm yolo_service cloudflare_tunnel
```

- [ ] **步骤 2：修改 Dockerfile**

检查 `~/yolo-service/Dockerfile`，确认 `CMD` 行不变（`uvicorn src.yolo.infer:app ...`），不需要改，因为 Python 包导入路径自动支持同目录下的 `.py` 文件。

- [ ] **步骤 3：构建新镜像**

```bash
cd ~/yolo-service
docker build -t yolo-api:async .
```

- [ ] **步骤 4：启动新容器**

```bash
docker run -d \
  --name yolo_service \
  --network host \
  --gpus all \
  --restart always \
  -v ~/yolo-service/models:/models \
  -v ~/yolo-service/data:/data \
  yolo-api:async
```

- [ ] **步骤 5：启动隧道**

```bash
docker run -d \
  --name cloudflare_tunnel \
  --network host \
  --restart always \
  -v ~/cloudflared:/etc/cloudflared/ \
  cloudflare/cloudflared:latest \
  tunnel --config /etc/cloudflared/config.yml run
```

- [ ] **步骤 6：确认服务运行**

```bash
sleep 5
docker logs yolo_service
```

预期：`Uvicorn running on http://0.0.0.0:8000`

---

## 任务 5：验证

- [ ] **① 上传提交任务：**

```bash
cp ~/yolo-service/data/input/test.jpg ~/test.jpg 2>/dev/null
curl -s -X POST "http://127.0.0.1:8000/detect/upload?task_id=test_001" \
  -F "files=@/home/alice/test.jpg" | python3 -m json.tool
```

预期：`{"task_id": "test_001", "status": "accepted", "image_count": 1}`

- [ ] **② 查进度：**

```bash
sleep 3
curl -s "http://127.0.0.1:8000/tasks?task_id=test_001" | python3 -m json.tool
```

预期：`status` 可以是 `running` 或 `done`

- [ ] **③ 查结果（完成后）：**

```bash
curl -s "http://127.0.0.1:8000/tasks?task_id=test_001" | python3 -m json.tool
```

预期：`status: "done"` + `results` 数组，包含 `predicts_img_url` 和 `predicts_info_url`

- [ ] **④ 查单张检测 JSON：**

```bash
curl -s "http://127.0.0.1:8000/tasks?task_id=test_001&file=test.jpg" | python3 -m json.tool
```

预期：返回该图片的检测结果 JSON

- [ ] **⑤ 查看标注图：**

```bash
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/tasks?task_id=test_001&file=test.jpg&type=annotated"
```

预期：200

- [ ] **⑥ 查全部任务：**

```bash
curl -s "http://127.0.0.1:8000/tasks" | python3 -m json.tool
```

预期：`{"tasks": ["test_001"], "count": 1}`

- [ ] **⑦ 本地路径提交：**

```bash
curl -s -X POST "http://127.0.0.1:8000/detect/local" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"local_test","image_paths":["/home/alice/test.jpg"]}' | python3 -m json.tool
```

预期：`{"task_id": "local_test", "status": "accepted", "image_count": 1}`

- [ ] **⑧ 旧接口确认已移除：**

```bash
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/detect"
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/upload"
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/images"
```

预期：全部返回 404

- [ ] **⑨ 非图片文件拒绝：**

```bash
echo "abc" > /tmp/fake.txt
curl -s -X POST "http://127.0.0.1:8000/detect/upload?task_id=bad" \
  -F "files=@/tmp/fake.txt"
```

预期：`{"detail": "No valid images were uploaded."}`

- [ ] **⑩ 公网验证：**

```bash
curl -s -X POST "https://yolo.alice1.xyz/detect/upload?task_id=public_test" \
  -F "files=@/home/alice/test.jpg" | python3 -m json.tool
```

预期：返回 task_id + accepted

---

## 回滚方案

如果新服务有问题，快速回到旧版：

```bash
# 停新容器
docker stop yolo_service cloudflare_tunnel
docker rm yolo_service cloudflare_tunnel

# 恢复旧 infer.py
cp ~/yolo-service/src/yolo/infer.py.bak ~/yolo-service/src/yolo/infer.py

# 重新构建并启动旧版
cd ~/yolo-service
docker build -t yolo-api:gpu .
docker run -d --name yolo_service --network host --gpus all --restart always \
  -v ~/yolo-service/models:/models -v ~/yolo-service/data:/data yolo-api:gpu
docker run -d --name cloudflare_tunnel --network host --restart always \
  -v ~/cloudflared:/etc/cloudflared/ cloudflare/cloudflared:latest \
  tunnel --config /etc/cloudflared/config.yml run
```
