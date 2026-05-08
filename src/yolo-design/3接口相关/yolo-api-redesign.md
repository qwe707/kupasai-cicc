# YOLO 检测服务 API 重构设计方案

**日期：** 2026-05-02
**背景：** 将 5060Ti 上现有的单文件臃肿 API 重构为清晰的异步任务体系，支持上传/本地路径双模式、进度查询、批量 GPU 加速。

---

## 1. 架构

```
5060Ti — FastAPI (uvicorn, 1 worker)

POST /detect/upload  ──── 上传文件提交任务
POST /detect/local   ──── 本地路径提交任务
GET  /tasks          ──── 查进度 / 查结果
GET  /images/{task_id}/{filename}  ──── 查看标注图（直链）

提交 → 后台线程 → 更新进度 → 写出标注图+JSON
                          
文件布局:
data/
├── input/{task_id}/{filename}             ← 输入图片
└── output/
    ├── predicts_img/{task_id}/{filename}  ← 标注图
    └── predicts_info/{task_id}/{filename}.json  ← 检测信息
```

## 2. 接口定义

### 2.1 POST /detect/upload — 上传文件提交任务

上传单张或多张图片，启动异步检测。

**请求：** `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | 是 | 任务标识（由调用者指定） |
| `files` | File[] | 是 | 图片文件，支持多选 |

**返回：** `202 Accepted`

```json
{"task_id": "my_task_001", "status": "accepted", "image_count": 2}
```

### 2.2 POST /detect/local — 本地路径提交任务

调用者直接传图片在服务端磁盘上的路径，不走上传。

**请求：** `application/json`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | string | 是 | 任务标识 |
| `image_paths` | string[] | 是 | 图片的绝对路径列表 |

```json
{"task_id": "batch_002", "image_paths": ["/data/raw/img1.jpg", "/mnt/d/photos/img2.png"]}
```

**返回：** `202 Accepted`

```json
{"task_id": "batch_002", "status": "accepted", "image_count": 2}
```

**处理方式：** 将路径列表中的文件复制到 `data/input/{task_id}/` 下（不硬链接，因为跨盘不兼容）。

### 2.3 GET /tasks — 查进度 / 查结果

**查全部任务（无参数）：**

```bash
curl http://localhost:8000/tasks
```

```json
{"tasks": ["my_task_001", "batch_002"], "count": 2}
```

**查指定任务进度（?task_id）：**

```bash
curl "http://localhost:8000/tasks?task_id=my_task_001"
```

```json
{"task_id": "my_task_001", "status": "running", "progress": 50, "completed": 1, "total": 2}
```

状态流转：`accepted` → `running` → `done` | `error`

**查指定任务结果（?task_id，结果已就绪时）：**

```json
{
  "task_id": "my_task_001",
  "status": "done",
  "results": [
    {
      "input_path": "/data/input/my_task_001/img1.jpg",
      "detections": [
        {"class_id": 0, "class_name": "person", "confidence": 0.927, "bbox_xyxy": [124.2, 198.8, 1090.2, 712.3]}
      ],
      "runtime": {"device": "cuda", "inference_ms": 19.4},
      "predicts_img_url": "https://yolo.alice1.xyz/images/my_task_001/img1.jpg",
      "predicts_info_url": "https://yolo.alice1.xyz/tasks?task_id=my_task_001&file=img1.jpg"
    }
  ]
}
```

**查单张图片的 JSON（?task_id + ?file）：**

```bash
curl "http://localhost:8000/tasks?task_id=my_task_001&file=img1.jpg"
```

返回该图片对应的完整检测 JSON（与 predicts_info 下的文件内容一致）。

**查看标注图：**

```bash
curl "https://yolo.alice1.xyz/images/my_task_001/img1.jpg"
```

新增 `GET /images/{task_id}/{filename}` 端点，直接返回图片二进制，浏览器可直接打开显示。

## 3. 文件布局

```
data/
├── input/
│   └── {task_id}/
│       ├── img1.jpg           ← 上传或复制来的原始图片
│       └── img2.jpg
└── output/
    ├── predicts_img/
    │   └── {task_id}/
    │       ├── img1.jpg       ← YOLO 标注后的图片
    │       └── img2.jpg
    └── predicts_info/
        └── {task_id}/
            ├── img1.json      ← 单张图片检测结果
            └── img2.json
```

每张图的 JSON 格式：

```json
{
  "input_path": "/data/input/{task_id}/img1.jpg",
  "detections": [
    {"class_id": 0, "class_name": "person", "confidence": 0.927, "bbox_xyxy": [124.2, 198.8, 1090.2, 712.3]}
  ],
  "runtime": {"device": "cuda", "inference_ms": 19.4},
  "predicts_img_url": "https://yolo.alice1.xyz/images/{task_id}/img1.jpg",
  "predicts_info_url": "https://yolo.alice1.xyz/tasks?task_id={task_id}&file=img1.jpg"
}
```

## 4. 后台任务机制

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

_executor = ThreadPoolExecutor(max_workers=1)  # GPU 串行
_task_progress: dict[str, dict] = {}  # 内存进度表

def _init_task(task_id: str, total: int):
    _task_progress[task_id] = {"status": "accepted", "total": total, "completed": 0, "progress": 0}

def _update_progress(task_id: str, completed: int):
    total = _task_progress[task_id]["total"]
    _task_progress[task_id].update(completed=completed, progress=int(completed / total * 100))

def _run_detection(task_id: str, image_paths: list[str]):
    _task_progress[task_id]["status"] = "running"
    for i, path in enumerate(image_paths):
        # 检测单张
        results = model.predict(source=path, device=0, batch=16, ...)
        # 保存标注图 + JSON
        ...
        _update_progress(task_id, i + 1)
    _task_progress[task_id]["status"] = "done"

# FastAPI 端点中启动后台线程
@app.post("/detect/upload")
async def detect_upload(...):
    _init_task(task_id, len(files))
    asyncio.get_event_loop().run_in_executor(_executor, _run_detection, task_id, saved_paths)
    return {"task_id": task_id, "status": "accepted", "image_count": n}
```

## 5. 多图加速

使用 Ultralytics 的 GPU 批量推理，无需 TensorRT：

```python
results = model.predict(
    source=image_paths,  # 传列表，自动批量
    device=0,
    batch=16,            # 批量大小
    imgsz=640,           # 固定尺寸，避免动态 shape 开销
    conf=0.25,
    save=False,          # 自己控制保存逻辑
)
```

- `batch=16` 对 5060Ti 16GB 显存安全（YOLOv11m 单图 ~500MB）
- 预估 16 图批处理耗时 **80-120ms**（vs 单张 19ms × 16 = 304ms）

## 6. 代码拆分

将现在单文件 `infer.py` 拆为 3 个文件：

```
src/yolo/
├── infer.py         ← FastAPI app + 路由（3 个端点），约 100 行
├── detector.py      ← YOLO 推理 + 后台任务管理，约 100 行
└── schemas.py       ← Pydantic 请求/响应模型
```

## 7. 旧接口清理

旧的 `POST /detect`、`POST /upload`、`POST /upload/batch`、`POST /detect/batch`、`GET /images`、`GET /images/{filename}` **全部移除**。原有功能由新接口替代：

| 旧接口 | 替代 |
|--------|------|
| `POST /detect` | `POST /detect/upload` 或 `POST /detect/local`（异步） |
| `POST /upload` + `/upload/batch` | `POST /detect/upload`（文件上传合入任务提交） |
| `POST /detect/batch` | 由后台线程自动批量处理 |
| `GET /images` | `GET /tasks` |
| `GET /images/{filename}` | `GET /images/{task_id}/{filename}`（直链） |

## 8. 验证清单

| # | 验证项 | 命令 | 预期 |
|---|--------|------|------|
| 1 | 上传提交 | `curl -X POST ... -F "task_id=t1" -F "files=@img.jpg"` | 202, task_id |
| 2 | 本地路径提交 | `curl -X POST ... -d '{"task_id":"t2","image_paths":["..."]}'` | 202 |
| 3 | 查进度 | `curl /tasks?task_id=t1` | progress + status |
| 4 | 查结果 | 等待 status=done 后查 | 完整 detections |
| 5 | 查单文件 JSON | `curl /tasks?task_id=t1&file=img.jpg` | 单张检测结果 |
| 6 | 查全部任务 | `curl /tasks` | 任务列表 |
| 7 | 非图片文件拒绝 | 上传 .py 文件 | 400 |
| 8 | 旧接口仍正常 | curl 旧 /detect | 200 |
