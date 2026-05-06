# YOLOv11m Detection Service

基于 FastAPI + YOLOv11m 的异步目标检测服务，支持 GPU 推理、文件上传/本地路径双模式提交、异步任务队列与进度查询。

## 文件夹结构

```
yolo-service/
├── Dockerfile                        # 基于 ultralytics/ultralytics:latest 的 GPU 镜像
├── requirements.txt                  # Python 依赖（FastAPI 栈）
├── README.md                         # 本文件
│
├── src/
│   └── yolo/
│       ├── infer.py                  # FastAPI 路由（3 个端点）
│       ├── detector.py               # YOLO 推理核心 + 后台任务 + 进度管理
│       └── schemas.py                # Pydantic 请求/响应模型
│
├── models/
│   └── yolo11m.pt                    # YOLOv11m 权重（.gitignore，启动前下载）
│
└── data/
    ├── input/
    │   └── {task_id}/                # 上传或复制来的原始图片
    │       ├── img1.jpg
    │       └── img2.jpg
    └── output/
        ├── predicts_img/
        │   └── {task_id}/            # 检测后的标注图
        │       ├── img1.jpg
        │       └── img2.jpg
        └── predicts_info/
            └── {task_id}/            # 每张图的检测结果 JSON
                ├── img1.json
                └── img2.json
```

## 架构

```
客户端                          FastAPI (5060Ti)
  │                                  │
  │─ POST /detect/upload ──────────→ │  ← 上传图片文件
  │─ POST /detect/local  ──────────→ │  ← 提交本地路径
  │─ GET  /tasks         ──────────→ │  ← 查进度 / 查结果
  │                                  │
  │                     ┌────────────┤
  │                     │ 后台线程    │
  │                     │ GPU 推理    │
  │                     │ batch=16   │
  │                     │ 更新进度    │
  │                     └────────────┤
  │                                  │
  │←─ task_id（立即返回）──────────── │
  │←─ progress（轮询）─────────────── │
  │←─ results（完成后获取）────────── │
```

## 接口

### `POST /detect/upload`
上传文件提交检测任务。

| 参数 | 类型 | 说明 |
|------|------|------|
| `task_id` | query | 任务标识 |
| `files` | form-data | 图片文件（支持多选） |

```bash
curl -X POST "https://yolo.domain.com/detect/upload?task_id=my_task" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg"
```

→ `{"task_id": "my_task", "status": "accepted", "image_count": 2}`

### `POST /detect/local`
提交本地图片路径。

```json
{"task_id": "local_task", "image_paths": ["/data/img1.jpg", "/data/img2.jpg"]}
```

→ `{"task_id": "local_task", "status": "accepted", "image_count": 2}`

### `GET /tasks`
查询任务。

```
/tasks              → 全部任务列表
/tasks?task_id=xxx  → 任务进度或结果
/tasks?task_id=xxx&file=img.jpg            → 单张图片的检测 JSON
/tasks?task_id=xxx&file=img.jpg&type=annotated → 查看标注图
```

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `PUBLIC_URL` | 是 | `https://your-domain.example.com` | 公网访问地址，返回结果中的 `predicts_img_url` 和 `predicts_info_url` 会使用此值。**部署前必须替换为自己的域名。** |

## 部署

```bash
# 1. 下载模型权重
wget -O models/yolo11m.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

# 2. 构建
docker build -t yolo-api:async .

# 3. 启动
docker run -d \
  --name yolo_service \
  --network host \
  --gpus all \
  --restart always \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  yolo-api:async
```

> GPU 推理需要宿主机安装 NVIDIA Container Toolkit。
