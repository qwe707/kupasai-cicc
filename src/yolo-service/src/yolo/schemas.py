from pydantic import BaseModel
from typing import List, Optional

class LocalTaskRequest(BaseModel):
    """POST /detect/local 的请求体"""
    task_id: str
    image_paths: List[str]
    focus_classes: str = "0"   # 关注类别，逗号分隔，如 "54,74"

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

class ScoreItem(BaseModel):
    index: int       # 目标在 detections 列表中的序号 (从 0 开始)
    score: int       # 评分值

class ScoreRequest(BaseModel):
    task_id: str
    file: str        # 图片文件名 (如 "img1.jpg")
    scores: list[ScoreItem]
