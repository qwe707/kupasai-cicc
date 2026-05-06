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
    # 使用前请通过 PUBLIC_URL 环境变量替换为自己的公网地址
    base = os.getenv("PUBLIC_URL", "https://your-domain.example.com")
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