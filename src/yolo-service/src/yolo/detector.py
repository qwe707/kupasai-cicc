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
import math
import numpy as np

MODEL_PATH = "/models/yolo11m.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"模型加载失败: {e}")

INPUT_DIR = "/data/input"
OUTPUT_IMG_DIR = "/data/output/predicts_img"
OUTPUT_INFO_DIR = "/data/output/predicts_info"
STITCH_DIR = "/data/output/stitches"
FINAL_DIR = "/data/output/final"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

_executor = ThreadPoolExecutor(max_workers=1)
_task_progress: dict[str, dict] = {}

def init_task(task_id: str, total: int, focus_classes: str = "0"):
    _task_progress[task_id] = {
        "status": "accepted", "total": total,
        "completed": 0, "progress": 0,
        "focus_classes": focus_classes,
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
    base = os.getenv("PUBLIC_URL", "https://your-domain.example.com")
    if type == "img":
        return f"{base}/images/{task_id}/{filename}"
    if type == "final":
        return f"{base}/final/{task_id}/{filename}"
    return f"{base}/tasks?task_id={task_id}&file={filename}"

def run_detection(task_id: str, focus_classes: str = "0"):
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

    cls_list = [int(x.strip()) for x in focus_classes.split(",")]
    stitch_crops(task_id, focus_classes=cls_list)

    _task_progress[task_id]["status"] = "done"

def _build_stitch_url(task_id: str, filename: str) -> str:
    base = os.getenv("PUBLIC_URL", "https://your-domain.example.com")
    return f"{base}/stitches/{task_id}/{filename}"

def stitch_crops(task_id: str, focus_classes: list[int]):
    """检测完成后裁剪指定类别目标并水平拼接为一张图"""
    info_dir = os.path.join(OUTPUT_INFO_DIR, task_id)
    stitch_dir = os.path.join(STITCH_DIR, task_id)
    os.makedirs(stitch_dir, exist_ok=True)

    for json_file in sorted(os.listdir(info_dir)):
        if not json_file.endswith(".json"):
            continue
        fp = os.path.join(info_dir, json_file)
        with open(fp) as f:
            data = json.load(f)

        targets = [(i, d) for i, d in enumerate(data["detections"])
                   if d["class_id"] in focus_classes]
        if not targets:
            continue

        orig_img = cv2.imread(data["input_path"])
        crops = []
        for idx, det in targets:
            x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(orig_img.shape[1], x2)
            y2 = min(orig_img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = orig_img[y1:y2, x1:x2].copy()
            label = str(idx)
            crops.append((crop, label))

        if not crops:
            continue
        if len(crops) == 1:
            c, label = crops[0]
            bar_h = 8
            fs = 0.3
            ft = 1
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            bar = 255 * np.ones((bar_h, max(c.shape[1], tw + 12), 3), dtype=np.uint8)
            cv2.putText(bar, label, ((bar.shape[1] - tw) // 2, 8),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft)
            stitch = cv2.vconcat([c, bar])
        else:
            # 动态网格 + crop 填满：按中位高度等比缩放 → 贪心逐行填 → 行内等比缩放到同宽
            imgs = [c for c, _ in crops]
            target_h = int(np.median([c.shape[0] for c in imgs]))
            scaled = []
            for c, label in crops:
                h, w = c.shape[:2]
                new_w = max(1, int(w * target_h / h))
                cr = cv2.resize(c, (new_w, target_h))
                scaled.append((cr, label))

            imgs = [c for c, _ in scaled]
            # 动态算行宽：最终图纵横比接近 4:3
            total_area = sum(c.shape[0] * c.shape[1] for c in imgs)
            target_ratio = 4 / 3
            ideal_w = int(math.sqrt(total_area * target_ratio))
            max_width = max(target_h * 2, min(ideal_w, target_h * 12))

            rows, cur_row, cur_w = [], [], 0
            for c, label in scaled:
                if cur_w + c.shape[1] > max_width and cur_row:
                    rows.append(cur_row)
                    cur_row, cur_w = [(c, label)], c.shape[1]
                else:
                    cur_row.append((c, label))
                    cur_w += c.shape[1]
            if cur_row:
                rows.append(cur_row)

            # 逐行填满 + 标签条，贴到 numpy 画布
            GAP = 2  # crop 间间隔像素
            cells = []
            row_heights = []
            label_bar_heights = []
            font_params = []
            for ri, row in enumerate(rows):
                imgs_in_row = [c for c, _ in row]
                row_w = sum(c.shape[1] for c in imgs_in_row)
                s = max_width / row_w
                if len(row) == 1:
                    s = min(s, 2.0)  # 单格行最多放大 2x，避免撑爆
                rh = int(target_h * s)
                row_heights.append(rh)
                widths = [int(c.shape[1] * s) for c in imgs_in_row]
                delta = max_width - sum(widths)
                if delta != 0 and widths:
                    # 逐格分摊，每格最多 +1，不让末格吸收全部误差
                    i = 0
                    while delta > 0 and i < len(widths):
                        widths[i] += 1
                        delta -= 1
                        i += 1
                    while delta < 0 and i < len(widths):
                        widths[i] -= 1
                        delta += 1
                        i += 1
                row_cells = []
                x = 0
                for j, ((c, label), cw) in enumerate(zip(row, widths)):
                    cr = cv2.resize(c, (max(1, cw), max(1, rh)))
                    row_cells.append((x, cr, label, cw))
                    x += cw + GAP
                bar_h = 8
                row_fs = 0.3
                thick = 1
                text_y = 7
                label_bar_heights.append(bar_h)
                font_params.append((row_fs, thick, text_y))
                cells.append(row_cells)

            total_h = sum(row_heights) + sum(label_bar_heights)
            canvas_w = max(
                int(cell_x + cell_w) for ri2, row_cells2 in enumerate(cells)
                for cell_x, _, _, cell_w in row_cells2
            ) if cells else max_width
            stitch = 255 * np.ones((max(1, total_h), canvas_w, 3), dtype=np.uint8)
            y = 0
            for ri, rh in enumerate(row_heights):
                row_cells = cells[ri]
                # 贴 crop 行
                for (cell_x, cell_img, _, _) in row_cells:
                    h, w = cell_img.shape[:2]
                    stitch[y:y + rh, cell_x:cell_x + w] = cell_img
                y += rh
                # 贴标签条
                bar_h = label_bar_heights[ri]
                row_fs, font_thick, text_y = font_params[ri]
                for (cell_x, _, label, cell_w) in row_cells:
                    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, row_fs, font_thick)
                    tx = cell_x + (cell_w - tw) // 2
                    cv2.putText(stitch[y:y + bar_h, :], label,
                                (max(0, tx), text_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), font_thick)
                y += bar_h

        stem = Path(json_file).stem
        stitch_path = os.path.join(stitch_dir, f"{stem}.jpg")
        cv2.imwrite(stitch_path, stitch)

        # 回写 stitch_url 到 JSON
        data["stitch_url"] = _build_stitch_url(task_id, f"{stem}.jpg")
        with open(fp, "w") as f:
            json.dump(data, f, indent=2)

def overlay_scores(task_id: str, file_stem: str, scores: list[dict]):
    """用原图 + bbox + 评分重画检测框，输出最终图到 FINAL_DIR"""
    json_path = os.path.join(OUTPUT_INFO_DIR, task_id, f"{file_stem}.json")
    if not os.path.isfile(json_path):
        return False

    with open(json_path) as f:
        data = json.load(f)

    for s in scores:
        idx = s["index"]
        if idx < len(data["detections"]):
            data["detections"][idx]["score"] = s["score"]

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # 读原图（不是 YOLO 标注图，避免标签重叠）
    orig_img = cv2.imread(data["input_path"])
    if orig_img is None:
        return False

    # 颜色映射：按 class_id 分配不同颜色
    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
        (200, 200, 0), (200, 0, 200), (0, 200, 200),
    ]

    for det in data["detections"]:
        color = COLORS[det["class_id"] % len(COLORS)]
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
        sc = det.get("score")
        label = f"{det['class_name']} {det['confidence']:.2f}"
        if sc is not None:
            label += f" score:{sc}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(orig_img, (x1, y1 - th - 5), (x1 + tw + 4, y1), color, -1)
        cv2.putText(orig_img, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1)

    final_dir = os.path.join(FINAL_DIR, task_id)
    os.makedirs(final_dir, exist_ok=True)
    cv2.imwrite(os.path.join(final_dir, f"{file_stem}.jpg"), orig_img)
    return True

def submit_task(task_id: str, focus_classes: str = "0"):
    asyncio.get_event_loop().run_in_executor(_executor, run_detection, task_id, focus_classes)
