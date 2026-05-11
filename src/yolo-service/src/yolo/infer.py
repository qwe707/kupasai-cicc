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

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# POST /detect/upload — 上传文件提交任务
# ---------------------------------------------------------------------------

@app.post("/detect/upload")
async def detect_upload(
    task_id: str = Query(...),
    focus_classes: str = Query("0"),
    files: list[UploadFile] = File(...),
):
    if not task_id:
        raise HTTPException(400, "task_id is required.")
    if not files:
        raise HTTPException(400, "At least one file is required.")

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

    detector.init_task(task_id, saved_count, focus_classes)
    detector.submit_task(task_id, focus_classes)

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

    detector.init_task(req.task_id, copied, req.focus_classes)
    detector.submit_task(req.task_id, req.focus_classes)

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
    if not task_id and not file:
        tasks = detector.list_tasks()
        return {"tasks": tasks, "count": len(tasks)}

    if task_id and file and type == "info":
        json_name = f"{Path(file).stem}.json"
        json_path = os.path.join(detector.OUTPUT_INFO_DIR, task_id, json_name)
        if not os.path.isfile(json_path):
            raise HTTPException(404, "Result not found.")
        with open(json_path) as f:
            return JSONResponse(content=json.load(f))

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

# ---------------------------------------------------------------------------
# GET /images/{task_id}/{filename} — 查看标注图（直链）
# ---------------------------------------------------------------------------

@app.get("/images/{task_id}/{filename}")
async def get_task_image(task_id: str, filename: str):
    img_path = os.path.join(detector.OUTPUT_IMG_DIR, task_id, filename)
    if not os.path.isfile(img_path):
        raise HTTPException(404, "Image not found.")
    media_type, _ = mimetypes.guess_type(img_path)
    if media_type is None:
        media_type = "application/octet-stream"
    return FileResponse(img_path, media_type=media_type, headers={
        "Content-Disposition": "inline",
        "Cache-Control": "public, max-age=3600",
    })

# ---------------------------------------------------------------------------
# GET /stitches/{task_id}/{filename} — Agent 获取拼接图
# ---------------------------------------------------------------------------

@app.get("/stitches/{task_id}/{filename}")
async def get_stitch(task_id: str, filename: str):
    img_path = os.path.join(detector.STITCH_DIR, task_id, filename)
    if not os.path.isfile(img_path):
        raise HTTPException(404, "Stitch not found.")
    media_type, _ = mimetypes.guess_type(img_path)
    if media_type is None:
        media_type = "application/octet-stream"
    return FileResponse(img_path, media_type=media_type, headers={
        "Content-Disposition": "inline",
        "Cache-Control": "public, max-age=3600",
    })

# ---------------------------------------------------------------------------
# POST /scores — Agent 回传评分，触发叠加
# ---------------------------------------------------------------------------

@app.post("/scores")
async def receive_scores(req: schemas.ScoreRequest):
    if not req.task_id or not req.file:
        raise HTTPException(400, "task_id and file are required.")
    file_stem = Path(req.file).stem
    success = detector.overlay_scores(req.task_id, file_stem,
                                       [{"index": s.index, "score": s.score} for s in req.scores])
    if not success:
        raise HTTPException(404, "Task/Image not found or already processed.")
    final_url = detector._build_public_url(req.task_id, f"{file_stem}.jpg", "final")
    return {"status": "ok", "file": req.file, "scores_applied": len(req.scores), "final_url": final_url}

# ---------------------------------------------------------------------------
# POST /scores/batch — 批量回传评分
# ---------------------------------------------------------------------------

@app.post("/scores/batch")
async def receive_scores_batch(req: schemas.BatchScoreRequest):
    results = []
    for item in req.items:
        file_stem = Path(item.file).stem
        success = detector.overlay_scores(
            req.task_id, file_stem,
            [{"index": s.index, "score": s.score} for s in item.scores]
        )
        final_url = detector._build_public_url(req.task_id, f"{file_stem}.jpg", "final") if success else None
        results.append({
            "file": item.file,
            "status": "ok" if success else "failed",
            "scores_applied": len(item.scores) if success else 0,
            "final_url": final_url,
        })
    return {"task_id": req.task_id, "results": results}

# ---------------------------------------------------------------------------
# GET /final/{task_id}/{filename} — 查看最终图（评分叠加后）
# ---------------------------------------------------------------------------

@app.get("/final/{task_id}/{filename}")
async def get_final_image(task_id: str, filename: str):
    img_path = os.path.join(detector.FINAL_DIR, task_id, filename)
    if not os.path.isfile(img_path):
        raise HTTPException(404, "Final image not found.")
    media_type, _ = mimetypes.guess_type(img_path)
    if media_type is None:
        media_type = "application/octet-stream"
    return FileResponse(img_path, media_type=media_type, headers={
        "Content-Disposition": "inline",
        "Cache-Control": "public, max-age=3600",
    })
