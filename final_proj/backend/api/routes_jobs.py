import os
import uuid
import asyncio
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from ..jobs.job_manager import job_manager
from ..services.storage_service import storage_service
from .ws_progress import ws_progress

router = APIRouter()

@router.post("/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    output_language: str = Form("en")
):
    # 1. Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported video format. Upload mp4, avi, mov, or mkv.")

    # 2. Generate job uuid
    job_id = str(uuid.uuid4())
    
    # 3. Save uploaded file to storage path
    video_path = storage_service.get_video_path(job_id)
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # 4. Create and start job background task
    job_manager.create_job(job_id, output_language)
    
    # Kick off graph loop in background task
    asyncio.create_task(job_manager.run_pipeline(job_id))
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Pipeline execution started in the background."
    }

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@router.get("/jobs/{job_id}/artifacts/{filename}")
async def get_artifact(job_id: str, filename: str):
    # Prevent directory traversal attacks
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
        
    job_dir = storage_service.get_job_dir(job_id)
    
    # File could be in job_dir directly, or subdirectories like frames/ or attention_overlays/
    # We will search recursively inside job_dir for filename
    file_path = None
    for root, dirs, files in os.walk(job_dir):
        if filename in files:
            file_path = os.path.join(root, filename)
            break
            
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artifact not found.")
        
    # Set correct media headers
    media_type = "application/octet-stream"
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".mp3"):
        media_type = "audio/mpeg"
    elif filename.endswith(".csv"):
        media_type = "text/csv"
        
    return FileResponse(file_path, media_type=media_type)

@router.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws_progress.connect(job_id, websocket)
    try:
        while True:
            # Keep socket alive and receive client messages if any
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_progress.disconnect(job_id, websocket)
    except Exception as e:
        print(f"[WebSocket] Socket error for job {job_id}: {e}")
        ws_progress.disconnect(job_id, websocket)
