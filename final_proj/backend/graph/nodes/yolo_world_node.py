import cv2
from ...config import settings
from ...services import yolo_static_service as yolo_static_mod
from ...services.storage_service import storage_service
from ..state import PipelineState, YoloDetection

async def yolo_world_node(state: PipelineState) -> dict:
    print("[Node: YOLO-World] Running static public-safety object detection...")
    job_id = state["job_id"]
    upscaled_frames = state.get("upscaled_frames", [])
    
    flat_detections = []
    per_frame_manifest = []
    
    for frame_rec in upscaled_frames:
        # Load frame image from path
        img = cv2.imread(frame_rec.image_path)
        if img is None:
            continue
            
        # Run detection
        dets = yolo_static_mod.yolo_static_service.detect(img)
        
        frame_dets = []
        for d in dets:
            yolo_det = YoloDetection(**d)
            flat_detections.append(yolo_det)
            frame_dets.append(yolo_det.model_dump())
            
        per_frame_manifest.append({
            "frame_id": frame_rec.frame_id,
            "timestamp_sec": frame_rec.timestamp_sec,
            "detections": frame_dets
        })
        
    # Save yolo_detections.json
    storage_service.save_json(job_id, "yolo_detections.json", {
        "model": yolo_static_mod.yolo_static_service.model_name,
        "vocabulary": yolo_static_mod.yolo_static_service.DEFAULT_VOCABULARY,
        "confidence_threshold": settings.YOLO_CONF_THRESHOLD,
        "per_frame": per_frame_manifest
    })
    
    print(f"[Node: YOLO-World] Completed. Found {len(flat_detections)} total public safety objects.")
    return {"yolo_detections": flat_detections}

