import cv2
import time
from ...services.scene_extraction_service import scene_extraction_service
from ...services.upscaler_service import upscaler_service
from ...services.storage_service import storage_service
from ..state import PipelineState, FrameRecord

async def extract_and_upscale_frames_node(state: PipelineState) -> dict:
    print("[Node: Frame Extraction] Sampling and upscaling video frames...")
    job_id = state["job_id"]
    video_path = state["video_path"]
    
    t0 = time.perf_counter()
    
    # 1. Extract scene cut timestamps
    timestamps = scene_extraction_service.extract_keyframe_timestamps(video_path, max_frames=15)
    
    # 2. Extract and upscale frames using cv2 and Real-ESRGAN
    cap = cv2.VideoCapture(video_path)
    upscaled_frames = []
    
    for idx, ts in enumerate(timestamps):
        frame_id = idx + 1
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if ret:
            # Downscale width if wider than 768px for upscaler speed/safety
            h, w = frame.shape[:2]
            if w > 768:
                ratio = 768.0 / w
                frame = cv2.resize(frame, (768, int(h * ratio)), interpolation=cv2.INTER_AREA)
                
            # Upscale BGR image frame
            upscaled = upscaler_service.upscale(frame)
            h_up, w_up = upscaled.shape[:2]
            
            # Save upscaled frame image
            output_path = storage_service.get_frame_path(job_id, frame_id)
            cv2.imwrite(output_path, upscaled)
            
            upscaled_frames.append(FrameRecord(
                frame_id=frame_id,
                timestamp_sec=float(ts),
                image_path=output_path,
                width=w_up,
                height=h_up
            ))
            
    cap.release()
    elapsed = time.perf_counter() - t0
    
    # 3. Create manifest and save JSON
    metadata = {
        "mode": "scene_detection",
        "total_frames_extracted": len(upscaled_frames),
        "upscale_model": settings.UPSCALER_MODEL_PATH,
        "upscale_factor": 4,
        "runtime_sec": elapsed
    }
    
    storage_service.save_json(job_id, "extraction_metadata.json", {
        **metadata,
        "frames": [f.model_dump() for f in upscaled_frames]
    })
    
    print(f"[Node: Frame Extraction] Completed in {elapsed:.2f}s. Extracted {len(upscaled_frames)} upscaled frames.")
    return {
        "upscaled_frames": upscaled_frames,
        "extraction_metadata": metadata
    }

from ...config import settings
