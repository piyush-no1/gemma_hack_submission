from ...services.motion_detection_service import motion_detection_service
from ...services.storage_service import storage_service
from ..state import PipelineState, MotionEvidence

async def abrupt_motion_node(state: PipelineState) -> dict:
    print("[Node: Abrupt Motion] Analyzing frame differences for rapid motion...")
    job_id = state["job_id"]
    video_path = state["video_path"]
    
    # Run downsampled grayscale absdiff contours with MAD z-score gating
    raw_evidences = motion_detection_service.detect_abrupt_motion(video_path)
    
    motion_evidence = [
        MotionEvidence(**evt) for evt in raw_evidences
    ]
    
    # Save motion_evidence.json
    storage_service.save_json(job_id, "motion_evidence.json", {
        "method": "frame_diff_contours_with_zscore_gating",
        "events": [evt.model_dump() for evt in motion_evidence]
    })
    
    print(f"[Node: Abrupt Motion] Completed. Logged {len(motion_evidence)} motion logs.")
    return {"motion_evidence": motion_evidence}
