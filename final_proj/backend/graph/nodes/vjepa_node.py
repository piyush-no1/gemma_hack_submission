import time
from ...services import vjepa_service as vjepa_mod
from ...services.storage_service import storage_service
from ..state import PipelineState, VJepaResult, VJepaAnomalyEvent

async def run_vjepa_node(state: PipelineState) -> dict:
    print("[Node: V-JEPA] Running latent predictor anomaly scan...")
    job_id = state["job_id"]
    video_path = state["video_path"]
    
    # Track execution time
    t0 = time.perf_counter()
    
    # Run service
    job_dir = storage_service.get_job_dir(job_id)
    res_dict = await vjepa_mod.vjepa_service.analyze(video_path, job_dir)
    
    elapsed = time.perf_counter() - t0
    res_dict["summary"]["runtime_sec"] = float(elapsed)
    
    # Save vjepa_result.json to storage
    storage_service.save_json(job_id, "vjepa_result.json", res_dict)
    
    # Map to PipelineState schema
    anomalies = [
        VJepaAnomalyEvent(**evt) for evt in res_dict["anomalies"]
    ]
    vjepa_result = VJepaResult(
        processing_metadata=res_dict["processing_metadata"],
        anomalies=anomalies,
        summary=res_dict["summary"]
    )
    
    print(f"[Node: V-JEPA] Completed. Detected {len(anomalies)} anomalies in {elapsed:.2f}s.")
    return {"vjepa_result": vjepa_result}
