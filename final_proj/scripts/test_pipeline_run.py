import os
import sys
import asyncio
import shutil
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import settings
import backend.services.vjepa_service as vjepa_mod
import backend.services.yolo_static_service as yolo_static_mod
import backend.services.yolo_dynamic_service as yolo_dynamic_mod
import backend.services.ocr_service as ocr_mod
import backend.services.gemma_vision_service as gemma_vision_mod
import backend.services.attention_rollout_service as attention_rollout_mod

from backend.services.vjepa_service import VJepaAnomalyService
from backend.services.yolo_static_service import YoloStaticDetector
from backend.services.yolo_dynamic_service import YoloDynamicDetector
from backend.services.ocr_service import OcrService
from backend.services.gemma_vision_service import GemmaVisionService
from backend.services.upscaler_service import upscaler_service
from backend.services.attention_rollout_service import AttentionRolloutService
from backend.services.storage_service import storage_service
from backend.graph.build_graph import build_graph

async def test_run():
    print("======================================================================")
    print("AI Surveillance Safety Platform - End-to-End Local Test Execution...")
    print("======================================================================")
    
    # 1. Initialize Singletons locally (mimics FastAPI startup)
    vjepa_mod.vjepa_service = VJepaAnomalyService()
    yolo_static_mod.yolo_static_service = YoloStaticDetector()
    yolo_dynamic_mod.yolo_dynamic_service = YoloDynamicDetector()
    ocr_mod.ocr_service = OcrService()
    gemma_vision_mod.gemma_vision_service = GemmaVisionService()
    attention_rollout_mod.attention_rollout_service = AttentionRolloutService(vjepa_mod.vjepa_service)
    
    # Warmup upscaler
    upscaler_service.warmup()

    # 2. Setup job parameters
    job_id = "test-job-e2e"
    video_source = "../blast.mp4"
    if not os.path.exists(video_source):
        video_source = "blast.mp4"
        
    if not os.path.exists(video_source):
        print(f"[Error] Source video 'blast.mp4' not found at {video_source}.")
        sys.exit(1)
        
    dest_video = storage_service.get_video_path(job_id)
    shutil.copyfile(video_source, dest_video)
    print(f"[Test] Copied {video_source} to {dest_video}")
    
    # Compile graph
    graph = build_graph()
    initial_state = {
        "job_id": job_id,
        "video_path": dest_video,
        "status": "running",
        "tool_iterations": 0,
        "output_language": "en"
    }

    # 3. Execute graph run
    t0 = time.time()
    print("[Test] Launching graph.ainvoke()...")
    final_state = await graph.ainvoke(initial_state, config={"recursion_limit": settings.GRAPH_RECURSION_LIMIT})
    elapsed = time.time() - t0
    
    print("======================================================================")
    print(f"Graph execution finished in {elapsed:.2f} seconds.")
    print("======================================================================")
    
    # 4. Check results
    report = final_state.get("final_report")
    if report:
        print(f"VERDICT SEVERITY: {report.severity.upper()}")
        print(f"SUMMARY: {report.incident_summary}")
        print(f"EXPLANATION: {report.explanation}")
        print(f"RECOMMENDATIONS: {report.recommended_action}")
    else:
        print("[Error] Final report was not generated in the state.")
        sys.exit(1)
        
    audio_path = final_state.get("speech_audio_path")
    if audio_path:
        print(f"Synthesized Speech Audio path: {audio_path}")
        full_audio_path = os.path.join(storage_service.root, audio_path)
        if os.path.exists(full_audio_path):
            print("  - Audio file exists on disk.")
        else:
            print("  - [Error] Audio file path matches but file not found on disk.")
            
    # Check overlays
    attention_dir = storage_service.get_attention_dir(job_id)
    if os.path.exists(attention_dir):
        files = os.listdir(attention_dir)
        print(f"Generated Attention heatmaps: {len(files)} files saved.")
        for f in files[:3]:
            print(f"  - Overlay: {f}")

    print("\n[Test] Success! Pipeline run fully validated.")

if __name__ == "__main__":
    asyncio.run(test_run())
