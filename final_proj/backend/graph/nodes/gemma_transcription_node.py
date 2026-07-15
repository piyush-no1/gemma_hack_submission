from ...services import gemma_vision_service as gemma_vision_mod
from ...services.storage_service import storage_service
from ..state import PipelineState, TranscriptEntry

async def gemma_transcription_node(state: PipelineState) -> dict:
    print("[Node: Gemma Transcription] Generating frame descriptions...")
    job_id = state["job_id"]
    upscaled_frames = state.get("upscaled_frames", [])
    
    entries = []
    for frame_rec in upscaled_frames:
        # Load frame bytes
        if not os.path.exists(frame_rec.image_path):
            continue
            
        with open(frame_rec.image_path, "rb") as f:
            image_bytes = f.read()
            
        try:
            description = await gemma_vision_mod.gemma_vision_service.transcribe_frame(image_bytes)
            entries.append(TranscriptEntry(
                timestamp_sec=frame_rec.timestamp_sec,
                description=description
            ))
            print(f"  [Frame {frame_rec.frame_id} @ {frame_rec.timestamp_sec:.2f}s]: {description[:60]}...")
        except Exception as e:
            print(f"  [Frame {frame_rec.frame_id}] Transcription failed: {e}. Gracefully skipping.")
            # Graceful degradation (non-blocking)
            entries.append(TranscriptEntry(
                timestamp_sec=frame_rec.timestamp_sec,
                description="[Visual transcription unavailable]"
            ))
            
    # Save transcript.json
    storage_service.save_json(job_id, "transcript.json", {
        "model": settings.OLLAMA_VISION_MODEL if settings.GEMMA_VISION_PROVIDER == "ollama" else settings.GEMINI_VISION_MODEL,
        "provider": settings.GEMMA_VISION_PROVIDER,
        "entries": [e.model_dump() for e in entries]
    })
    
    print(f"[Node: Gemma Transcription] Completed. Generated {len(entries)} transcript entries.")
    return {"transcript": entries}

import os
from ...config import settings
