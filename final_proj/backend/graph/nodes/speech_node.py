import os
from ...services.tts_service import tts_service
from ...services.storage_service import storage_service
from ..state import PipelineState

async def speech_node(state: PipelineState) -> dict:
    print("[Node: Speech TTS] TTS synthesis is disabled. Skipping speech synthesis.")
    return {"speech_audio_path": None}
