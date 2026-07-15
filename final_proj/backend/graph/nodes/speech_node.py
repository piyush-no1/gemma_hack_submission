from ..state import PipelineState

async def speech_node(state: PipelineState) -> dict:
    print("[Node: Speech TTS] TTS synthesis is disabled. Skipping.")
    return {"speech_audio_path": None}
