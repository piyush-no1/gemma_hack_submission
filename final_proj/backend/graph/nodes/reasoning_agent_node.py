from ...config import settings
from ...agent.llm_client import LLMClient
from ...services.storage_service import storage_service
from ..state import PipelineState, ReasoningResult

async def reasoning_agent_node(state: PipelineState) -> dict:
    print("[Node: Gemma Reasoning] Invoking 26B deep reasoning agent...")
    job_id = state["job_id"]
    
    # 1. Compile all perception details for reasoning context
    transcript = state.get("transcript", [])
    transcript_text = "\n".join([f"{t.timestamp_sec:.1f}s: {t.description}" for t in transcript])
    
    yolo_dets = state.get("yolo_detections", [])
    yolo_text = ", ".join(list(set([d.label for d in yolo_dets]))) if yolo_dets else "None"
    
    motion_ev = state.get("motion_evidence", [])
    motion_text = f"{len([m for m in motion_ev if m.is_abrupt])} abrupt motion events logged"
    
    vjepa_res = state.get("vjepa_result")
    vjepa_len = len(vjepa_res.anomalies) if vjepa_res and vjepa_res.anomalies else 0
    
    # Tool outcomes
    tool_results = state.get("tool_results")
    tool_logs = ""
    if tool_results:
        if tool_results.rag:
            tool_logs += f"\n- RAG taxonomy matches: {tool_results.rag.get('matches', [])}"
        if tool_results.dynamic_yolo:
            tool_logs += f"\n- Dynamic YOLO detections: {[d.label for d in tool_results.dynamic_yolo]}"
        if tool_results.ocr:
            tool_logs += f"\n- OCR Extracted text: {tool_results.ocr.get('text', [])}"
        if tool_results.crowd_analytics:
            ca = tool_results.crowd_analytics
            tool_logs += f"\n- Crowd state: {ca.get('state')} (Count: {ca.get('people_count')}, Density: {ca.get('density')})"
            
    system_prompt = """You are a senior Emergency Reasoning Agent in a visual safety surveillance platform.
Analyze the multi-modal evidence and tool results to formulate a detailed reasoning report.
Identify:
1. The type of suspected safety incident (e.g. Altercation, Robbery, Fire, Medical Emergency, Normal).
2. A confidence score between 0.0 and 1.0.
3. A clinical, objective explanation detailing the sequence of actions, physical mechanics, and visual indicators.
Do not speculate or include emotional statements. Return JSON corresponding to the schema."""

    user_prompt = f"""[JOB ID: {job_id}]
Perception Context:
- Visual Transcript:\n{transcript_text}
- YOLO Static Objects: {yolo_text}
- Motion Activity: {motion_text}
- V-JEPA Anomaly events count: {vjepa_len}
{f"Gathered Evidence: {tool_logs}" if tool_logs else ""}

Please perform a forensic assessment and output your verdict."""

    client = LLMClient(model_name=settings.OLLAMA_MODEL_REASONING)
    
    try:
        result = client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            pydantic_schema=ReasoningResult
        )
    except Exception:
        result = ReasoningResult(
            incident="Unknown Anomaly",
            confidence=0.5,
            reasoning="Reasoning failed due to inference limits. Defaulting to general anomaly."
        )
        
    # Save reasoning_result.json to storage
    storage_service.save_json(job_id, "reasoning_result.json", result)
    
    print(f"[Node: Gemma Reasoning] Finished. Incident: '{result.incident}' with confidence {result.confidence:.2f}")
    return {"reasoning_result": result}
