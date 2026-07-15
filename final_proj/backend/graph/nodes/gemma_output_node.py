from ...config import settings
from ...agent.llm_client import LLMClient
from ...services.storage_service import storage_service
from ...rag.episodic_memory import log_episodic_entry
from ..state import PipelineState, FinalReport

async def gemma_output_node(state: PipelineState) -> dict:
    print("[Node: Gemma Output] Formatting final safety report...")
    job_id = state["job_id"]
    output_language = state.get("output_language", "en")
    
    # 1. Compile all gathered logs
    reasoning = state.get("reasoning_result")
    reasoning_desc = reasoning.reasoning if reasoning else "No deep reasoning analysis requested. General safety evaluation."
    incident_type = reasoning.incident if reasoning else "Safety Surveillance Event"
    confidence = reasoning.confidence if reasoning else 0.70
    
    transcript = state.get("transcript", [])
    transcript_text = "\n".join([f"- Time {t.timestamp_sec:.1f}s: {t.description}" for t in transcript])
    
    yolo_dets = state.get("yolo_detections", [])
    yolo_text = ", ".join(list(set([d.label for d in yolo_dets]))) if yolo_dets else "None"
    
    tool_results = state.get("tool_results")
    tool_text = ""
    if tool_results:
        if tool_results.ocr:
            tool_text += f"\n- OCR signage read: {tool_results.ocr.get('text', [])}"
        if tool_results.crowd_analytics:
            ca = tool_results.crowd_analytics
            tool_text += f"\n- Crowd flow: {ca.get('state')} (Count: {ca.get('people_count')}, density: {ca.get('density')})"
        if tool_results.rag:
            tool_text += f"\n- RAG Knowledge retrieved: {tool_results.rag.get('matches', [])}"

    system_prompt = f"""You are a multi-lingual Emergency Report Writer AI.
Translate and compile the safety report strictly in the requested target language: '{output_language}'.
Output a JSON matching the requested schema.
Constraints:
- Keep the `explanation` field extremely concise (2-4 sentences max), clear, and professional, suitable to be read aloud in under 30 seconds.
- The `severity` field must be one of: low, medium, high.
Output matches the schema format."""

    user_prompt = f"""Target Language: {output_language}

Incident Details:
- Type: {incident_type}
- Confidence: {confidence:.2f}
- Detailed Reasoning:\n{reasoning_desc}

Perception evidence:
- Visual transcript descriptions:\n{transcript_text}
- YOLO static labels: {yolo_text}
{f"Auxiliary evidence logs: {tool_text}" if tool_text else ""}

Please format this safety report into the final structure."""

    client = LLMClient(model_name=settings.OLLAMA_MODEL_OUTPUT)
    
    try:
        report = client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            pydantic_schema=FinalReport
        )
    except Exception:
        # Robust fallback
        report = FinalReport(
            language=output_language,
            incident_summary=f"Safety anomaly of type {incident_type}",
            explanation="Unusual activity patterns observed in video sequence. Area security advised to review.",
            evidence_summary=f"Visual descriptions contain: {yolo_text}",
            recommended_action="Dispatch security personnel for standard inspection.",
            severity="medium"
        )
        
    # Save final_report.json
    storage_service.save_json(job_id, "final_report.json", report)
    
    # 2. Log final safety report verdict into long-term episodic memory
    log_episodic_entry(
        job_id=job_id,
        scene_description=report.incident_summary,
        verdict=report.severity,
        confidence_pct=float(confidence * 100.0),
        reasoning=report.explanation
    )
    
    print(f"[Node: Gemma Output] Completed. Saved multilingual report in '{output_language}' (Severity: {report.severity}).")
    return {"final_report": report}
