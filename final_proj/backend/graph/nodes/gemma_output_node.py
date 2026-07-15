from ...config import settings
from ...agent.llm_client import LLMClient
from ...services.storage_service import storage_service
from ...rag.episodic_memory import log_episodic_entry
from ..state import PipelineState, FinalReport

async def gemma_output_node(state: PipelineState) -> dict:
    """
    Final report generation layer.
    Uses gemma4:2b to convert the orchestrator judge's structured verdict
    into a clear, human-readable emergency report for the frontend.
    The judge (gemma4:4b) has already done all the research and declared
    its verdict — this node simply formats it into the final report.
    """
    print("[Node: Gemma Output] Generating final emergency report (gemma4:2b)...")
    job_id = state["job_id"]
    output_language = state.get("output_language", "en")

    # --- Primary source: orchestrator judge's explicit verdict ---
    judge_eval = state.get("judge_eval")
    judge_verdict = judge_eval.verdict if judge_eval and judge_eval.verdict else "Unknown Incident"
    judge_is_emergency = judge_eval.is_emergency if judge_eval else False
    judge_reasoning = judge_eval.verdict_reasoning if judge_eval and judge_eval.verdict_reasoning else ""
    judge_confidence = judge_eval.confidence if judge_eval else 0.5

    # --- Secondary source: deep reasoning agent (if it ran) ---
    reasoning = state.get("reasoning_result")
    deep_reasoning_text = reasoning.reasoning if reasoning else ""
    # If deep reasoning was done and is more specific, prefer it for the verdict
    if reasoning and reasoning.confidence > judge_confidence:
        judge_verdict = reasoning.incident
        deep_reasoning_text = reasoning.reasoning

    # --- Supporting evidence summary ---
    transcript = state.get("transcript", [])
    transcript_text = "\n".join([f"- Time {t.timestamp_sec:.1f}s: {t.description}" for t in transcript])

    yolo_dets = state.get("yolo_detections", [])
    yolo_labels = ", ".join(list(set([d.label for d in yolo_dets]))) if yolo_dets else "None detected"

    motion_ev = state.get("motion_evidence", [])
    abrupt_count = sum(1 for m in motion_ev if m.is_abrupt)

    tool_results = state.get("tool_results")
    auxiliary_text = ""
    if tool_results:
        if tool_results.ocr and tool_results.ocr.get("text"):
            auxiliary_text += f"\n- OCR text found: {tool_results.ocr['text']}"
        if tool_results.crowd_analytics:
            ca = tool_results.crowd_analytics
            auxiliary_text += f"\n- Crowd state: {ca.get('state')} (count: {ca.get('people_count')}, density: {ca.get('density')})"
        if tool_results.dynamic_yolo:
            labels = [d.label for d in tool_results.dynamic_yolo]
            auxiliary_text += f"\n- Dynamic scan objects: {labels}"

    # --- Determine severity from is_emergency + confidence ---
    if judge_is_emergency and judge_confidence >= 0.75:
        severity_hint = "high"
    elif judge_is_emergency or judge_confidence >= 0.5:
        severity_hint = "medium"
    else:
        severity_hint = "low"

    # --- Build prompt for gemma4:2b final report writer ---
    system_prompt = f"""You are a concise Emergency Report Writer AI for a surveillance platform.
You receive a structured verdict from the Orchestrator Agent and produce a final human-readable report.
Target language: '{output_language}'.

Output a JSON object with exactly these fields:
- language: the target language code (e.g. "en", "hi", "es")
- incident_summary: One sentence. State the incident type and location context.
- explanation: 2-4 sentences. Describe what happened, key visual evidence, timeline of events.
- evidence_summary: One sentence. List the key signals that confirmed this verdict.
- recommended_action: One sentence. What should security/emergency services do right now.
- severity: exactly one of: "low", "medium", "high"

Be precise, clinical, and professional. Do not speculate beyond the evidence provided."""

    user_prompt = f"""Orchestrator Verdict:
- Incident Type: {judge_verdict}
- Emergency: {"YES — IMMEDIATE RESPONSE REQUIRED" if judge_is_emergency else "NO — Standard monitoring"}
- Confidence: {judge_confidence:.0%}
- Judge Reasoning: {judge_reasoning}
{f"- Deep Reasoning Analysis: {deep_reasoning_text}" if deep_reasoning_text else ""}

Supporting Evidence:
- Visual Transcript:
{transcript_text if transcript_text else "  No visual transcript available."}
- YOLO-Detected Objects: {yolo_labels}
- Abrupt Motion Events: {abrupt_count} detected{auxiliary_text}
- Suggested Severity: {severity_hint}

Generate the final emergency report in language '{output_language}'."""

    client = LLMClient(model_name=settings.OLLAMA_MODEL_OUTPUT)

    try:
        report = client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            pydantic_schema=FinalReport
        )
    except Exception:
        # Robust fallback — construct a basic report from available data
        is_normal = "normal" in judge_verdict.lower() or "no incident" in judge_verdict.lower()
        report = FinalReport(
            language=output_language,
            incident_summary=f"{'EMERGENCY: ' if judge_is_emergency else ''}{judge_verdict} detected by surveillance system.",
            explanation=judge_reasoning or "Unusual activity patterns observed. Security review recommended.",
            evidence_summary=f"Detected objects: {yolo_labels}. Abrupt motion events: {abrupt_count}.",
            recommended_action="Contact security personnel for immediate inspection." if judge_is_emergency else "Log incident and continue monitoring.",
            severity=severity_hint
        )

    # Save final_report.json to storage
    storage_service.save_json(job_id, "final_report.json", report)

    # Log to episodic memory for future reference
    log_episodic_entry(
        job_id=job_id,
        scene_description=report.incident_summary,
        verdict=report.severity,
        confidence_pct=float(judge_confidence * 100.0),
        reasoning=report.explanation
    )

    print(f"[Node: Gemma Output] Report complete. Verdict: '{judge_verdict}' | Emergency: {judge_is_emergency} | Severity: {report.severity}")
    return {"final_report": report}
