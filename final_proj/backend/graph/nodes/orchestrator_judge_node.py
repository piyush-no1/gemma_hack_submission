from ...config import settings
from ...agent.llm_client import LLMClient
from ...services.storage_service import storage_service
from ..state import PipelineState, JudgeEvaluation

async def orchestrator_judge_node(state: PipelineState) -> dict:
    print("[Node: Orchestrator Judge] Evaluation phase...")
    job_id = state["job_id"]
    
    # 1. Compile V-JEPA evidence
    vjepa_res = state.get("vjepa_result")
    vjepa_text = "V-JEPA Anomaly Events:\n"
    if vjepa_res and vjepa_res.anomalies:
        for idx, a in enumerate(vjepa_res.anomalies):
            vjepa_text += f"- Anomaly {idx+1}: Time={a.timestamp_sec:.2f}s | Error={a.raw_prediction_error:.3f} | Z-Score={a.z_score:.2f} | Conf={a.confidence_pct:.1f}%\n"
    else:
        vjepa_text += "- No latent anomalies detected.\n"
        
    # 2. Compile visual transcript
    transcript = state.get("transcript", [])
    transcript_text = "Visual Description Transcript:\n"
    if transcript:
        for t in transcript:
            transcript_text += f"- Time {t.timestamp_sec:.2f}s: {t.description}\n"
    else:
        transcript_text += "- Visual transcript unavailable.\n"
        
    # 3. Compile YOLO static detections
    yolo_dets = state.get("yolo_detections", [])
    yolo_text = "YOLO-World Object Detections (Static Vocabulary):\n"
    if yolo_dets:
        # Group by label to keep it concise
        counts = {}
        for d in yolo_dets:
            counts[d.label] = counts.get(d.label, 0) + 1
        for label, count in counts.items():
            yolo_text += f"- Detected '{label}' {count} times across frames.\n"
    else:
        yolo_text += "- No static objects of interest detected.\n"
        
    # 4. Compile Abrupt Motion evidence
    motion_ev = state.get("motion_evidence", [])
    motion_text = "Abrupt Motion Events:\n"
    abrupt_count = 0
    if motion_ev:
        for m in motion_ev:
            if m.is_abrupt:
                abrupt_count += 1
                motion_text += f"- Time {m.timestamp_sec:.2f}s: Motion Score={m.motion_score:.2f} | Z-Score={m.z_score:.2f}\n"
        if abrupt_count == 0:
            motion_text += "- No abrupt motion events detected (only normal background motion).\n"
    else:
        motion_text += "- Motion detection data unavailable.\n"

    # 5. Compile previous loop results (Tools & Reasoning)
    tool_results = state.get("tool_results")
    tool_text = "Previous Iteration Tool Results:\n"
    if tool_results:
        if tool_results.rag:
            tool_text += f"- RAG lookup matches: {tool_results.rag.get('matches', [])}\n"
        if tool_results.dynamic_yolo:
            tool_text += f"- Dynamic YOLO detections: {[d.label for d in tool_results.dynamic_yolo]}\n"
        if tool_results.ocr:
            tool_text += f"- OCR Extracted text: {tool_results.ocr.get('text', [])}\n"
        if tool_results.crowd_analytics:
            ca = tool_results.crowd_analytics
            tool_text += f"- Crowd Analytics: State={ca.get('state')}, Density={ca.get('density')}, Flow={ca.get('flow_direction')}\n"
        if tool_results.attention_rollout:
            tool_text += f"- Attention maps rendered for timestamps: {[r.get('timestamp_sec') for r in tool_results.attention_rollout]}\n"
    else:
        tool_text += "- No tool results available from previous loops.\n"
        
    reasoning_result = state.get("reasoning_result")
    reasoning_text = "Previous Iteration Reasoning Result:\n"
    if reasoning_result:
        reasoning_text += f"- Suspected Incident: {reasoning_result.incident}\n- Confidence: {reasoning_result.confidence:.2f}\n- Analysis: {reasoning_result.reasoning}\n"
    else:
        reasoning_text += "- No reasoning analysis compiled yet.\n"

    # 6. Build prompts
    system_prompt = """You are the Orchestrator/Judge in a public-safety video surveillance platform.
Fuse multi-modal evidence (V-JEPA anomalies, YOLO static objects, visual transcripts, and motion contours) into a unified safety assessment.
Decide:
1. Do you need the Gemma Reasoning agent to think more deeply about the suspected incident (`needs_reasoning` = true)?
2. Do you need any of these tools to retrieve/verify more evidence (`tool_calls` list):
   - "RAG": Search the knowledge database for UCF-Crime categories. Must provide query string in `rag_query`.
   - "DynamicYOLO": Search for specific custom objects (e.g. 'fire extinguisher', 'red bag', 'vehicle'). Must fill `dynamic_yolo_prompts`.
   - "OCR": Read text overlays or signboards inside frames.
   - "CrowdAnalytics": Run dense optical flow analysis on people clusters to detect panic, stampede, or congestions.
   - "AttentionRollout": Generate query rollout heatmaps on V-JEPA frame indices. Must fill `attention_rollout_timestamps` with floats.
3. If you have sufficient information or have run out of tool iterations, return empty tool list and needs_reasoning = false.

You NEVER write the final incident explanation or report. Output strictly conforms to the JSON schema."""

    user_prompt = f"""[JOB ID: {job_id}]
Perception Evidence Fused:
-------------------------
{vjepa_text}
{transcript_text}
{yolo_text}
{motion_text}

Agent Loop State:
-----------------
Iteration: {state.get("tool_iterations", 0)} / {settings.MAX_TOOL_ITERATIONS}
{tool_text}
{reasoning_text}

Estimate your current confidence. Output details for tool triggers or deeper reasoning requests."""

    client = LLMClient(model_name=settings.OLLAMA_MODEL_JUDGE)
    
    try:
        evaluation = client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            pydantic_schema=JudgeEvaluation
        )
    except Exception:
        # Robust fallback
        evaluation = JudgeEvaluation(
            confidence=0.5,
            needs_reasoning=True,
            tool_calls=[],
            dynamic_yolo_prompts=[],
            rag_query=None,
            attention_rollout_timestamps=[]
        )
        
    # Save judge_eval.json to storage
    storage_service.save_json(job_id, "judge_eval.json", evaluation)
    
    print(f"[Node: Orchestrator Judge] Evaluation finished. Tools triggered: {evaluation.tool_calls}. Needs reasoning: {evaluation.needs_reasoning}.")
    return {"judge_eval": evaluation}
