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
Your job is to decide — as efficiently as possible — whether you have SUFFICIENT evidence to call the incident.

DECISION RULES (follow in order):
1. If the multi-modal evidence already makes the situation CLEAR (confidence >= 0.75), set:
   - needs_reasoning = false
   - tool_calls = []
   → The pipeline will go directly to the final report. Do NOT call tools or reasoning unnecessarily.

2. Only set needs_reasoning = true if:
   - The incident type is genuinely ambiguous despite all evidence, AND
   - reasoning has NOT been done yet (see Agent Loop State below), AND
   - You believe 26B reasoning will meaningfully change the conclusion.

3. Only add tool_calls if a specific tool would reveal NEW evidence you cannot infer from existing data:
   - "RAG": Only if you need to classify a rare/unusual event category.
   - "DynamicYOLO": Only if there's a specific object (e.g. weapon, fire) you strongly suspect but YOLO hasn't confirmed.
   - "OCR": Only if visible text in frames is critical to the incident (e.g. warning signs, license plates).
   - "CrowdAnalytics": Only if crowd panic/density is the key ambiguity.

4. If you have already used tools in previous iterations, DO NOT repeat the same tools again.

5. If tool_iterations >= MAX_TOOL_ITERATIONS, set needs_reasoning = false and tool_calls = [] immediately.

VERDICT FIELDS (ALWAYS fill these, even when requesting more tools or reasoning):
- verdict: The incident type you currently believe is most likely. Choose from:
  "Fire", "Explosion", "Robbery", "Assault/Fighting", "Medical Emergency",
  "Shoplifting", "Vandalism", "Crowd Panic/Stampede", "Suspicious Activity",
  "Road Accident", "Normal/No Incident". Use "Unknown" only if you have zero evidence.
- is_emergency: true if this requires immediate response (fire, explosion, assault, medical, robbery, stampede).
- verdict_reasoning: 1-2 sentences explaining the key visual evidence behind your verdict.

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
Tool Iteration: {state.get("tool_iterations", 0)} / {settings.MAX_TOOL_ITERATIONS}
Reasoning already done: {"YES — do NOT request reasoning again" if reasoning_result else "NO"}
{tool_text}
{reasoning_text}

Assess your confidence NOW based on the evidence above.
- If the incident is clear, output confidence >= 0.75 and set needs_reasoning=false, tool_calls=[].
- If genuinely unclear, request only the MINIMUM additional information needed.
- ALWAYS fill verdict, is_emergency, and verdict_reasoning based on your current best assessment.
Output your decision."""

    client = LLMClient(model_name=settings.OLLAMA_MODEL_JUDGE)
    
    try:
        evaluation = client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            pydantic_schema=JudgeEvaluation
        )
    except Exception as err:
        print(f"[Node: Orchestrator Judge] LLM Call failed: {err}. Building dynamic safe fallback...")
        vjepa = state.get("vjepa_result")
        vjepa_anomalies = len(vjepa.anomalies) if vjepa and vjepa.anomalies else 0
        motion = state.get("motion_evidence", [])
        abrupt_motion = any(m.is_abrupt for m in motion)
        
        # Determine if this should be flagged as an emergency based on perception signals
        is_emergency = (vjepa_anomalies > 0 or abrupt_motion)
        verdict = "Potential Safety/Anomaly Event" if is_emergency else "No Incident"
        verdict_reasoning = (
            f"Local LLM fallback. The system detected {vjepa_anomalies} visual anomaly zones "
            f"and abrupt motion signatures: {abrupt_motion}."
        ) if is_emergency else "No significant anomalies or rapid movement signatures detected."
        
        evaluation = JudgeEvaluation(
            confidence=0.7,
            needs_reasoning=True,
            tool_calls=[],
            dynamic_yolo_prompts=[],
            rag_query=None,
            attention_rollout_timestamps=[],
            is_emergency=is_emergency,
            verdict=verdict,
            verdict_reasoning=verdict_reasoning
        )
        
    # Save judge_eval.json to storage
    storage_service.save_json(job_id, "judge_eval.json", evaluation)
    
    print(f"[Node: Orchestrator Judge] Evaluation finished. Tools triggered: {evaluation.tool_calls}. Needs reasoning: {evaluation.needs_reasoning}.")
    return {"judge_eval": evaluation}
