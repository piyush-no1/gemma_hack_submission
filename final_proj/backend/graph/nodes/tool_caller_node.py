import cv2
from ...services.storage_service import storage_service
from ...services import yolo_dynamic_service as yolo_dynamic_mod
from ...services import ocr_service as ocr_mod
from ...services.crowd_analysis_service import crowd_analysis_service
from ...services import attention_rollout_service as attention_rollout_mod
from ...rag.retrieval_tool import retrieve_anomaly_knowledge
from ..state import PipelineState, ToolResults, YoloDetection

async def tool_caller_node(state: PipelineState) -> dict:
    print("[Node: Tool Caller] Dispatching tools requested by Orchestrator...")
    job_id = state["job_id"]
    video_path = state["video_path"]
    judge_eval = state["judge_eval"]
    upscaled_frames = state.get("upscaled_frames", [])
    yolo_detections = state.get("yolo_detections", [])
    motion_evidence = state.get("motion_evidence", [])
    
    # Load or initialize tool results
    current_tools = state.get("tool_results")
    if not current_tools:
        current_tools = ToolResults()
        
    if not judge_eval or not judge_eval.tool_calls:
        print("[Node: Tool Caller] No tool calls requested.")
        return {"tool_iterations": state.get("tool_iterations", 0) + 1}

    # 1. RAG Retrieve Anomaly Knowledge
    if "RAG" in judge_eval.tool_calls and judge_eval.rag_query:
        print(f"  [Tool: RAG] Executing hybrid search for query: '{judge_eval.rag_query}'...")
        rag_data = retrieve_anomaly_knowledge(judge_eval.rag_query)
        current_tools.rag = rag_data
        
    # 2. Dynamic YOLO Object Detection
    if "DynamicYOLO" in judge_eval.tool_calls and judge_eval.dynamic_yolo_prompts:
        print(f"  [Tool: DynamicYOLO] Custom vocabulary: {judge_eval.dynamic_yolo_prompts}...")
        dynamic_dets = []
        for frame_rec in upscaled_frames:
            img = cv2.imread(frame_rec.image_path)
            if img is not None:
                dets = await yolo_dynamic_mod.yolo_dynamic_service.detect_with_custom_classes(img, judge_eval.dynamic_yolo_prompts)
                for d in dets:
                    dynamic_dets.append(YoloDetection(**d))
        current_tools.dynamic_yolo = dynamic_dets
        
    # 3. OCR Text Extraction
    if "OCR" in judge_eval.tool_calls:
        print("  [Tool: OCR] Extracting frame text contents...")
        # Subsample frames to run OCR on (max 3 frames to limit latency)
        ocr_texts = []
        ocr_confs = []
        subsampled = upscaled_frames[::max(1, len(upscaled_frames) // 3)]
        for frame_rec in subsampled:
            img = cv2.imread(frame_rec.image_path)
            if img is not None:
                res = await ocr_mod.ocr_service.extract_text(img)
                ocr_texts.extend(res["text"])
                ocr_confs.extend(res["confidence"])
        current_tools.ocr = {
            "text": ocr_texts,
            "confidence": ocr_confs
        }
        
    # 4. Crowd Analytics Optical Flow
    if "CrowdAnalytics" in judge_eval.tool_calls:
        print("  [Tool: CrowdAnalytics] Analyzing density and flow fields...")
        crowd_res = crowd_analysis_service.analyze(upscaled_frames, yolo_detections, motion_evidence)
        current_tools.crowd_analytics = crowd_res
        
    # 5. V-JEPA Attention Rollout
    if "AttentionRollout" in judge_eval.tool_calls and judge_eval.attention_rollout_timestamps:
        print(f"  [Tool: AttentionRollout] Custom rollouts for timestamps: {judge_eval.attention_rollout_timestamps}...")
        rollouts = await attention_rollout_mod.attention_rollout_service.generate_rollouts(
            job_id, video_path, judge_eval.attention_rollout_timestamps
        )
        current_tools.attention_rollout = rollouts

    # Save tool_results.json
    storage_service.save_json(job_id, "tool_results.json", current_tools)
    
    next_iterations = state.get("tool_iterations", 0) + 1
    print(f"[Node: Tool Caller] Completed. Tool Iteration: {next_iterations}.")
    return {
        "tool_results": current_tools,
        "tool_iterations": next_iterations
    }
