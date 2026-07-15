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

    import asyncio

    # Declare async tasks for parallel execution
    async def task_rag():
        if "RAG" in judge_eval.tool_calls and judge_eval.rag_query:
            try:
                print(f"  [Tool: RAG] Executing hybrid search in background thread...")
                rag_data = await asyncio.to_thread(retrieve_anomaly_knowledge, judge_eval.rag_query)
                current_tools.rag = rag_data
            except Exception as e:
                print(f"  [Tool: RAG] Tool execution failed: {e}. Continuing pipeline execution.")
                current_tools.rag = {"matches": [], "error": str(e)}

    async def task_dynamic_yolo():
        if "DynamicYOLO" in judge_eval.tool_calls and judge_eval.dynamic_yolo_prompts:
            try:
                print(f"  [Tool: DynamicYOLO] Custom vocabulary: {judge_eval.dynamic_yolo_prompts}...")
                dynamic_dets = []
                
                async def read_and_detect(frame_rec):
                    img = await asyncio.to_thread(cv2.imread, frame_rec.image_path)
                    if img is None:
                        return []
                    return await yolo_dynamic_mod.yolo_dynamic_service.detect_with_custom_classes(img, judge_eval.dynamic_yolo_prompts)
                
                # Execute frame readings and detections concurrently
                frame_tasks = [read_and_detect(f) for f in upscaled_frames]
                results = await asyncio.gather(*frame_tasks)
                for dets in results:
                    for d in dets:
                        if isinstance(d, YoloDetection):
                            dynamic_dets.append(d)
                        else:
                            dynamic_dets.append(YoloDetection(**d))
                current_tools.dynamic_yolo = dynamic_dets
            except Exception as e:
                print(f"  [Tool: DynamicYOLO] Tool execution failed: {e}. Continuing pipeline execution.")
                current_tools.dynamic_yolo = []

    async def task_ocr():
        if "OCR" in judge_eval.tool_calls:
            try:
                print("  [Tool: OCR] Extracting frame text contents...")
                ocr_texts = []
                ocr_confs = []
                subsampled = upscaled_frames[::max(1, len(upscaled_frames) // 3)]
                
                async def read_and_ocr(frame_rec):
                    img = await asyncio.to_thread(cv2.imread, frame_rec.image_path)
                    if img is not None:
                        return await ocr_mod.ocr_service.extract_text(img)
                    return {"text": [], "confidence": []}
                
                # Execute frame readings and OCR extractions concurrently
                frame_tasks = [read_and_ocr(f) for f in subsampled]
                results = await asyncio.gather(*frame_tasks)
                for res in results:
                    ocr_texts.extend(res["text"])
                    ocr_confs.extend(res["confidence"])
                current_tools.ocr = {
                    "text": ocr_texts,
                    "confidence": ocr_confs
                }
            except Exception as e:
                print(f"  [Tool: OCR] Tool execution failed: {e}. Continuing pipeline execution.")
                current_tools.ocr = {"text": [], "confidence": [], "error": str(e)}

    async def task_crowd():
        if "CrowdAnalytics" in judge_eval.tool_calls:
            try:
                print("  [Tool: CrowdAnalytics] Analyzing density and flow fields in background thread...")
                crowd_res = await asyncio.to_thread(
                    crowd_analysis_service.analyze,
                    upscaled_frames,
                    yolo_detections,
                    motion_evidence
                )
                current_tools.crowd_analytics = crowd_res
            except Exception as e:
                print(f"  [Tool: CrowdAnalytics] Tool execution failed: {e}. Continuing pipeline execution.")
                current_tools.crowd_analytics = {
                    "state": "unknown",
                    "people_count": 0.0,
                    "density": "unknown",
                    "flow_direction": "unknown",
                    "error": str(e)
                }

    # Execute all tools concurrently
    await asyncio.gather(
        task_rag(),
        task_dynamic_yolo(),
        task_ocr(),
        task_crowd()
    )

    # Save tool_results.json
    storage_service.save_json(job_id, "tool_results.json", current_tools)
    
    next_iterations = state.get("tool_iterations", 0) + 1
    print(f"[Node: Tool Caller] Completed. Tool Iteration: {next_iterations}.")
    return {
        "tool_results": current_tools,
        "tool_iterations": next_iterations
    }
