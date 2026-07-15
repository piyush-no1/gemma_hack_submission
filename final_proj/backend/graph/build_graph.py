from langgraph.graph import StateGraph, START, END
from .state import PipelineState
from .router import judge_router
from .nodes.vjepa_node import run_vjepa_node
from .nodes.frame_extraction_node import extract_and_upscale_frames_node
from .nodes.yolo_world_node import yolo_world_node
from .nodes.gemma_transcription_node import gemma_transcription_node
from .nodes.abrupt_motion_node import abrupt_motion_node
from .nodes.orchestrator_judge_node import orchestrator_judge_node
from .nodes.reasoning_agent_node import reasoning_agent_node
from .nodes.tool_caller_node import tool_caller_node
from .nodes.gemma_output_node import gemma_output_node
from .nodes.speech_node import speech_node

def build_graph():
    graph = StateGraph(PipelineState)
    
    # Register all nodes
    graph.add_node("run_vjepa", run_vjepa_node)
    graph.add_node("extract_and_upscale_frames", extract_and_upscale_frames_node)
    graph.add_node("yolo_world", yolo_world_node)
    graph.add_node("gemma_transcription", gemma_transcription_node)
    graph.add_node("abrupt_motion_detection", abrupt_motion_node)
    
    graph.add_node("orchestrator_judge", orchestrator_judge_node)
    graph.add_node("reasoning_agent", reasoning_agent_node)
    graph.add_node("tool_caller", tool_caller_node)
    graph.add_node("gemma_output_module", gemma_output_node)
    graph.add_node("reasoning_to_speech", speech_node)

    # 1. Parallel start pathways
    graph.add_edge(START, "run_vjepa")
    graph.add_edge(START, "extract_and_upscale_frames")
    graph.add_edge(START, "abrupt_motion_detection")
    
    # 2. Sequential perception splits
    graph.add_edge("extract_and_upscale_frames", "yolo_world")
    graph.add_edge("extract_and_upscale_frames", "gemma_transcription")
    
    # 3. Fan-in into Orchestrator/Judge
    graph.add_edge("run_vjepa", "orchestrator_judge")
    graph.add_edge("yolo_world", "orchestrator_judge")
    graph.add_edge("gemma_transcription", "orchestrator_judge")
    graph.add_edge("abrupt_motion_detection", "orchestrator_judge")
    
    # 4. Conditional Judge outcomes (Router)
    graph.add_conditional_edges("orchestrator_judge", judge_router, {
        "reasoning_agent": "reasoning_agent",
        "tool_caller": "tool_caller",
        "gemma_output_module": "gemma_output_module"
    })
    
    # Loop back to Judge for iterative loops
    graph.add_edge("reasoning_agent", "orchestrator_judge")
    graph.add_edge("tool_caller", "orchestrator_judge")
    
    # 5. Terminal reporting phase
    graph.add_edge("gemma_output_module", "reasoning_to_speech")
    graph.add_edge("reasoning_to_speech", END)
    
    return graph.compile()
