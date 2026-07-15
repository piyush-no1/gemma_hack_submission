from .state import PipelineState
from ..config import settings

def judge_router(state: PipelineState) -> str:
    """
    Directs graph routing after Orchestrator/Judge analysis.
    Prevents loops by checking iteration counters and reasoning completion flags.
    """
    eval_obj = state.get("judge_eval")
    tool_iterations = state.get("tool_iterations", 0)
    reasoning_done = state.get("reasoning_result") is not None

    if not eval_obj:
        return "gemma_output_module"  # safe fallback

    if eval_obj.needs_reasoning and not reasoning_done:
        return "reasoning_agent"

    if eval_obj.tool_calls and tool_iterations < settings.MAX_TOOL_ITERATIONS:
        return "tool_caller"

    return "gemma_output_module"
