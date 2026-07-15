from .state import PipelineState
from ..config import settings

def judge_router(state: PipelineState) -> str:
    """
    Routes after Orchestrator/Judge evaluation.
    - If judge is confident enough (>= 0.75), go straight to output.
    - Only invoke reasoning agent if genuinely ambiguous AND not yet done.
    - Only invoke tools if judge requested them AND iterations budget remains.
    - Reasoning runs at most ONCE per pipeline run.
    """
    eval_obj = state.get("judge_eval")
    tool_iterations = state.get("tool_iterations", 0)
    reasoning_done = state.get("reasoning_result") is not None

    # No eval: safe fallback to output
    if not eval_obj:
        return "gemma_output_module"

    # Hard override: if confidence is high enough, skip everything and produce the report
    if eval_obj.confidence >= settings.JUDGE_CONFIDENCE_THRESHOLD:
        return "gemma_output_module"

    # Hard override: exhausted tool budget — go to output
    if tool_iterations >= settings.MAX_TOOL_ITERATIONS:
        return "gemma_output_module"

    # Run reasoning only if judge wants it AND it hasn't been done yet
    if eval_obj.needs_reasoning and not reasoning_done:
        return "reasoning_agent"

    # Run tools only if judge requested them and budget remains
    if eval_obj.tool_calls and tool_iterations < settings.MAX_TOOL_ITERATIONS:
        return "tool_caller"

    # Fallback: enough information gathered, produce the report
    return "gemma_output_module"
