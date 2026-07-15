from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ---------- Perception-layer records ----------

class VJepaAnomalyEvent(BaseModel):
    event_id: str
    timestamp_sec: float
    raw_prediction_error: float
    smoothed_score: float
    z_score: float
    confidence_pct: float
    attention_overlay_path: Optional[str] = None

class VJepaResult(BaseModel):
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    anomalies: List[VJepaAnomalyEvent] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)

class FrameRecord(BaseModel):
    frame_id: int
    timestamp_sec: float
    image_path: str
    width: int
    height: int

class YoloDetection(BaseModel):
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]

class TranscriptEntry(BaseModel):
    timestamp_sec: float
    description: str

class MotionEvidence(BaseModel):
    timestamp_sec: float
    motion_score: float
    z_score: float
    is_abrupt: bool
    regions: List[List[float]] = Field(default_factory=list)

# ---------- Orchestration-layer records ----------

class JudgeEvaluation(BaseModel):
    confidence: float
    needs_reasoning: bool
    tool_calls: List[str] = Field(default_factory=list)  # subset of ["RAG", "DynamicYOLO", "OCR", "CrowdAnalytics"]
    dynamic_yolo_prompts: List[str] = Field(default_factory=list)
    rag_query: Optional[str] = None
    attention_rollout_timestamps: List[float] = Field(default_factory=list)
    # Final conclusion fields — filled when judge has decided
    is_emergency: bool = False            # True if the scene represents an emergency situation
    verdict: str = ""                     # Human-readable incident type, e.g. "Fire", "Robbery", "Medical Emergency", "Normal"
    verdict_reasoning: str = ""           # Brief explanation of why the judge reached this conclusion

class ToolResults(BaseModel):
    rag: Optional[Dict[str, Any]] = None
    dynamic_yolo: Optional[List[YoloDetection]] = None
    ocr: Optional[Dict[str, Any]] = None
    crowd_analytics: Optional[Dict[str, Any]] = None
    attention_rollout: Optional[List[Dict[str, Any]]] = None

class ReasoningResult(BaseModel):
    incident: str
    confidence: float
    reasoning: str

class FinalReport(BaseModel):
    language: str
    incident_summary: str
    explanation: str
    evidence_summary: str
    recommended_action: str
    severity: str

# ---------- Top-level pipeline state ----------

class PipelineState(TypedDict, total=False):
    job_id: str
    video_path: str
    status: str
    error: Optional[str]

    vjepa_result: VJepaResult
    upscaled_frames: List[FrameRecord]
    extraction_metadata: Dict[str, Any]
    yolo_detections: List[YoloDetection]
    transcript: List[TranscriptEntry]
    motion_evidence: List[MotionEvidence]

    judge_eval: Optional[JudgeEvaluation]
    tool_results: ToolResults
    tool_iterations: int
    reasoning_result: Optional[ReasoningResult]

    final_report: Optional[FinalReport]
    speech_audio_path: Optional[str]
    output_language: str  # default "en", user-selectable
