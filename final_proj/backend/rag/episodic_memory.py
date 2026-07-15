import lancedb
from ..config import settings
from .embed_store import embed

def log_episodic_entry(job_id: str, scene_description: str, verdict: str, confidence_pct: float, reasoning: str):
    """
    Appends a new safety report verdict to the LanceDB knowledge_nodes table under collection='episodic'.
    This allows the system to recall past decisions during surveillance loops.
    """
    text = (
        f"Camera {job_id}: {scene_description}. "
        f"Verdict: {verdict}. "
        f"Confidence: {confidence_pct}%. "
        f"Reasoning: {reasoning}"
    )
    
    try:
        db = lancedb.connect(settings.VECTORSTORE_ROOT)
        table_name = "knowledge_nodes"
        
        if table_name not in db.table_names():
            print(f"[EpisodicMemory] Warning: Table '{table_name}' does not exist yet. Cannot log episodic memory.")
            return
            
        table = db.open_table(table_name)
        vector = embed(text)
        
        table.add([{
            "node_id": f"episodic::{job_id}::{verdict.replace(' ', '_')}",
            "type": "episodic",
            "label": verdict,
            "text": text,
            "vector": vector,
            "collection": "episodic"
        }])
        print(f"[EpisodicMemory] Logged safety verdict for job {job_id} in LanceDB.")
    except Exception as e:
        print(f"[EpisodicMemory] Error logging episodic entry: {e}")
