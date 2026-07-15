import time
import asyncio
from typing import Dict, Any, Optional
from ..graph.build_graph import build_graph
from ..api.ws_progress import ws_progress
from ..services.storage_service import storage_service
from ..config import settings

# Global compiled graph compiled once
compiled_graph = build_graph()

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    def create_job(self, job_id: str, output_language: str) -> Dict[str, Any]:
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": time.time(),
            "output_language": output_language,
            "error": None,
            "result": None
        }
        return self.jobs[job_id]

    def start_job(self, job_id: str) -> asyncio.Task:
        task = asyncio.create_task(self.run_pipeline(job_id))
        self.tasks[job_id] = task
        return task

    def cancel_job(self, job_id: str) -> bool:
        task = self.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        # Read from local memory
        job = self.jobs.get(job_id)
        if not job:
            return None
            
        # Add links to generated files if completed
        if job["status"] == "completed":
            job["detected_anomalies_csv"] = f"/api/jobs/{job_id}/artifacts/detected_anomalies.csv"
            job["timeline_plot"] = f"/api/jobs/{job_id}/artifacts/timeline.png"
        return job

    def update_job(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    async def run_pipeline(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return
            
        video_path = storage_service.get_video_path(job_id)
        initial_state = {
            "job_id": job_id,
            "video_path": video_path,
            "status": "running",
            "tool_iterations": 0,
            "output_language": job["output_language"]
        }
        
        self.update_job(job_id, status="running")
        await ws_progress.broadcast(job_id, {
            "job_id": job_id,
            "status": "running",
            "node": "START",
            "timestamp": time.time()
        })
        
        try:
            # Run LangGraph state machine with an explicit recursion limit
            async for event in compiled_graph.astream(
                initial_state,
                config={"recursion_limit": settings.GRAPH_RECURSION_LIMIT},
                stream_mode="updates"
            ):
                # Broadcast each node execution completed update
                for node_name, node_update in event.items():
                    print(f"[Pipeline Progress] Node '{node_name}' finished.")
                    
                    # Convert Pydantic state elements to serializable dicts if needed
                    serializable_update = {}
                    for k, v in node_update.items():
                        if hasattr(v, "dict"):
                            serializable_update[k] = v.dict()
                        elif hasattr(v, "model_dump"):
                            serializable_update[k] = v.model_dump()
                        else:
                            serializable_update[k] = v

                    await ws_progress.broadcast(job_id, {
                        "job_id": job_id,
                        "status": "completed",
                        "node": node_name,
                        "timestamp": time.time(),
                        "update": serializable_update
                    })
                    
            # Set job completed, fetch final state manifest files if written
            final_report = storage_service.read_json(job_id, "final_report.json")
            
            self.update_job(job_id, status="completed", result=final_report)
            await ws_progress.broadcast(job_id, {
                "job_id": job_id,
                "status": "completed",
                "node": "END",
                "timestamp": time.time(),
                "result": final_report
            })
            print(f"[Pipeline Success] Job {job_id} finished successfully.")
            
        except asyncio.CancelledError:
            self.update_job(job_id, status="failed", error="Pipeline execution stopped by user request.")
            await ws_progress.broadcast(job_id, {
                "job_id": job_id,
                "status": "failed",
                "error": "Pipeline execution stopped by user request.",
                "timestamp": time.time()
            })
            print(f"[Pipeline Cancellation] Job {job_id} cancelled.")
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_job(job_id, status="failed", error=str(e))
            await ws_progress.broadcast(job_id, {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            })
            print(f"[Pipeline Failure] Job {job_id} failed: {e}")
        finally:
            self.tasks.pop(job_id, None)

# Singleton Instance
job_manager = JobManager()
