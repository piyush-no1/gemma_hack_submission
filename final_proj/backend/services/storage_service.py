import os
import json
from typing import Any, Dict
from ..config import settings

class StorageService:
    def __init__(self, storage_root: str = settings.STORAGE_ROOT):
        self.root = os.path.abspath(storage_root)
        os.makedirs(self.root, exist_ok=True)

    def get_job_dir(self, job_id: str) -> str:
        job_dir = os.path.join(self.root, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir

    def get_video_path(self, job_id: str) -> str:
        return os.path.join(self.get_job_dir(job_id), "input.mp4")

    def get_frames_dir(self, job_id: str) -> str:
        frames_dir = os.path.join(self.get_job_dir(job_id), "frames")
        os.makedirs(frames_dir, exist_ok=True)
        return frames_dir

    def get_frame_path(self, job_id: str, frame_id: int) -> str:
        return os.path.join(self.get_frames_dir(job_id), f"frame_{frame_id:04d}.jpg")

    def get_attention_dir(self, job_id: str) -> str:
        attention_dir = os.path.join(self.get_job_dir(job_id), "attention_overlays")
        os.makedirs(attention_dir, exist_ok=True)
        return attention_dir

    def get_attention_overlay_path(self, job_id: str, timestamp_sec: float) -> str:
        return os.path.join(self.get_attention_dir(job_id), f"anomaly_attention_{timestamp_sec:.2f}s.png")

    def get_speech_audio_path(self, job_id: str) -> str:
        return os.path.join(self.get_job_dir(job_id), "report_audio.mp3")

    def get_timeline_plot_path(self, job_id: str) -> str:
        return os.path.join(self.get_job_dir(job_id), "timeline.png")

    def get_detected_anomalies_csv_path(self, job_id: str) -> str:
        return os.path.join(self.get_job_dir(job_id), "detected_anomalies.csv")

    def save_json(self, job_id: str, filename: str, data: Any):
        path = os.path.join(self.get_job_dir(job_id), filename)
        with open(path, "w", encoding="utf-8") as f:
            if hasattr(data, "dict"):
                json.dump(data.dict(), f, indent=2)
            elif hasattr(data, "model_dump"):
                json.dump(data.model_dump(), f, indent=2)
            else:
                json.dump(data, f, indent=2)

    def read_json(self, job_id: str, filename: str) -> Any:
        path = os.path.join(self.get_job_dir(job_id), filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

# Singleton Instance
storage_service = StorageService()
