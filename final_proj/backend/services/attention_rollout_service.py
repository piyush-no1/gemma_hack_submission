import os
import cv2
from typing import List, Dict, Any
from .vjepa_service import VJepaAnomalyService
from .storage_service import storage_service

class AttentionRolloutService:
    def __init__(self, vjepa_service: VJepaAnomalyService):
        self.vjepa_service = vjepa_service

    async def generate_rollouts(self, job_id: str, video_path: str, timestamps: List[float]) -> List[Dict[str, Any]]:
        """
        Generates on-demand attention rollout heatmaps for the specified timestamps of interest
        sharing the loaded VJepa service weights.
        """
        results = []
        if not timestamps:
            return results

        # 1. Load video frames using vjepa service logic (stride=2)
        frames, fps = self.vjepa_service._load_video(video_path, stride=2)
        total_frames = len(frames)
        
        if total_frames < 16:
            return results

        for t in timestamps:
            # 2. Find closest window centered around target timestamp t
            # Target frame index in our sampled list
            target_idx = int((t * fps) / 2.0)
            
            # The target frame should ideally align with the start of target clip (index 8 of 16)
            start_idx = target_idx - 8
            
            # Clip bounds
            start_idx = max(0, min(total_frames - 16, start_idx))
            joint_clip = frames[start_idx : start_idx + 16]
            
            # 3. Compute attention rollout using shared V-JEPA model
            rollout = await self.vjepa_service.compute_attention_rollout_for_clip(joint_clip)
            
            # 4. Save overlay frame using storage service structure
            output_path = storage_service.get_attention_overlay_path(job_id, t)
            self.vjepa_service.save_attention_overlay(t, rollout, joint_clip, output_path)
            
            # Store relative or absolute path for state
            results.append({
                "timestamp_sec": float(t),
                "overlay_path": os.path.normpath(output_path)
            })
            
        return results

# Singleton Instance (init at startup)
attention_rollout_service = None
