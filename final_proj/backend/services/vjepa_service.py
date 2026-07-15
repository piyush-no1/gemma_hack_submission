import os
import cv2
import math
import torch
import asyncio
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoProcessor, AutoModel
from ..config import settings
from .detection_primitives import (
    RobustStatistics,
    ExponentialSmoother,
    HysteresisDetector,
    ConsecutiveConfirmation
)

class VJepaAnomalyService:
    def __init__(self, model_name: str = settings.VJEPA_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = None
        self.model = None
        # Thread safety lock for model forward passes and captured attention lists
        self.lock = asyncio.Lock()
        self._captured_attentions: list = []
        self._load_model()

    def _load_model(self):
        print(f"[V-JEPA] Loading model '{self.model_name}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        # Load initially with fast SDPA
        self.model = AutoModel.from_pretrained(self.model_name, attn_implementation="sdpa")
        
        # Register attention hook
        for layer in self.model.encoder.layer:
            layer.attention.register_forward_hook(self._attention_hook_fn)
            
        self.model.to(self.device)
        self.model.eval()
        print("[V-JEPA] Model loaded and hook registered successfully.")

    def _attention_hook_fn(self, module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self._captured_attentions.append(output[1].detach())

    def _load_video(self, video_path: str, stride: int = 2) -> Tuple[List[np.ndarray], float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        frames = []
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_index += 1
            
        cap.release()
        return frames, fps

    def _generate_windows(self, frames: List[np.ndarray]) -> Tuple[List[List[np.ndarray]], List[int]]:
        clips = []
        timestamps = []
        total = len(frames)
        if total < 16:
            return clips, timestamps
            
        for start in range(0, total - 16 + 1, 4):
            clip = frames[start : start + 16]
            clips.append(clip)
            timestamps.append(start + 8)
        return clips, timestamps

    async def compute_prediction_error(self, joint_clip: List[np.ndarray]) -> float:
        """Runs fast forward pass using SDPA (under lock)."""
        async with self.lock:
            # Clear hook list in case of residue
            self._captured_attentions.clear()
            self.model.config._attn_implementation = "sdpa"
            
            inputs = self.processor(videos=list(joint_clip), return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to(self.device)
            
            # Predictor context & target masks
            context_mask = [torch.arange(1024, device=self.device).unsqueeze(0)]
            target_mask = [torch.arange(1024, 2048, device=self.device).unsqueeze(0)]
            
            with torch.no_grad():
                outputs = self.model(
                    pixel_values,
                    context_mask=context_mask,
                    target_mask=target_mask
                )
                
            predicted = outputs.predictor_output.last_hidden_state
            actual = outputs.predictor_output.target_hidden_state
            
            predicted_norm = F.normalize(predicted, dim=-1)
            actual_norm = F.normalize(actual, dim=-1)
            
            cosine_sim = (predicted_norm * actual_norm).sum(dim=-1)
            mean_cosine_sim = cosine_sim.mean().item()
            
            return 1.0 - mean_cosine_sim

    async def compute_attention_rollout_for_clip(self, joint_clip: List[np.ndarray]) -> torch.Tensor:
        """Runs forward pass with Eager Attention to capture weights and compute rollout."""
        async with self.lock:
            self._captured_attentions.clear()
            self.model.config._attn_implementation = "eager"
            
            inputs = self.processor(videos=list(joint_clip), return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to(self.device)
            context_mask = [torch.arange(1024, device=self.device).unsqueeze(0)]
            target_mask = [torch.arange(1024, 2048, device=self.device).unsqueeze(0)]
            
            with torch.no_grad():
                _ = self.model(
                    pixel_values,
                    context_mask=context_mask,
                    target_mask=target_mask
                )
                
            attentions = list(self._captured_attentions)
            self._captured_attentions.clear()
            self.model.config._attn_implementation = "sdpa" # restore fast mode
            
            if not attentions:
                # Return identity matrix if no attention maps captured
                return torch.eye(2048, device=self.device)
                
            # Compute rollout recursively: R = (0.5 * A + 0.5 * I) * R
            num_layers = len(attentions)
            num_tokens = attentions[0].shape[-1]
            rollout = torch.eye(num_tokens, device=self.device)
            
            for i in range(num_layers):
                layer_attn = attentions[i][0].mean(dim=0) # average heads
                layer_attn = 0.5 * layer_attn + 0.5 * torch.eye(num_tokens, device=self.device)
                layer_attn = layer_attn / (layer_attn.sum(dim=-1, keepdim=True) + 1e-8)
                rollout = torch.matmul(layer_attn, rollout)
                
            return rollout

    def save_attention_overlay(self, timestamp: float, rollout: torch.Tensor, joint_clip: List[np.ndarray], output_path: str):
        # rollout shape: (2048, 2048)
        importance = rollout.mean(dim=0) # Shape: (2048,)
        importance = importance.view(8, 16, 16)
        
        # Focus on the target segment (steps 4 to 7)
        target_importance = importance[4:].mean(dim=0) # Shape: (16, 16)
        
        # Representative frame (index 12)
        frame_idx = min(12, len(joint_clip) - 1)
        base_img = joint_clip[frame_idx].copy()
        h, w, c = base_img.shape
        
        # Normalize heatmap to [0, 1]
        heatmap_np = target_importance.cpu().numpy()
        heatmap_min = heatmap_np.min()
        heatmap_max = heatmap_np.max()
        heatmap_np = (heatmap_np - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
        
        # Upscale heatmap to match image size
        heatmap_resized = cv2.resize(heatmap_np, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply JET colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Blend: 60% original image, 40% heatmap
        blended = cv2.addWeighted(base_img, 0.6, heatmap_color, 0.4, 0)
        
        # Save frame (OpenCV writes BGR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    async def analyze(self, video_path: str, output_dir: str, on_progress=None) -> Dict[str, Any]:
        """Runs sliding window V-JEPA anomaly scan using fast SDPA and dynamic rollout hooks."""
        frames, fps = self._load_video(video_path, stride=2)
        clips, timestamps = self._generate_windows(frames)
        total_clips = len(clips)
        
        stats = RobustStatistics(baseline_size=50)
        smoother = ExponentialSmoother(alpha=0.3)
        detector = HysteresisDetector(enter_threshold=2.5, exit_threshold=1.2)
        confirmation = ConsecutiveConfirmation(required_confirmations=max(2, int(math.ceil(1.0 / ((4 * 2) / fps)))))
        
        anomalies = []
        raw_scores = []
        smoothed_scores = []
        z_scores = []
        
        for idx, (clip, frame_index) in enumerate(zip(clips, timestamps)):
            prediction_error = await self.compute_prediction_error(clip)
            
            raw_scores.append(prediction_error)
            smoothed = smoother.update(prediction_error)
            smoothed_scores.append(smoothed)
            
            z = stats.compute_zscore(smoothed)
            z_scores.append(z)
            stats.update(smoothed)
            
            anomaly = detector.update(z)
            confirmed = confirmation.update(anomaly)
            
            timestamp_sec = (frame_index * 2) / fps
            confidence = 50.0 + 15.0 * z
            confidence = float(np.clip(confidence, 0.0, 100.0))
            
            event_id = f"vjepa_evt_{len(anomalies) + 1:03d}"
            
            if confirmed:
                # Dynamically calculate and save rollout overlay path
                overlay_filename = f"anomaly_attention_{timestamp_sec:.2f}s.png"
                overlay_path = os.path.join(output_dir, "vjepa_overlays", overlay_filename)
                
                # Dynamic eager attention rollout pass
                rollout = await self.compute_attention_rollout_for_clip(clip)
                self.save_attention_overlay(timestamp_sec, rollout, clip, overlay_path)
                
                anomalies.append({
                    "event_id": event_id,
                    "timestamp_sec": float(timestamp_sec),
                    "raw_prediction_error": float(prediction_error),
                    "smoothed_score": float(smoothed),
                    "z_score": float(z),
                    "confidence_pct": float(confidence),
                    "attention_overlay_path": os.path.normpath(overlay_path)
                })
                
            if on_progress:
                on_progress(idx + 1, total_clips)
                
        # Generate summary
        summary = {
            "total_anomalies_detected": len(anomalies),
            "video_duration_sec": float(len(frames) * 2 / fps),
            "runtime_sec": 0.0 # Will be populated by the calling node
        }
        
        return {
            "processing_metadata": {
                "model_name": self.model_name,
                "device": self.device,
                "fps": float(fps),
                "frame_stride": 2,
                "clip_len": 16,
                "window_step": 4,
                "total_clips_processed": total_clips
            },
            "anomalies": anomalies,
            "summary": summary
        }

# Singleton Instance (configured in lifespan startup)
vjepa_service = None
