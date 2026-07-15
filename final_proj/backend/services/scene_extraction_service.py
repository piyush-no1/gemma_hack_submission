import cv2
import os
from typing import List, Dict, Any
from scenedetect import detect, ContentDetector

class SceneExtractionService:
    def extract_keyframe_timestamps(self, video_path: str, max_frames: int = 15) -> List[float]:
        """
        Detects scene changes and extracts keyframe timestamps corresponding to scene midpoints.
        Enforces a minimum spacing of 0.333s (3 FPS max) and caps total keyframes to max_frames.
        Falls back to 5 uniform frames if no scene changes were detected.
        """
        try:
            # Run scene detection
            scene_list = detect(video_path, ContentDetector(threshold=8.0, min_scene_len=10))
            
            timestamps = []
            for scene in scene_list:
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                mid_sec = start_sec + (end_sec - start_sec) / 2.0
                timestamps.append(mid_sec)
                
            # Filter timestamps to prevent too closely spaced frames (max 3 FPS)
            min_spacing = 0.333
            filtered_timestamps = []
            for ts in timestamps:
                if not filtered_timestamps or (ts - filtered_timestamps[-1]) >= min_spacing:
                    filtered_timestamps.append(ts)
            timestamps = filtered_timestamps
            
            # Subsample to cap at max_frames
            if len(timestamps) > max_frames:
                step = len(timestamps) / float(max_frames)
                timestamps = [timestamps[int(i * step)] for i in range(max_frames)]
                
            # Fallback if no scenes detected
            if not timestamps:
                cap = cv2.VideoCapture(video_path)
                duration_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                if fps > 0:
                    duration_sec = duration_frames / fps
                    segment = duration_sec / 6.0
                    timestamps = [segment * i for i in range(1, 6)]
                    
            return timestamps
        except Exception as e:
            print(f"[SceneExtraction] Error during scene detection: {e}. Falling back to uniform sampling.")
            # Fallback to uniform sampling
            cap = cv2.VideoCapture(video_path)
            duration_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if fps > 0 and duration_frames > 0:
                duration_sec = duration_frames / fps
                segment = duration_sec / 6.0
                return [segment * i for i in range(1, 6)]
            return [0.0]

# Singleton Instance
scene_extraction_service = SceneExtractionService()
