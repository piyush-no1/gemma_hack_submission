import cv2
import numpy as np
from typing import List, Dict, Any
from .detection_primitives import RobustStatistics, HysteresisDetector

class MotionDetectionService:
    def __init__(self, sample_rate_sec: float = 0.5):
        self.sample_rate_sec = sample_rate_sec

    def detect_abrupt_motion(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Scans the video directly by sampling frames at a low resolution.
        Computes motion score as area of contours of frame absolute differences,
        and applies RobustStatistics + HysteresisDetector to calculate rolling z-scores
        and flag abrupt movements dynamically.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
            
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = frame_count / fps
        
        # Initialize primitives
        stats = RobustStatistics(baseline_size=60)
        detector = HysteresisDetector(enter_threshold=2.5, exit_threshold=1.2)
        
        evidence_list = []
        prev_gray = None
        current_sec = 0.0
        
        while current_sec < duration_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess: convert to gray and downsample for performance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))
            
            if prev_gray is not None:
                # Absolute difference
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                
                # Extract contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                regions = []
                contour_area_sum = 0.0
                h_orig, w_orig = frame.shape[:2]
                
                # Scale contours back to original dimensions
                scale_x = w_orig / 320.0
                scale_y = h_orig / 240.0
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500: # Filter noise
                        x, y, w, h = cv2.boundingRect(contour)
                        # Bounding box scaled back to original resolution
                        regions.append([
                            float(x * scale_x),
                            float(y * scale_y),
                            float((x + w) * scale_x),
                            float((y + h) * scale_y)
                        ])
                        contour_area_sum += area
                
                # Normalize raw score as a percentage of downsampled resolution area
                raw_score = (contour_area_sum / (320.0 * 240.0)) * 100.0
                
                # Apply z-score primitives
                stats.update(raw_score)
                z_score = stats.compute_zscore(raw_score)
                is_abrupt = detector.update(z_score)
                
                if raw_score > 0.0:
                    evidence_list.append({
                        "timestamp_sec": float(current_sec),
                        "motion_score": float(raw_score),
                        "z_score": float(z_score),
                        "is_abrupt": bool(is_abrupt),
                        "regions": regions
                    })
                    
            prev_gray = gray
            current_sec += self.sample_rate_sec
            
        cap.release()
        return evidence_list

# Singleton Instance
motion_detection_service = MotionDetectionService()
