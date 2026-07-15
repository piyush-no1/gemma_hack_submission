import cv2
import numpy as np
from typing import List, Dict, Any
from ..graph.state import FrameRecord, YoloDetection, MotionEvidence

class CrowdAnalysisService:
    def analyze(
        self,
        frames: List[FrameRecord],
        yolo_detections: List[YoloDetection],
        motion_evidence: List[MotionEvidence]
    ) -> Dict[str, Any]:
        """
        Classifies the crowd state in the video into: stable, congested, abnormal_movement, panic, stampede.
        Uses person counts from YOLO to estimate density and dense Farneback optical flow within
        person bounding box regions to compute velocity magnitude and angular variance.
        """
        if not frames or len(frames) < 2:
            return {
                "people_count": 0.0,
                "density": "Low",
                "state": "stable",
                "flow_direction": "unknown",
                "confidence": 100.0
            }

        # 1. Calculate average person count and density
        person_counts = []
        person_boxes_per_frame = []
        
        for frame in frames:
            # Detections for this frame
            frame_dets = [d for d in yolo_detections if d.label == "person"]
            person_counts.append(len(frame_dets))
            person_boxes_per_frame.append(frame_dets)
            
        avg_people = float(np.mean(person_counts)) if person_counts else 0.0
        
        # Density categorization
        if avg_people < 2.0:
            density_str = "Low"
        elif avg_people < 6.0:
            density_str = "Moderate"
        else:
            density_str = "High"
            
        if avg_people == 0:
            return {
                "people_count": 0.0,
                "density": "Low",
                "state": "stable",
                "flow_direction": "unknown",
                "confidence": 100.0
            }

        # 2. Compute Farneback optical flow on person regions
        flow_magnitudes = []
        flow_angles = []
        
        # Sort frames to ensure correct temporal sequence
        sorted_frames = sorted(frames, key=lambda f: f.timestamp_sec)
        
        for i in range(1, len(sorted_frames)):
            prev_frame = sorted_frames[i-1]
            curr_frame = sorted_frames[i]
            
            img_prev = cv2.imread(prev_frame.image_path, cv2.IMREAD_GRAYSCALE)
            img_curr = cv2.imread(curr_frame.image_path, cv2.IMREAD_GRAYSCALE)
            
            if img_prev is None or img_curr is None:
                continue
                
            # Resize to a smaller common size for fast optical flow
            img_prev_small = cv2.resize(img_prev, (320, 240))
            img_curr_small = cv2.resize(img_curr, (320, 240))
            
            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                img_prev_small, img_curr_small, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Extract flow inside person boxes (scaled to 320x240)
            h_orig, w_orig = img_prev.shape[:2]
            scale_x = 320.0 / w_orig
            scale_y = 240.0 / h_orig
            
            # Current frame index person boxes
            boxes = person_boxes_per_frame[i]
            
            for box_obj in boxes:
                # scale box coordinates
                x1 = int(box_obj.box[0] * scale_x)
                y1 = int(box_obj.box[1] * scale_y)
                x2 = int(box_obj.box[2] * scale_x)
                y2 = int(box_obj.box[3] * scale_y)
                
                # Clip bounds
                x1 = max(0, min(319, x1))
                x2 = max(0, min(319, x2))
                y1 = max(0, min(239, y1))
                y2 = max(0, min(239, y2))
                
                if x2 > x1 and y2 > y1:
                    box_flow = flow[y1:y2, x1:x2]
                    fx = box_flow[..., 0]
                    fy = box_flow[..., 1]
                    
                    mag, ang = cv2.cartToPolar(fx, fy)
                    flow_magnitudes.extend(mag.flatten())
                    flow_angles.extend(ang.flatten())

        mean_mag = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
        var_ang = float(np.var(flow_angles)) if flow_angles else 0.0
        
        # Determine dominant direction
        flow_dir = "unknown"
        if flow_angles:
            avg_ang_rad = np.mean(flow_angles)
            avg_ang_deg = np.degrees(avg_ang_rad) % 360
            if 45 <= avg_ang_deg < 135:
                flow_dir = "downward"
            elif 135 <= avg_ang_deg < 225:
                flow_dir = "leftward"
            elif 225 <= avg_ang_deg < 315:
                flow_dir = "upward"
            else:
                flow_dir = "rightward"

        # 3. Classify Crowd State
        state = "stable"
        confidence = 80.0
        
        # Cross-reference abrupt motion events
        has_abrupt_motion = any(evt.is_abrupt for evt in motion_evidence)
        
        if density_str == "High":
            if mean_mag > 2.5 and var_ang > 1.0:
                if has_abrupt_motion:
                    state = "stampede"
                    confidence = 90.0
                else:
                    state = "panic"
                    confidence = 85.0
            elif mean_mag < 1.0:
                state = "congested"
                confidence = 85.0
            else:
                state = "stable"
        elif density_str == "Moderate":
            if mean_mag > 2.0 and var_ang > 1.2:
                state = "abnormal_movement"
                confidence = 75.0
            else:
                state = "stable"
        else:
            state = "stable"

        return {
            "people_count": float(avg_people),
            "density": density_str,
            "state": state,
            "flow_direction": flow_dir,
            "confidence": float(confidence),
            "flow_magnitude": float(mean_mag),
            "flow_variance": float(var_ang)
        }

# Singleton Instance
crowd_analysis_service = CrowdAnalysisService()
