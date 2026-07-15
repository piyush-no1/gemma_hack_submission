import os
import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLOWorld
from ..config import settings

import torch

class YoloStaticDetector:
    DEFAULT_VOCABULARY = [
        "person", "child", "crowd", "security guard", "firefighter", "police officer",
        "fire", "smoke", "flame", "explosion",
        "gun", "pistol", "rifle", "knife", "bat",
        "gas cylinder", "chemical barrel", "fuel container", "flammable material", "hazardous material",
        "fire extinguisher", "emergency exit", "first aid kit", "safety helmet", "safety vest",
        "car", "truck", "motorcycle", "bus", "forklift",
        "broken window", "damaged door", "barricade", "traffic cone"
    ]

    def __init__(self, model_name: str = settings.YOLO_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        print(f"[YOLO Static] Loading YOLO-World model '{self.model_name}' on device '{self.device}'...")
        # YOLO-World will automatically download weights from ultralytics if missing
        self.model = YOLOWorld(self.model_name)
        self.model.to(self.device)
        self.model.set_classes(self.DEFAULT_VOCABULARY)
        print("[YOLO Static] YOLO-World classes registered.")

    def detect(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs YOLO static vocab detection on a BGR image array.
        Returns a list of dicts: {"label": str, "confidence": float, "box": [x1, y1, x2, y2]}
        """
        # Run inference
        results = self.model(
            img_np,
            conf=settings.YOLO_CONF_THRESHOLD,
            imgsz=settings.YOLO_IMG_SIZE,
            verbose=False,
            device=self.device
        )
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = self.model.names[cls_id]
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": xyxy
                })
        return detections

# Singleton Instance (init at startup)
yolo_static_service = None
