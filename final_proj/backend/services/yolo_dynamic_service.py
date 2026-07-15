import os
import cv2
import numpy as np
import asyncio
from typing import List, Dict, Any
from ultralytics import YOLOWorld
from ..config import settings

import torch

class YoloDynamicDetector:
    def __init__(self, model_name: str = settings.YOLO_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.lock = asyncio.Lock()
        self._initialize_model()

    def _initialize_model(self):
        print(f"[YOLO Dynamic] Loading dedicated dynamic YOLO-World model '{self.model_name}' on device '{self.device}'...")
        self.model = YOLOWorld(self.model_name)
        self.model.to(self.device)

    async def detect_with_custom_classes(self, img_np: np.ndarray, classes: List[str]) -> List[Dict[str, Any]]:
        """
        Sets vocabulary dynamically per-request and runs YOLO inference under an async lock
        to ensure thread-safe class assignment under concurrency.
        """
        if not classes:
            return []

        async with self.lock:
            # Set the custom vocabulary requested by the Judge
            self.model.set_classes(classes)
            
            # Run inference
            # To run blockings in executor is cleaner, but ultralytics model inference is quick
            loop = asyncio.get_event_loop()
            
            def run_inference():
                return self.model(
                    img_np,
                    conf=settings.YOLO_CONF_THRESHOLD,
                    imgsz=settings.YOLO_IMG_SIZE,
                    verbose=False,
                    device=self.device
                )
                
            results = await loop.run_in_executor(None, run_inference)
            
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
yolo_dynamic_service = None
