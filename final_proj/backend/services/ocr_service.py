import easyocr
import cv2
import numpy as np
import asyncio
from typing import List, Dict, Any

import torch

class OcrService:
    def __init__(self, languages: List[str] = None):
        if languages is None:
            languages = ["en"]
        self.languages = languages
        self.reader = None
        self.lock = asyncio.Lock()
        self._initialize_reader()

    def _initialize_reader(self):
        gpu_available = torch.cuda.is_available()
        print(f"[OCR] Loading EasyOCR reader for languages: {self.languages} (GPU={gpu_available})...")
        # Initialize EasyOCR reader dynamically on GPU if available
        self.reader = easyocr.Reader(self.languages, gpu=gpu_available)
        print("[OCR] EasyOCR initialized.")

    async def extract_text(self, img_np: np.ndarray) -> Dict[str, Any]:
        """
        Extracts visible text strings and confidence levels from an image.
        Uses a lock to prevent concurrent execution conflicts in easyocr.
        """
        async with self.lock:
            loop = asyncio.get_event_loop()
            
            def run_ocr():
                # easyocr readtext accepts numpy arrays
                return self.reader.readtext(img_np)
                
            results = await loop.run_in_executor(None, run_ocr)
            
            texts = []
            confidences = []
            for (bbox, text, prob) in results:
                texts.append(text)
                confidences.append(float(prob))
                
            return {
                "text": texts,
                "confidence": confidences
            }

# Singleton Instance (init at startup)
ocr_service = None
