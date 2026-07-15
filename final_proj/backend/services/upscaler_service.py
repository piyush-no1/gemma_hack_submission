import os
import sys
import time
import numpy as np
import cv2
import onnxruntime as ort
from ..config import settings

class UpscalerService:
    def __init__(self, model_path: str = settings.UPSCALER_MODEL_PATH):
        # Resolve path relative to settings or project root
        self.model_path = os.path.abspath(model_path)
        self.session = None
        self.input_name = None
        self.output_name = None
        self._setup_cuda_dlls()
        self._initialize_session()

    def _setup_cuda_dlls(self):
        # Add NVIDIA CUDA/cuDNN DLL paths on Windows so ORT can load them.
        if sys.platform == 'win32':
            import site
            search_paths = site.getsitepackages()
            if hasattr(site, 'getusersitepackages'):
                search_paths.append(site.getusersitepackages())
                
            registered = False
            for sp_dir in search_paths:
                nvidia_base = os.path.join(sp_dir, "nvidia")
                if os.path.exists(nvidia_base):
                    for sub in ["cuda_runtime", "cublas", "cudnn", "cufft", "curand", "cusolver", "cusparse", "nvjitlink", "cuda_nvrtc"]:
                        bin_path = os.path.join(nvidia_base, sub, "bin")
                        if os.path.exists(bin_path):
                            try:
                                os.add_dll_directory(bin_path)
                                registered = True
                            except Exception:
                                pass
            if registered:
                # Add to PATH env as fallback
                pass

    def _initialize_session(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
        opts.inter_op_num_threads = 1
        
        available_providers = ort.get_available_providers()
        
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 1 * 1024 * 1024 * 1024, # Limit to 1GB VRAM for safety
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
            ))
        providers.append("CPUExecutionProvider")
        
        try:
            if os.path.exists(self.model_path):
                self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
                inputs = self.session.get_inputs()
                self.input_name = inputs[0].name
                outputs = self.session.get_outputs()
                self.output_name = outputs[0].name
                print(f"[Upscaler] ONNX Session initialized using: {self.session.get_providers()[0]} from {self.model_path}")
            else:
                print(f"[Upscaler] Model weights not found at '{self.model_path}'. Running in Bicubic-interpolation bypass mode.")
        except Exception as e:
            print(f"[Upscaler] Error loading ONNX model: {e}. Falling back to CPU...")
            try:
                if os.path.exists(self.model_path):
                    self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
                    inputs = self.session.get_inputs()
                    self.input_name = inputs[0].name
                    outputs = self.session.get_outputs()
                    self.output_name = outputs[0].name
            except Exception as cpu_err:
                print(f"[Upscaler] CPU Fallback failed: {cpu_err}")

    def warmup(self, width: int = 256, height: int = 144):
        if not self.session:
            return
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            for _ in range(2):
                _ = self.upscale(dummy_frame)
            print("[Upscaler] Warmup completed successfully.")
        except Exception as e:
            print(f"[Upscaler] Warmup failed: {e}")

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        if not self.session:
            # Fallback to OpenCV Bicubic interpolation
            h, w = frame.shape[:2]
            return cv2.resize(frame, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

        try:
            # Pre-processing
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_norm, (2, 0, 1))
            input_tensor = np.expand_dims(img_transposed, axis=0)

            # Inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            output_tensor = outputs[0]

            # Post-processing
            output_tensor = np.squeeze(output_tensor, axis=0)
            output_tensor = np.clip(output_tensor, 0.0, 1.0)
            img_out = np.transpose(output_tensor, (1, 2, 0))
            img_out_uint8 = (img_out * 255.0).round().astype(np.uint8)
            upscaled_frame = cv2.cvtColor(img_out_uint8, cv2.COLOR_RGB2BGR)
            return upscaled_frame
        except Exception as e:
            print(f"[Upscaler] Upscale error: {e}. Falling back to Bicubic interpolation.")
            h, w = frame.shape[:2]
            return cv2.resize(frame, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

# Singleton Instance
upscaler_service = UpscalerService()
