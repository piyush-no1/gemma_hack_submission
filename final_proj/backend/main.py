import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .config import settings

# Import services to initialize them as singletons
from .services.vjepa_service import VJepaAnomalyService
from .services.yolo_static_service import YoloStaticDetector
from .services.yolo_dynamic_service import YoloDynamicDetector
from .services.ocr_service import OcrService
from .services.gemma_vision_service import GemmaVisionService
from .services.upscaler_service import upscaler_service
from .services.attention_rollout_service import AttentionRolloutService
from .api import routes_jobs

# We define services as module variables initialized during startup
from .services import vjepa_service as vjepa_mod
from .services import yolo_static_service as yolo_static_mod
from .services import yolo_dynamic_service as yolo_dynamic_mod
from .services import ocr_service as ocr_mod
from .services import gemma_vision_service as gemma_vision_mod
from .services import attention_rollout_service as attention_rollout_mod

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("======================================================================")
    print("Starting Multi-Modal Video Safety Surveillance Platform...")
    print("======================================================================")
    
    # 1. Initialize heavy singletons
    vjepa_mod.vjepa_service = VJepaAnomalyService()
    yolo_static_mod.yolo_static_service = YoloStaticDetector()
    yolo_dynamic_mod.yolo_dynamic_service = YoloDynamicDetector()
    ocr_mod.ocr_service = OcrService()
    gemma_vision_mod.gemma_vision_service = GemmaVisionService()
    
    # Inject loaded VJepa service instance into Attention Rollout service
    attention_rollout_mod.attention_rollout_service = AttentionRolloutService(vjepa_mod.vjepa_service)
    
    # 2. Warm up upscaler GPU kernels
    print("[Lifespan] Running GPU warm-up phase on Upscaler model...")
    upscaler_service.warmup()
    
    print("[Lifespan] Initialization complete. Server is ready.")
    yield
    print("[Lifespan] Shutting down safety surveillance engine.")

app = FastAPI(
    title="Video Safety Surveillance Platform",
    description="Agentic Multi-Modal Video Safety Platform with V-JEPA, YOLO, OCR, Crowd Flow, and RAG",
    version="2.0",
    lifespan=lifespan
)

# Enable CORS for local cross-origin connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(routes_jobs.router, prefix="/api")

# Mount frontend files statically
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"[Warning] Frontend directory not found at {frontend_path}. Statically served dashboard is disabled.")

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
