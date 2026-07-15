import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Local LLM runtime (Ollama)
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL_JUDGE: str = "gemma4:e4b"
    OLLAMA_MODEL_REASONING: str = "gemma4:12b"
    OLLAMA_MODEL_OUTPUT: str = "gemma4:4b"
    OLLAMA_EMBEDDING_MODEL: str = "embeddinggemma"

    # Gemma Vision provider: "ollama" | "google_genai"
    GEMMA_VISION_PROVIDER: str = "ollama"
    OLLAMA_VISION_MODEL: str = "gemma4:e4b"
    
    # Google API Key can be set via GOOGLE_API_KEY or GEMINI_API_KEY
    GOOGLE_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GEMINI_VISION_MODEL: str = "gemini-2.5-flash"

    # YOLO-World
    YOLO_MODEL_NAME: str = "yolov8s-world.pt"
    YOLO_CONF_THRESHOLD: float = 0.3
    YOLO_IMG_SIZE: int = 1280

    # Upscaler
    UPSCALER_MODEL_PATH: str = "models/realesr-general-x4v3.onnx"

    # V-JEPA
    VJEPA_MODEL_NAME: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"

    # TTS provider: "edge_tts" | "stub"
    TTS_PROVIDER: str = "edge_tts"

    # Orchestration
    MAX_TOOL_ITERATIONS: int = 1
    GRAPH_RECURSION_LIMIT: int = 12
    JUDGE_CONFIDENCE_THRESHOLD: float = 0.65  # Skip tools+reasoning above this confidence

    # Application Settings
    MAX_UPLOAD_MB: int = 500
    STORAGE_ROOT: str = "./storage"
    VECTORSTORE_ROOT: str = "./vectorstore"
    PORT: int = 8080
    HOST: str = "127.0.0.1"

    def get_google_api_key(self) -> str:
        return self.GOOGLE_API_KEY or self.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

# Load configuration singleton
settings = Settings()
