import os
import base64
from typing import Protocol
from ..config import settings
from ..agent.prompts.gemma_vision_prompt import GEMMA_VISION_SYSTEM_PROMPT, GEMMA_VISION_USER_PROMPT

class GemmaVisionProvider(Protocol):
    async def generate_with_image(self, image_bytes: bytes, prompt: str) -> str:
        ...

class OllamaGemmaVisionProvider:
    def __init__(self, model_name: str = settings.OLLAMA_VISION_MODEL):
        self.model_name = model_name

    async def generate_with_image(self, image_bytes: bytes, prompt: str) -> str:
        import ollama
        # System instructions combined with prompt since Ollama's vision endpoints
        # usually run as standard user chat sequences.
        full_prompt = f"{GEMMA_VISION_SYSTEM_PROMPT}\n\n{prompt}"
        try:
            # We run in executor to keep it async and non-blocking
            import asyncio
            from functools import partial
            
            loop = asyncio.get_event_loop()
            func = partial(
                ollama.generate,
                model=self.model_name,
                prompt=full_prompt,
                images=[image_bytes]
            )
            response = await loop.run_in_executor(None, func)
            return response.get("response", "").strip()
        except Exception as e:
            print(f"[OllamaVision] Local vision inference failed: {e}")
            raise e

class GoogleGenAIGemmaVisionProvider:
    def __init__(self, model_name: str = settings.GEMINI_VISION_MODEL):
        self.model_name = model_name
        self.api_key = settings.get_google_api_key()
        self.client = None
        if self.api_key:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)

    async def generate_with_image(self, image_bytes: bytes, prompt: str) -> str:
        if not self.client:
            raise ValueError("GEMINI_API_KEY is not configured for Cloud vision provider.")
            
        import PIL.Image
        import io
        import asyncio
        from google.genai import types
        
        # Convert bytes to PIL Image
        pil_image = PIL.Image.open(io.BytesIO(image_bytes))
        
        # Helper to execute blocking SDK call in event loop executor
        def block_call():
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, pil_image],
                config=types.GenerateContentConfig(
                    system_instruction=GEMMA_VISION_SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            )
            return response.text

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, block_call)
            return result.strip() if result else ""
        except Exception as e:
            print(f"[GeminiVision] Cloud vision inference failed: {e}")
            raise e

class GemmaVisionService:
    def __init__(self):
        self.provider = None
        self.provider_name = settings.GEMMA_VISION_PROVIDER
        self._initialize_provider()

    def _initialize_provider(self):
        if self.provider_name == "google_genai":
            try:
                self.provider = GoogleGenAIGemmaVisionProvider()
                print("[GemmaVision] Initialized Google GenAI Cloud Vision provider.")
            except Exception as e:
                print(f"[GemmaVision] Google GenAI provider initialization failed: {e}. Falling back to Ollama.")
                self.provider = OllamaGemmaVisionProvider()
        else:
            self.provider = OllamaGemmaVisionProvider()
            print(f"[GemmaVision] Initialized local Ollama Vision provider (model: {settings.OLLAMA_VISION_MODEL}).")

    async def transcribe_frame(self, image_bytes: bytes) -> str:
        """Helper to run visual transcription prompt on frame bytes."""
        return await self.provider.generate_with_image(image_bytes, GEMMA_VISION_USER_PROMPT)

# Singleton Instance (init at startup)
gemma_vision_service = None
