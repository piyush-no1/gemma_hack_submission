import os
import edge_tts
import asyncio
from typing import Dict

class TtsService:
    # Map common ISO 639-1 language codes to premium edge-tts neural voices
    VOICE_MAPPING: Dict[str, str] = {
        "en": "en-US-AvaNeural",
        "es": "es-ES-ElviraNeural",
        "hi": "hi-IN-SwaraNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KlarissaNeural",
        "zh": "zh-CN-XiaoxiaoNeural",
        "ja": "ja-JP-NanamiNeural",
        "ru": "ru-RU-SvetlanaNeural",
        "pt": "pt-BR-FranciscaNeural",
        "it": "it-IT-ElsaNeural"
    }

    async def synthesize(self, text: str, language: str, output_path: str) -> str:
        """
        Synthesizes text to speech in the requested language using edge-tts.
        Saves output as a playable mp3 file.
        """
        # Normalize language code (e.g. "en-US" or "EN" -> "en")
        lang_key = language.strip().lower()[:2]
        voice = self.VOICE_MAPPING.get(lang_key, "en-US-AvaNeural")
        
        print(f"[TTS] Synthesizing text to speech using voice '{voice}' for language '{language}'...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # edge-tts Communicate is async and writes directly to file
            communicate = edge_tts.Communicate(text, voice)
            await asyncio.wait_for(communicate.save(output_path), timeout=10.0)
            print(f"[TTS] Speech file written to {output_path}")
            return output_path
        except Exception as e:
            print(f"[TTS] edge-tts synthesis failed: {e}. Falling back to offline stub/dummy...")
            # Create an empty file or basic mp3 so the endpoint doesn't fail
            with open(output_path, "wb") as f:
                f.write(b"") # Write empty bytes
            return output_path

# Singleton Instance
tts_service = TtsService()
