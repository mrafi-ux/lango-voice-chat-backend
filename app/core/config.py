"""Application configuration."""

import json
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = "sqlite+aiosqlite:///./voicecare.db"
    
    # Server
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    
    # External services
    libre_translate_url: str = "https://libretranslate.com/translate"
    
    # CORS
    cors_origins: str = '["http://localhost:3000"]'
    
    # Voice Settings
    max_voice_seconds: int = 120
    
    # STT Configuration
    stt_provider: str = "elevenlabs"  # options: elevenlabs, openai, whisper
    stt_fallback_enabled: bool = True
    whisper_model: str = "tiny"  # Changed from "small" to "tiny" for better reliability
    whisper_compute: str = "cpu"
    whisper_beam_size: int = 1
    whisper_vad: bool = True
    whisper_temperature: float = 0.0
    whisper_no_speech_threshold: float = 0.45
    elevenlabs_stt_model: str = "scribe_v1"
    openai_api_key: Optional[str] = None
    openai_stt_model: str = "whisper-1"
    
    # TTS Configuration
    tts_provider: str = "elevenlabs"  # options: elevenlabs, openai, browser
    
    # Translation Configuration
    translation_provider: str = "libre"  # options: libre, openai
    openai_translate_model: str = "gpt-4o"
    piper_voices_dir: str = "./voices/piper"
    elevenlabs_api_key: Optional[str] = None  # Set your valid API key here
    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "alloy"
    
    # Authentication
    auth_secret_key: str = "your-secret-key-change-in-production"
    auth_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    
    # Logging
    log_level: str = "INFO"

    # Translation
    # 'auto' uses OpenAI if either STT or TTS is set to OpenAI; otherwise uses 'libre'
    translation_provider: str = "auto"  # options: auto, openai, libre
    openai_translate_model: str = "gpt-4o"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from JSON string."""
        try:
            return json.loads(self.cors_origins)
        except json.JSONDecodeError:
            return ["http://localhost:3000"]
    
    @property
    def translation_provider_effective(self) -> str:
        """Get effective translation provider based on STT provider."""
        if self.stt_provider == "openai" and self.openai_api_key:
            return "openai"
        return self.translation_provider

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def translation_provider_effective(self) -> str:
        """Effective translation provider based on config."""
        val = (self.translation_provider or "auto").lower()
        if val in {"openai", "libre"}:
            return val
        # auto mode: prefer OpenAI if used elsewhere
        if (self.stt_provider or "").lower() == "openai" or (self.tts_provider or "").lower() == "openai":
            return "openai"
        return "libre"


# Global settings instance
settings = Settings()
