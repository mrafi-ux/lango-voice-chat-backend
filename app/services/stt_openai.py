"""OpenAI STT service using Whisper API via OpenAI SDK."""

from typing import Optional, Dict

# Make OpenAI dependency optional so the app can start without it
try:  # pragma: no cover - import guard
    from openai import OpenAI  # type: ignore
except Exception:  # ModuleNotFoundError or other import-time issues
    OpenAI = None  # type: ignore[assignment]

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class OpenAISTTService:
    """Speech-to-Text via OpenAI's hosted Whisper."""

    def __init__(self) -> None:
        self.api_key = getattr(settings, 'openai_api_key', None)
        self.model_id = getattr(settings, 'openai_stt_model', 'whisper-1')
        self.client: Optional["OpenAI"] = None

        if OpenAI is None:
            logger.warning("'openai' package not installed; OpenAI STT disabled")
        elif self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to init OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not configured")

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None
    ) -> Dict[str, str]:
        if not self.client:
            return {"text": "", "language": language or "en", "error": "OpenAI API key not configured"}

        try:
            # OpenAI expects (filename, bytes, mimetype)
            resp = self.client.audio.transcriptions.create(
                model=self.model_id,
                file=("audio.webm", audio_data, "audio/webm"),
                language=(language or 'en')
            )

            text = getattr(resp, 'text', '') or ''
            logger.info(f"OpenAI STT transcription: '{text[:100]}...'")
            return {"text": text.strip(), "language": language or 'en'}

        except Exception as e:
            logger.error(f"OpenAI STT failed: {e}")
            return {"text": "", "language": language or "en", "error": str(e)}


openai_stt_service = OpenAISTTService()

