"""Translation via OpenAI Chat Completions."""

from typing import Optional

# Make OpenAI dependency optional so the app can start without it
try:  # pragma: no cover - import guard
    from openai import OpenAI  # type: ignore
except Exception:  # ModuleNotFoundError or other import-time issues
    OpenAI = None  # type: ignore[assignment]

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class OpenAITranslationService:
    def __init__(self) -> None:
        self.api_key = getattr(settings, 'openai_api_key', None)
        self.model_id = getattr(settings, 'openai_translate_model', 'gpt-4o')
        self.client: Optional["OpenAI"] = None
        if OpenAI is None:
            logger.warning("'openai' package not installed; OpenAI translation disabled")
        elif self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to init OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not configured")

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not self.client:
            return text
        prompt = f"Translate into {target_lang}. Respond with translation only."
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI translation failed: {e}")
            return text


openai_translation_service = OpenAITranslationService()

