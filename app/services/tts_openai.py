"""OpenAI TTS service using tts-1 via OpenAI SDK."""

from typing import Optional, Tuple

# Make OpenAI dependency optional so the app can start without it
try:  # pragma: no cover - import guard
    from openai import OpenAI  # type: ignore
except Exception:  # ModuleNotFoundError or other import-time issues
    OpenAI = None  # type: ignore[assignment]

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class OpenAITTSService:
    def __init__(self) -> None:
        self.api_key = getattr(settings, 'openai_api_key', None)
        self.model_id = getattr(settings, 'openai_tts_model', 'tts-1')
        self.voice = getattr(settings, 'openai_tts_voice', 'alloy')
        self.client: Optional["OpenAI"] = None
        if OpenAI is None:
            logger.warning("'openai' package not installed; OpenAI TTS disabled")
        elif self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to init OpenAI client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not configured")

    async def synthesize(self, text: str, lang: str, voice_hint: Optional[str] = None, sender_gender: Optional[str] = None, sender_id: Optional[str] = None) -> Tuple[bytes, str, bool, Optional[str]]:
        if not self.client:
            return b"", "audio/mpeg", True, None

        try:
            # Get or assign persistent gender for TTS
            effective_gender = await self._get_or_assign_tts_gender(sender_gender, sender_id)
            
            # Select voice based on gender if no voice_hint provided
            voice = voice_hint or self._select_voice_by_gender(effective_gender)
            speech = self.client.audio.speech.create(
                model=self.model_id,
                voice=voice,
                input=text
            )
            audio_bytes: bytes = speech.content  # type: ignore[attr-defined]
            return audio_bytes, "audio/mpeg", False, voice
        except Exception as e:
            logger.error(f"OpenAI TTS failed: {e}")
            return b"", "audio/mpeg", True, None

    def _select_voice_by_gender(self, sender_gender: Optional[str] = None) -> str:
        """Select OpenAI TTS voice based on sender gender."""
        # OpenAI TTS voices and their gender characteristics
        gender_voices = {
            "female": ["nova", "shimmer"],  # Higher-pitched, more feminine voices
            "male": ["echo", "fable", "onyx"],  # Lower-pitched, more masculine voices
            "neutral": ["alloy"]  # Neutral voice as default
        }
        
        # Determine gender preference
        gender_key = "neutral"  # Default fallback
        if sender_gender:
            gender_lower = sender_gender.lower().strip()
            logger.info(f"OpenAI TTS - Processing sender gender: '{sender_gender}' -> '{gender_lower}'")
            if gender_lower in ["female", "woman", "f", "fem"]:
                gender_key = "female"
            elif gender_lower in ["male", "man", "m", "mas"]:
                gender_key = "male"
            else:
                # If gender is not clearly male/female, randomly choose between male and female
                import random
                gender_key = random.choice(["male", "female"])
                logger.info(f"OpenAI TTS - Unknown gender '{sender_gender}', randomly selected: {gender_key}")
        else:
            logger.info("OpenAI TTS - No sender gender provided, using neutral voice")
        
        # Select first available voice for the gender
        voices = gender_voices.get(gender_key, gender_voices["neutral"])
        selected_voice = voices[0] if voices else "alloy"
        
        logger.info(f"Selected OpenAI voice '{selected_voice}' for gender '{gender_key}'")
        return selected_voice

    async def _get_or_assign_tts_gender(self, sender_gender: Optional[str], sender_id: Optional[str]) -> str:
        """Get or assign a persistent TTS gender for the user."""
        # If we have a clear gender, use it
        if sender_gender:
            gender_lower = sender_gender.lower().strip()
            if gender_lower in ["female", "woman", "f", "fem"]:
                return "female"
            elif gender_lower in ["male", "man", "m", "mas"]:
                return "male"
        
        # If no clear gender and we have a sender_id, check/assign persistent gender
        if sender_id:
            try:
                from ..db.session import AsyncSessionLocal
                from ..db.models import User
                from sqlalchemy import select, update
                
                async with AsyncSessionLocal() as session:
                    # Check if user already has a tts_gender assigned
                    result = await session.execute(
                        select(User.tts_gender).where(User.id == sender_id)
                    )
                    existing_tts_gender = result.scalar_one_or_none()
                    
                    if existing_tts_gender:
                        logger.info(f"Using existing TTS gender for user {sender_id}: {existing_tts_gender}")
                        return existing_tts_gender
                    
                    # Assign a random gender and store it
                    import random
                    assigned_gender = random.choice(["male", "female"])
                    
                    await session.execute(
                        update(User)
                        .where(User.id == sender_id)
                        .values(tts_gender=assigned_gender)
                    )
                    await session.commit()
                    
                    logger.info(f"Assigned persistent TTS gender '{assigned_gender}' to user {sender_id}")
                    return assigned_gender
                    
            except Exception as e:
                logger.error(f"Failed to get/assign TTS gender for user {sender_id}: {e}")
        
        # Fallback: randomly choose for this session only
        import random
        fallback_gender = random.choice(["male", "female"])
        logger.info(f"No sender_id provided, using random gender for this session: {fallback_gender}")
        return fallback_gender


openai_tts_service = OpenAITTSService()

