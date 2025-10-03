"""ElevenLabs TTS service for premium voice synthesis."""

import asyncio
from typing import Optional, Dict, List, Any
import base64

import httpx

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class ElevenLabsTTSService:
    """ElevenLabs-based Text-to-Speech service."""
    
    def __init__(self):
        self.voices_cache: Optional[List[Dict]] = None
        self.base_url = "https://api.elevenlabs.io/v1"
        
    async def _load_voices_cache(self):
        """Load available ElevenLabs voices."""
        if self.voices_cache is not None:
            return
            
        if not settings.elevenlabs_api_key:
            logger.warning("ElevenLabs API key not configured")
            self.voices_cache = []
            return
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/voices",
                    headers={"xi-api-key": settings.elevenlabs_api_key},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.voices_cache = data.get("voices", [])
                    logger.info(f"Loaded {len(self.voices_cache)} ElevenLabs voices")
                elif response.status_code == 401:
                    logger.warning("ElevenLabs API key invalid - disabling ElevenLabs TTS")
                    self.voices_cache = []
                    # Don't clear the API key here as it might be valid for other operations
                else:
                    logger.error(f"Failed to load ElevenLabs voices: {response.status_code}")
                    self.voices_cache = []
                    
        except Exception as e:
            logger.error(f"Error loading ElevenLabs voices: {e}")
            self.voices_cache = []
    
    async def get_voices_list(self) -> List[Dict]:
        """Get list of available voices with details."""
        await self._load_voices_cache()
        return self.voices_cache or []
    
    async def synthesize_elevenlabs(
        self, 
        text: str, 
        lang: str, 
        voice_hint: Optional[str] = None,
        sender_gender: Optional[str] = None,
        sender_id: Optional[str] = None
    ) -> tuple[bytes, str, bool, Optional[str]]:
        """
        Synthesize speech using ElevenLabs.
        
        Args:
            text: Text to synthesize
            lang: Target language code
            voice_hint: Optional voice ID or name hint
            sender_gender: Optional sender gender to match voice gender
            sender_id: Optional sender ID for persistent gender assignment
            
        Returns:
            Tuple of (audio_bytes, content_type, needs_browser_fallback, voice_used)
        """
        if not settings.elevenlabs_api_key:
            logger.warning("ElevenLabs API key not configured")
            return b"", "audio/mpeg", True, None
        
        await self._load_voices_cache()
        
        # Get or assign persistent gender for TTS
        effective_gender = await self._get_or_assign_tts_gender(sender_gender, sender_id)
        
        # Find best voice for language and gender
        voice_id = self._find_best_voice(lang, voice_hint, effective_gender)
        
        if not voice_id:
            logger.warning(f"No ElevenLabs voice found for language: {lang}")
            return b"", "audio/mpeg", True, None
        
        try:
            # Synthesize with ElevenLabs
            audio_bytes = await self._synthesize_with_voice(text, voice_id)
            return audio_bytes, "audio/mpeg", False, voice_id
            
        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            return b"", "audio/mpeg", True, None
    
    def _find_best_voice(self, lang: str, voice_hint: Optional[str] = None, sender_gender: Optional[str] = None) -> Optional[str]:
        """Find the best voice ID for a language."""
        if not self.voices_cache:
            # If we could not load voices, try hardcoded defaults
            return self._fallback_voice_for(lang)
        
        # If voice hint provided, try to match it first (could be voice_id or name)
        if voice_hint:
            # Try exact voice_id match first
            for voice in self.voices_cache:
                if voice["voice_id"] == voice_hint:
                    logger.info(f"Using exact voice ID match: {voice['name']}")
                    return voice["voice_id"]
            
            # Try name match
            for voice in self.voices_cache:
                if voice_hint.lower() in voice["name"].lower():
                    logger.info(f"Using voice name match: {voice['name']}")
                    return voice["voice_id"]
        
        # Language preference mapping for multilingual model with gender information
        lang_preferences = {
            "en": {
                "female": ["Rachel", "Sarah", "Laura", "Alice", "Matilda", "Jessica"],
                "male": ["Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Rachel", "Sarah", "Clyde", "Roger", "Laura", "Thomas", "Charlie", "George", "Alice", "Matilda", "Callum", "River", "Harry", "Liam", "Will", "Jessica", "Eric", "Chris", "Brian", "Daniel"]
            },
            "es": {
                "female": ["Matilda", "Isabella", "Valentina", "Rachel", "Sarah", "Laura", "Alice"],
                "male": ["Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Matilda", "Isabella", "Valentina", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice"]
            },
            "fr": {
                "female": ["Charlotte", "Alice", "Camille", "Rachel", "Sarah", "Laura", "Matilda"],
                "male": ["Antoine", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Charlotte", "Alice", "Camille", "Antoine", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Matilda"]
            },
            "de": {
                "female": ["Giselle", "Ingrid", "Rachel", "Sarah", "Laura", "Alice", "Matilda"],
                "male": ["Hans", "Klaus", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Hans", "Klaus", "Giselle", "Ingrid", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice", "Matilda"]
            },
            "it": {
                "female": ["Bianca", "Giulia", "Rachel", "Sarah", "Laura", "Alice", "Matilda"],
                "male": ["Giorgio", "Marco", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Giorgio", "Bianca", "Marco", "Giulia", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice", "Matilda"]
            },
            "pt": {
                "female": ["Camila", "Fernanda", "Rachel", "Sarah", "Laura", "Alice", "Matilda"],
                "male": ["Ricardo", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Camila", "Ricardo", "Fernanda", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice", "Matilda"]
            },
            "ar": {
                "female": ["Amara", "Rachel", "Sarah", "Laura", "Alice", "Matilda"],
                "male": ["Khalil", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Khalil", "Amara", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice", "Matilda"]
            },
            "hi": {
                "female": ["Aditi", "Rachel", "Sarah", "Laura", "Alice", "Matilda"],
                "male": ["Ravi", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Aditi", "Ravi", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice", "Matilda"]
            },
            "ja": {
                "female": ["Akiko", "Rachel", "Sarah", "Laura", "Alice", "Matilda"],
                "male": ["Takeshi", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Takeshi", "Akiko", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice", "Matilda"]
            },
            "ko": {
                "female": ["Soo-jin", "Rachel", "Sarah", "Laura", "Alice", "Matilda"],
                "male": ["Jin", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Jin", "Soo-jin", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice", "Matilda"]
            },
            "zh": {
                "female": ["Li", "Rachel", "Sarah", "Laura", "Alice", "Matilda"],
                "male": ["Wei", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel"],
                "neutral": ["Wei", "Li", "Clyde", "Roger", "Thomas", "Charlie", "George", "Callum", "River", "Harry", "Liam", "Will", "Eric", "Chris", "Brian", "Daniel", "Rachel", "Sarah", "Laura", "Alice", "Matilda"]
            }
        }
        
        lang_code = lang.split('-')[0].lower()
        lang_voices = lang_preferences.get(lang_code, lang_preferences["en"])
        
        # Determine gender preference
        gender_key = "neutral"  # Default fallback
        if sender_gender:
            gender_lower = sender_gender.lower().strip()
            logger.info(f"Processing sender gender: '{sender_gender}' -> '{gender_lower}'")
            if gender_lower in ["female", "woman", "f", "fem"]:
                gender_key = "female"
            elif gender_lower in ["male", "man", "m", "mas"]:
                gender_key = "male"
            else:
                # If gender is not clearly male/female, randomly choose between male and female
                import random
                gender_key = random.choice(["male", "female"])
                logger.info(f"Unknown gender '{sender_gender}', randomly selected: {gender_key}")
        else:
            logger.info("No sender gender provided, using neutral voice")
        
        # Get preferred names for the gender
        preferred_names = lang_voices.get(gender_key, lang_voices["neutral"])
        
        # Try to find voice by language and gender preference
        for preferred_name in preferred_names:
            for voice in self.voices_cache:
                if preferred_name.lower() in voice["name"].lower():
                    logger.info(f"Using language and gender preference match: {voice['name']} for {lang_code} ({gender_key})")
                    return voice["voice_id"]
        
        # Fallback to first available voice
        if self.voices_cache:
            fallback_voice = self.voices_cache[0]
            logger.info(f"Using fallback voice: {fallback_voice['name']}")
            return fallback_voice["voice_id"]
        
        # As a last resort, return a known public voice id (Rachel)
        return self._fallback_voice_for(lang, sender_gender)

    def _fallback_voice_for(self, lang: str, sender_gender: Optional[str] = None) -> Optional[str]:
        """Return a known public ElevenLabs premade voice id as fallback.
        The multilingual model can speak many languages regardless of the voice name.
        """
        # Known public ElevenLabs voice IDs
        # Rachel (female): 21m00Tcm4TlvDq8ikWAM
        # Antoni (male): ErXwobaYiN019PkySvjV
        # Bella (female): EXAVITQu4vr4xnSDxMaL
        # Elli (female): MF3mGyEYCl7XYWbV9V6O
        # Josh (male): TxGEqnHWrfWFTfGW9XjX
        # Arnold (male): VR6AewLTigWG4xSOukaG
        # Adam (male): pNInz6obpgDQGcFmaJgB
        # Sam (male): yoZ06aMxZJJ28mfd3POQ
        
        # Gender-based fallback voices
        gender_voices = {
            "female": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "male": "ErXwobaYiN019PkySvjV",    # Antoni
            "neutral": "21m00Tcm4TlvDq8ikWAM"  # Rachel as default
        }
        
        # Determine gender preference
        gender_key = "neutral"  # Default fallback
        if sender_gender:
            gender_lower = sender_gender.lower().strip()
            logger.info(f"Fallback voice selection - Processing sender gender: '{sender_gender}' -> '{gender_lower}'")
            if gender_lower in ["female", "woman", "f", "fem"]:
                gender_key = "female"
            elif gender_lower in ["male", "man", "m", "mas"]:
                gender_key = "male"
            else:
                # If gender is not clearly male/female, randomly choose between male and female
                import random
                gender_key = random.choice(["male", "female"])
                logger.info(f"Fallback voice selection - Unknown gender '{sender_gender}', randomly selected: {gender_key}")
        else:
            logger.info("Fallback voice selection - No sender gender provided, using neutral voice")
        
        voice_id = gender_voices.get(gender_key, gender_voices["neutral"])
        logger.info(f"Using hardcoded fallback voice id for {lang} ({gender_key}): {voice_id}")
        return voice_id
    
    async def _synthesize_with_voice(self, text: str, voice_id: str) -> bytes:
        """Synthesize text with specific voice."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": settings.elevenlabs_api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_multilingual_v2",  # Use multilingual model
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75,
                            "style": 0.0,
                            "use_speaker_boost": True
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    audio_bytes = response.content
                    logger.info(f"ElevenLabs synthesis successful: {len(audio_bytes)} bytes with voice {voice_id}")
                    return audio_bytes
                else:
                    logger.error(f"ElevenLabs synthesis failed: {response.status_code} - {response.text}")
                    raise RuntimeError(f"ElevenLabs API error: {response.status_code}")
                    
        except httpx.TimeoutException:
            logger.error("ElevenLabs synthesis timed out")
            raise RuntimeError("ElevenLabs synthesis timed out")
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}")
            raise
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """Get list of supported languages with available voices."""
        # Updated for multilingual model with gender information
        return [
            {
                "code": "en",
                "name": "English", 
                "voices": {
                    "female": ["Rachel", "Bella", "Elli"],
                    "male": ["Antoni", "Josh", "Arnold", "Adam", "Sam"],
                    "all": ["Rachel", "Bella", "Antoni", "Elli", "Josh", "Arnold", "Adam", "Sam"]
                },
                "count": 8
            },
            {
                "code": "es", 
                "name": "Spanish",
                "voices": {
                    "female": ["Matilda", "Isabella", "Valentina"],
                    "male": ["Diego"],
                    "all": ["Matilda", "Isabella", "Diego", "Valentina"]
                },
                "count": 4
            },
            {
                "code": "fr",
                "name": "French", 
                "voices": {
                    "female": ["Charlotte", "Alice", "Camille"],
                    "male": ["Antoine"],
                    "all": ["Charlotte", "Alice", "Antoine", "Camille"]
                },
                "count": 4
            },
            {
                "code": "de",
                "name": "German",
                "voices": {
                    "female": ["Giselle", "Ingrid"],
                    "male": ["Hans", "Klaus"],
                    "all": ["Hans", "Giselle", "Klaus", "Ingrid"]
                },
                "count": 4
            },
            {
                "code": "it",
                "name": "Italian",
                "voices": {
                    "female": ["Bianca", "Giulia"],
                    "male": ["Giorgio", "Marco"],
                    "all": ["Giorgio", "Bianca", "Marco", "Giulia"]
                },
                "count": 4
            },
            {
                "code": "pt",
                "name": "Portuguese",
                "voices": {
                    "female": ["Camila", "Fernanda"],
                    "male": ["Ricardo"],
                    "all": ["Camila", "Ricardo", "Fernanda"]
                },
                "count": 3
            },
            {
                "code": "ar",
                "name": "Arabic",
                "voices": {
                    "female": ["Amara"],
                    "male": ["Khalil"],
                    "all": ["Khalil", "Amara"]
                },
                "count": 2
            },
            {
                "code": "hi",
                "name": "Hindi",
                "voices": {
                    "female": ["Aditi"],
                    "male": ["Ravi"],
                    "all": ["Aditi", "Ravi"]
                },
                "count": 2
            },
            {
                "code": "ja",
                "name": "Japanese",
                "voices": {
                    "female": ["Akiko"],
                    "male": ["Takeshi"],
                    "all": ["Takeshi", "Akiko"]
                },
                "count": 2
            },
            {
                "code": "ko",
                "name": "Korean",
                "voices": {
                    "female": ["Soo-jin"],
                    "male": ["Jin"],
                    "all": ["Jin", "Soo-jin"]
                },
                "count": 2
            },
            {
                "code": "zh",
                "name": "Chinese",
                "voices": {
                    "female": ["Li"],
                    "male": ["Wei"],
                    "all": ["Wei", "Li"]
                },
                "count": 2
            }
        ]
    
    async def check_elevenlabs_available(self) -> bool:
        """Check if ElevenLabs API is available."""
        if not settings.elevenlabs_api_key:
            return False
            
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/voices",
                    headers={"xi-api-key": settings.elevenlabs_api_key},
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception:
            return False

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


# Global service instance
elevenlabs_tts_service = ElevenLabsTTSService() 