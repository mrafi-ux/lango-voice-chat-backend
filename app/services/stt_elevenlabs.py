"""ElevenLabs STT service for speech recognition (Scribe v1)."""

import asyncio
from typing import Optional, Dict

import httpx

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class ElevenLabsSTTService:
    """Speech-to-Text service using ElevenLabs Scribe API."""

    def __init__(self) -> None:
        self.api_key = getattr(settings, 'elevenlabs_api_key', None)
        self.model_id = getattr(settings, 'elevenlabs_stt_model', 'scribe_v1')
        self.endpoint = 'https://api.elevenlabs.io/v1/speech-to-text'

    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Transcribe audio bytes to text using ElevenLabs.

        Returns dictionary: {"text": str, "language": str, "confidence": float? , "error": str?}
        """
        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY missing in configuration")
            return {"text": "", "language": language or "en", "error": "ElevenLabs API key not configured"}

        # Check minimum duration for ElevenLabs STT (approximately 0.5 seconds)
        # WebM/Opus at 16kHz typically needs ~6000 bytes for 0.5 seconds
        if len(audio_data) < 6000:
            logger.info(f"Audio too short for ElevenLabs STT ({len(audio_data)} bytes), skipping")
            return {"text": "", "language": language or "en", "error": "Audio too short", "no_speech": True}

        try:
            # Prepare multipart form
            headers = {
                'xi-api-key': self.api_key,
            }

            # Prepare form data; only include language hint if provided
            lang_hint = (language or '').strip()
            form_data = {
                'model_id': self.model_id,
            }
            if lang_hint:
                form_data['language_code'] = lang_hint[:2]

            # Use a single shared AsyncClient for the request
            async with httpx.AsyncClient(timeout=60) as client:
                # ElevenLabs STT works best with MP3/WAV, but we'll try WebM first since that's what frontend sends
                files = {'file': ('audio.webm', audio_data, 'audio/webm')}
                resp = await client.post(self.endpoint, headers=headers, data=form_data, files=files)
                
                # If WebM fails, try MP3 format
                if resp.status_code >= 400 and 'webm' in str(resp.text).lower():
                    logger.info("WebM format failed, trying MP3 format")
                    files = {'file': ('audio.mp3', audio_data, 'audio/mpeg')}
                    resp = await client.post(self.endpoint, headers=headers, data=form_data, files=files)

            if resp.status_code >= 400:
                logger.error(f"ElevenLabs STT error {resp.status_code}: {resp.text}")
                return {"text": "", "language": (lang_hint or "en")[:2], "error": f"HTTP {resp.status_code}", "no_speech": True}

            data = resp.json()

            # ElevenLabs returns various shapes; try common fields
            text = data.get('text') or data.get('transcript') or data.get('content') or ''
            detected_lang = data.get('language') or data.get('language_code') or data.get('detected_language') or (lang_hint[:2] if lang_hint else None)
            confidence = data.get('confidence') or None
            
            # Debug logging for ElevenLabs response
            logger.info(f"ElevenLabs STT response keys: {list(data.keys())}")
            logger.info(f"ElevenLabs STT raw response: {data}")

            # Check if we got empty text - this might indicate an issue
            if not text or not text.strip():
                logger.warning(f"ElevenLabs STT returned empty text. Response: {data}")
                return {"text": "", "language": (detected_lang or (lang_hint or "en")[:2]), "error": "Empty transcription", "no_speech": True}

            logger.info(f"ElevenLabs STT transcription: '{text[:100]}...' (lang: {detected_lang})")
            return {
                "text": (text or '').strip(),
                "language": (detected_lang or (lang_hint or "en")[:2]),
                "confidence": confidence,
                "no_speech": False
            }

        except Exception as e:
            logger.error(f"ElevenLabs STT failed: {e}")
            return {"text": "", "language": language or "en", "error": str(e)}


# Global instance
elevenlabs_stt_service = ElevenLabsSTTService()


