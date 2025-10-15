"""STT (Speech-to-Text) API routes."""

from typing import Dict, Optional, Any
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from pydantic import BaseModel

from ...services.stt_whisper import whisper_stt_service
from ...services.stt_elevenlabs import elevenlabs_stt_service
from ...services.stt_openai import openai_stt_service
from ...services.stt_mock import mock_stt_service
from ...core.logging import get_logger
from ...core.config import settings

logger = get_logger(__name__)

router = APIRouter()


class STTResponse(BaseModel):
    text: str
    language: str
    confidence: Optional[float] = None
    provider: str = "whisper"
    error: Optional[str] = None
    no_speech: bool = False


@router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None)
) -> STTResponse:
    """
    Transcribe audio file to text.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        language: Optional language hint (e.g., 'en', 'es')
        
    Returns:
        Transcription result with text and detected language
    """
    try:
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if audio.size and audio.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Max size: {max_size} bytes"
            )
        
        # Read audio data
        audio_data = await audio.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        logger.info(f"STT transcription request: {len(audio_data)} bytes, lang hint: {language}")
        
        # Choose provider
        result = None
        provider = settings.stt_provider or "whisper"
        fallback_enabled = getattr(settings, 'stt_fallback_enabled', True)
        logger.info(f"STT provider selected: {provider} (fallback_enabled={fallback_enabled})")
        logger.info(f"Processing STT request for audio size: {len(audio_data)} bytes")

        async def retry_auto_if_empty(current_provider: str, current_result: Dict[str, Any]):
            # Don't retry if there's an error or if we have text
            if current_result.get("error") or (current_result.get("text") or "").strip():
                return current_result
            logger.info(f"STT empty text on {current_provider}; retrying with auto language")
            if current_provider == "elevenlabs":
                return await elevenlabs_stt_service.transcribe_audio(audio_data, None)
            if current_provider == "openai":
                return await openai_stt_service.transcribe_audio(audio_data, None)
            return await whisper_stt_service.transcribe_audio(audio_data, None)

        try:
            if provider == "elevenlabs":
                result = await elevenlabs_stt_service.transcribe_audio(audio_data, language)
                logger.info(f"ElevenLabs STT result: {result}")
                result = await retry_auto_if_empty("elevenlabs", result)
                logger.info(f"ElevenLabs STT after retry: {result}")
                if fallback_enabled and ("error" in result and result["error"] or not (result.get("text") or "").strip()):
                    logger.warning(f"ElevenLabs STT failed/empty -> fallback to Whisper")
                    provider = "whisper"
                    result = await whisper_stt_service.transcribe_audio(audio_data, language)
                    result = await retry_auto_if_empty("whisper", result)
            elif provider == "openai":
                result = await openai_stt_service.transcribe_audio(audio_data, language)
                result = await retry_auto_if_empty("openai", result)
                if fallback_enabled and ("error" in result and result["error"] or not (result.get("text") or "").strip()):
                    logger.warning(f"OpenAI STT failed/empty -> fallback to ElevenLabs")
                    provider = "elevenlabs"
                    result = await elevenlabs_stt_service.transcribe_audio(audio_data, language)
                    result = await retry_auto_if_empty("elevenlabs", result)
            else:
                result = await whisper_stt_service.transcribe_audio(audio_data, language)
                result = await retry_auto_if_empty("whisper", result)
                if fallback_enabled and ("error" in result and result["error"] or not (result.get("text") or "").strip()):
                    logger.warning(f"Whisper STT failed/empty -> fallback to ElevenLabs")
                    provider = "elevenlabs"
                    result = await elevenlabs_stt_service.transcribe_audio(audio_data, language)
                    result = await retry_auto_if_empty("elevenlabs", result)

            # Final fallback to mock if still failing
            if ("error" in result and result["error"]) or not (result.get("text") or "").strip():
                if not fallback_enabled:
                    logger.error("STT returned error/empty and fallback disabled")
                    return STTResponse(text="", language=language or "en", provider=provider, error=result.get("error") or "No speech detected")
                logger.warning(f"Primary STT failed/empty, falling back to mock")
                provider = "mock"
                result = await mock_stt_service.transcribe_audio(audio_data, language)

        except Exception as stt_error:
            logger.warning(f"STT exception ({provider}): {stt_error}, falling back to mock")
            provider = "mock"
            result = await mock_stt_service.transcribe_audio(audio_data, language)
        
        # Check for final transcription errors
        if "error" in result and result["error"]:
            logger.error(f"STT transcription failed completely: {result['error']}")
            return STTResponse(
                text="",
                language=language or "en",
                provider=provider,
                error=result["error"],
                no_speech=bool(result.get("no_speech", False))
            )
        
        # Successful transcription
        logger.info(f"STT transcription successful ({provider}): '{result['text'][:100]}...' (lang: {result['language']})")
        logger.info(f"STT processing completed. Provider: {provider}, Text length: {len(result.get('text', ''))}")
        
        return STTResponse(
            text=result["text"],
            language=result["language"],
            confidence=result.get("confidence"),
            provider=provider,
            no_speech=bool(result.get("no_speech", False))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        
        # Last resort: try mock service
        try:
            logger.info("Attempting final fallback to mock STT")
            result = await mock_stt_service.transcribe_audio(b"", language)
            return STTResponse(
                text=result["text"],
                language=result["language"],
                confidence=result.get("confidence"),
                provider="mock",
                error="Whisper unavailable, using mock transcription"
            )
        except Exception as mock_error:
            logger.error(f"Mock STT also failed: {mock_error}")
            raise HTTPException(status_code=500, detail=f"All transcription services failed: {str(e)}")


@router.get("/languages")
async def get_supported_languages() -> Dict[str, Any]:
    """
    Get list of languages supported by the STT provider.
    
    Returns:
        Dictionary with provider name and supported languages
    """
    try:
        # Try to get Whisper languages first
        try:
            languages = whisper_stt_service.get_supported_languages()
            return {
                "provider": "whisper",
                "languages": languages,
                "count": len(languages)
            }
        except Exception as whisper_error:
            logger.warning(f"Whisper languages failed: {whisper_error}, using mock")
            languages = mock_stt_service.get_supported_languages()
            return {
                "provider": "mock",
                "languages": languages,
                "count": len(languages),
                "note": "Using mock STT service (Whisper unavailable)"
            }
            
    except Exception as e:
        logger.error(f"Failed to get STT languages: {e}")
        return {
            "provider": "error",
            "languages": [],
            "count": 0,
            "error": str(e)
        } 
