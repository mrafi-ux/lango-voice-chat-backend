"""API routes for system capabilities."""

from typing import Dict, Any
from fastapi import APIRouter

from ...services.stt_whisper import whisper_stt_service
from ...services.stt_mock import mock_stt_service
from ...services.tts_elevenlabs import elevenlabs_tts_service
from ...services.translate_libre import translate_service as libre_translate_service
from ...services.translate_openai import openai_translation_service
from ...core.config import settings

router = APIRouter()


@router.get("/languages")
async def get_language_capabilities() -> Dict[str, Any]:
    """Get supported languages for all services."""
    try:
        # Get STT languages based on configured provider
        try:
            if settings.stt_provider == "elevenlabs":
                stt_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"]  # ElevenLabs supported languages
            elif settings.stt_provider == "openai":
                stt_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi", "tr", "pl", "nl"]  # OpenAI Whisper supported languages
            else:  # whisper
                stt_languages = whisper_stt_service.get_supported_languages()
        except Exception:
            stt_languages = mock_stt_service.get_supported_languages()
        
        return {
            "stt": {
                "provider": settings.stt_provider,
                "languages": stt_languages
            },
            "translation": {
                # We use libre list for capabilities display regardless of provider
                "provider": settings.translation_provider_effective,
                "languages": await libre_translate_service.get_supported_languages()
            },
            "tts": {
                "provider": settings.tts_provider,
                "languages": elevenlabs_tts_service.get_supported_languages()
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "stt": {"provider": "mock", "languages": mock_stt_service.get_supported_languages()},
            "translation": {"provider": "mock", "languages": await libre_translate_service.get_supported_languages()},
            "tts": {"provider": "mock", "languages": ["en", "es"]}
        }


@router.get("/providers")
async def get_provider_info() -> Dict[str, Any]:
    """Get information about service providers."""
    return {
        "stt": {
            "primary": settings.stt_provider,
            "fallback": "mock"
        },
        "translation": {
            "primary": settings.translation_provider_effective,
            "fallback": "libre"
        },
        "tts": {
            "primary": settings.tts_provider,
            "fallback": "mock"
        }
    }


@router.get("/models")
async def get_model_info() -> Dict[str, Any]:
    """Expose concrete model identifiers used by the backend.

    The frontend can call this to display which models are active.
    """
    # STT model by provider
    stt_model = None
    if settings.stt_provider == "openai":
        stt_model = settings.openai_stt_model
    elif settings.stt_provider == "whisper":
        stt_model = settings.whisper_model
    elif settings.stt_provider == "elevenlabs":
        stt_model = settings.elevenlabs_stt_model

    # TTS model by provider
    tts_model = None
    tts_voice = None
    if settings.tts_provider == "openai":
        tts_model = settings.openai_tts_model
        tts_voice = settings.openai_tts_voice
    elif settings.tts_provider == "elevenlabs":
        # We use the multilingual v2 model in ElevenLabs client
        tts_model = "eleven_multilingual_v2"
    elif settings.tts_provider == "browser":
        tts_model = "web_speech_api"

    # Translation model/provider info (dynamic)
    if settings.translation_provider_effective == "openai":
        translation_provider = "openai"
        translation_model = settings.openai_translate_model
    else:
        translation_provider = "deep-translator:google"
        translation_model = "google-translate-free"

    return {
        "stt": {
            "provider": settings.stt_provider,
            "model": stt_model,
        },
        "tts": {
            "provider": settings.tts_provider,
            "model": tts_model,
            "voice": tts_voice,
        },
        "translation": {
            "provider": translation_provider,
            "model": translation_model,
        },
    }


@router.post("/translate")
async def translate_text(request: Dict[str, Any]) -> Dict[str, Any]:
    """Translate text using the translation service."""
    try:
        text = request.get("text", "")
        source_lang = request.get("source", "en")
        target_lang = request.get("target", "es")
        
        if not text:
            return {"error": "Text is required"}
        
        # Choose translation provider dynamically
        if settings.translation_provider_effective == "openai":
            translated_text = await openai_translation_service.translate(text, source_lang, target_lang)
        else:
            translated_text = await libre_translate_service.translate(text, source_lang, target_lang)
        
        return {
            "translatedText": translated_text,
            "source": source_lang,
            "target": target_lang
        }
    except Exception as e:
        return {"error": str(e)} 


@router.get("/elevenlabs/validate")
async def validate_elevenlabs() -> Dict[str, Any]:
    """Validate ElevenLabs configuration and API key.

    Returns booleans for key presence and API availability to help debug 401 issues.
    """
    try:
        key_present = bool(getattr(settings, "elevenlabs_api_key", None))
        available = await elevenlabs_tts_service.check_elevenlabs_available()
        return {
            "key_present": key_present,
            "tts_available": available,
            "stt_model": getattr(settings, "elevenlabs_stt_model", None),
            "notes": "tts_available checks /v1/voices with provided API key"
        }
    except Exception as e:
        return {
            "key_present": bool(getattr(settings, "elevenlabs_api_key", None)),
            "tts_available": False,
            "error": str(e)
        }
