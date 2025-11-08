"""TTS (Text-to-Speech) API routes."""

from typing import Optional, List, Dict
import base64

from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import StreamingResponse
import time
from pydantic import BaseModel

from ...core.config import settings
from ...core.logging import get_logger
from ...services.tts_elevenlabs import elevenlabs_tts_service
from ...services.tts_openai import openai_tts_service
from ...services.tts_openai_realtime import openai_realtime_tts_service
from ...services.metrics import metrics_service

logger = get_logger(__name__)
router = APIRouter()


class TTSRequest(BaseModel):
    """Request for TTS synthesis."""
    text: str
    lang: str
    voice_hint: Optional[str] = None
    sender_gender: Optional[str] = None
    sender_id: Optional[str] = None


class TTSResponse(BaseModel):
    """Response from TTS synthesis."""
    audio_base64: str
    content_type: str
    provider: str
    voice_used: Optional[str] = None
    needs_browser_fallback: bool = False
    # Echo original text so clients can attach/display transcription with audio
    original_text: Optional[str] = None


@router.get("/voices")
async def get_available_voices() -> Dict[str, List[Dict]]:
    """
    Get list of available voices from the current TTS provider.
    
    Returns:
        Dictionary with provider name and available voices
    """
    try:
        if settings.tts_provider == "elevenlabs":
            voices = await elevenlabs_tts_service.get_voices_list()
            return {
                "provider": "elevenlabs",
                "voices": voices
            }
        else:
            return {
                "provider": settings.tts_provider,
                "voices": []
            }
            
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        return {
            "provider": settings.tts_provider,
            "voices": [],
            "error": str(e)
        }


@router.post("/speak", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest) -> TTSResponse:
    """
    Synthesize text to speech.
    
    Args:
        request: TTS request with text, language, and optional voice hint
        
    Returns:
        Audio data as base64 with metadata
    """
    try:
        logger.info(f"TTS synthesis: '{request.text[:50]}...' in {request.lang}")
        
        # Choose provider based on configuration
        voice_used = None
        if settings.tts_provider == "elevenlabs" and settings.elevenlabs_api_key:
            audio_bytes, content_type, needs_fallback, voice_used = await elevenlabs_tts_service.synthesize_elevenlabs(
                request.text, request.lang, request.voice_hint, request.sender_gender, request.sender_id
            )
            provider = "elevenlabs"
        elif settings.tts_provider == "elevenlabs" and not settings.elevenlabs_api_key:
            logger.warning("ElevenLabs TTS requested but API key not available, falling back to OpenAI")
            audio_bytes, content_type, needs_fallback, voice_used = await openai_tts_service.synthesize(
                request.text, request.lang, request.voice_hint, request.sender_gender, request.sender_id
            )
            provider = "openai"
        elif settings.tts_provider == "openai":
            audio_bytes, content_type, needs_fallback, voice_used = await openai_tts_service.synthesize(
                request.text, request.lang, request.voice_hint, request.sender_gender, request.sender_id
            )
            provider = "openai"
            
        elif settings.tts_provider == "browser":
            # Browser TTS should be handled client-side
            return TTSResponse(
                audio_base64="",
                content_type="audio/wav",
                provider="browser",
                needs_browser_fallback=True
            )
            
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unknown TTS provider: {settings.tts_provider}"
            )
        
        # If synthesis failed, signal browser fallback
        if needs_fallback or not audio_bytes:
            logger.warning(f"TTS synthesis failed, signaling browser fallback")
            return TTSResponse(
                audio_base64="",
                content_type=content_type,
                provider=provider,
                needs_browser_fallback=True,
                original_text=request.text
            )
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"TTS synthesis successful: {len(audio_bytes)} bytes ({provider})")
        
        return TTSResponse(
            audio_base64=audio_base64,
            content_type=content_type,
            provider=provider,
            voice_used=voice_used,
            needs_browser_fallback=False,
            original_text=request.text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        # Return browser fallback instead of error
        return TTSResponse(
            audio_base64="",
            content_type="audio/wav",
            provider=settings.tts_provider,
            needs_browser_fallback=True,
            original_text=request.text if hasattr(request, "text") else None
        ) 


@router.get("/stream")
async def tts_stream(
    text: str,
    lang: str,
    voice_hint: Optional[str] = None,
    fmt: Optional[str] = None,
    sr: Optional[int] = None,
):
    """Stream TTS audio via OpenAI Realtime (falls back to ElevenLabs if needed)."""
    if settings.use_openai_realtime and settings.openai_realtime_transport == "webrtc" and getattr(settings, "enforce_realtime_single_route", False):
        raise HTTPException(status_code=400, detail="Realtime transport is 'webrtc'. Use WebRTC client flow instead of HTTP stream.")
    media_type = "audio/mpeg"
    target_fmt = (fmt or settings.openai_tts_format or "mp3").lower()
    if target_fmt in ("wav", "pcm16"):
        media_type = "audio/wav"

    async def gen():
        start = time.time()
        total = 0
        metrics_service.record_tts_stream_start()
        try:
            async for chunk in openai_realtime_tts_service.stream_tts(
                text=text,
                lang=lang,
                voice_hint=voice_hint,
                audio_format=target_fmt,
                sample_rate_hz=sr or settings.openai_tts_sample_rate or 24000,
                connect_timeout=10.0,
                read_timeout=60.0,
                retries=1,
            ):
                total += len(chunk)
                yield chunk
        except Exception as e:
            logger.error(f"Streaming TTS failed: {e}")
            metrics_service.record_tts_stream_error()
        finally:
            elapsed = (time.time() - start) * 1000
            logger.info(f"TTS stream done: {total} bytes in {elapsed:.0f}ms")
            metrics_service.record_tts_stream_end(total, int(elapsed))

    return StreamingResponse(gen(), media_type=media_type)