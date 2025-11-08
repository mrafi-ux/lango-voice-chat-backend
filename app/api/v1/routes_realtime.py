"""Endpoints for OpenAI Realtime (WebRTC) support."""

import base64
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


class TranslateTTSRequest(BaseModel):
    """Request payload for translating text and retrieving TTS audio."""

    text: str = Field(..., description="Source text to translate and speak.")
    target_language: str = Field(..., description="Language to translate the speech into.")
    source_language: Optional[str] = Field(
        default="auto", description="Source language (auto-detect if omitted)."
    )
    voice: Optional[str] = Field(
        default=None, description="Preferred OpenAI voice. Defaults to configured voice."
    )
    audio_format: Optional[str] = Field(
        default="wav", description="Audio format: wav, mp3, or pcm16."
    )
    filename: Optional[str] = Field(
        default=None, description="Optional filename to save (extension inferred)."
    )
    include_base64_audio: bool = Field(
        default=True, description="Return base64 audio in response alongside saved file path."
    )


class TranslateTTSResponse(BaseModel):
    """Response payload for translate + TTS endpoint."""

    file_path: str
    content_type: str
    voice_used: str
    model: str
    translation_text: Optional[str] = None
    audio_base64: Optional[str] = None


@router.post("/webrtc/token")
async def create_webrtc_ephemeral_token(
    model: Optional[str] = None,
    voice: Optional[str] = None,
    format: Optional[str] = None,
    sample_rate: Optional[int] = None,
):
    print("Entered create_webrtc_ephemeral_token function")
    """Mint a short-lived ephemeral token for browser WebRTC connection to OpenAI Realtime.

    Returns a client_secret/token that the browser can use to establish a PeerConnection
    directly with OpenAI. Keep your permanent API key server-side only.
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    payload = {
        "model": model or settings.openai_realtime_model or "gpt-4o-realtime-preview-2024-12",
        "voice": voice or settings.openai_tts_voice or "alloy",
        "output_audio_format": "pcm16",
        # Do not set instructions here to avoid conflicting with client-provided session/response prompts
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                "https://api.openai.com/v1/realtime/sessions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if res.status_code != 200:
            logger.error(f"Failed to create ephemeral token: {res.status_code} {res.text}")
            raise HTTPException(status_code=502, detail="Failed to create ephemeral token")


        # print("Response is :::::: ",res)
        data = res.json()
        print(data)
        token_value = None
        client_secret = data.get("client_secret")
        if isinstance(client_secret, dict):
            token_value = client_secret.get("value") or client_secret.get("token")
        elif isinstance(client_secret, str):
            token_value = client_secret

        if not token_value and isinstance(data.get("token"), str):
            token_value = data["token"]

        if not token_value:
            logger.warning(f"Ephemeral token response missing client_secret value: {data}")
            raise HTTPException(status_code=502, detail="Ephemeral token response invalid")

        return {"token": token_value}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ephemeral token request timed out")
    except Exception:
        logger.exception("Ephemeral token error")
        raise HTTPException(status_code=500, detail="Unexpected error minting token")


# @router.post("/translate-tts", response_model=TranslateTTSResponse)
# async def translate_and_speak(request: TranslateTTSRequest) -> TranslateTTSResponse:
#     """Translate incoming text and return synthesized audio, persisting it locally."""
#     if not settings.openai_api_key:
#         raise HTTPException(status_code=500, detail="OpenAI API key not configured")

#     target_format = (request.audio_format or "wav").lower()
#     supported_formats = {"wav", "mp3", "pcm16"}
#     if target_format not in supported_formats:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported audio format '{target_format}'. Supported: {', '.join(sorted(supported_formats))}",
#         )

#     voice = request.voice or settings.openai_tts_voice or "alloy"
#     model = settings.openai_responses_tts_model or "gpt-4o-mini-tts"
#     source_lang = (request.source_language or "auto").strip() or "auto"
#     target_lang = request.target_language.strip() or "Spanish"

#     if source_lang.lower() == "auto":
#         source_phrase = "the detected source language"
#     else:
#         source_phrase = source_lang

#     instructions = (
#         f"Translate the provided text from {source_phrase} into {target_lang}. "
#         "Respond with speech only—do not include any written text or commentary."
#     )

#     payload = {
#         "model": model,
#         "modalities": ["audio","text"],
#         "instructions": instructions,
#         "input": [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "input_text",
#                         "text": request.text,
#                     }
#                 ],
#             }
#         ],
#         "audio": {
#             "voice": voice,
#             "format": target_format,
#         },
#     }

#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             response = await client.post(
#                 "https://api.openai.com/v1/responses",
#                 headers={
#                     "Authorization": f"Bearer {settings.openai_api_key}",
#                     "Content-Type": "application/json",
#                 },
#                 json=payload,
#             )
#             print("Response is :::::: ",response)
#     except httpx.RequestError as exc:
#         logger.error(f"HTTP error while requesting TTS audio: {exc}")
#         raise HTTPException(status_code=502, detail="Failed to reach OpenAI responses endpoint") from exc

#     if response.status_code != 200:
#         detail = response.text
#         logger.error(f"OpenAI responses API error {response.status_code}: {detail}")
#         raise HTTPException(status_code=502, detail="OpenAI responses API returned an error")

#     data = response.json()

#     def _find_audio(blob):
#         if isinstance(blob, dict):
#             audio_block = blob.get("audio")
#             if isinstance(audio_block, dict):
#                 data_val = audio_block.get("data") or audio_block.get("b64_json")
#                 if data_val:
#                     return (
#                         data_val,
#                         audio_block.get("format") or target_format,
#                         audio_block.get("voice") or voice,
#                         audio_block.get("transcript") or blob.get("transcript"),
#                     )
#             if isinstance(blob.get("data"), str) and blob.get("type") in {"output_audio", "audio"}:
#                 return (
#                     blob["data"],
#                     blob.get("format") or target_format,
#                     blob.get("voice") or voice,
#                     blob.get("transcript"),
#                 )
#             for value in blob.values():
#                 found = _find_audio(value)
#                 if found:
#                     return found
#         elif isinstance(blob, list):
#             for item in blob:
#                 found = _find_audio(item)
#                 if found:
#                     return found
#         return None

#     def _find_text(blob):
#         if isinstance(blob, dict):
#             if blob.get("type") in {"output_text", "text"} and isinstance(blob.get("text"), str):
#                 text_val = blob["text"].strip()
#                 if text_val:
#                     return text_val
#             if isinstance(blob.get("content"), str) and blob.get("content").strip():
#                 return blob["content"].strip()
#             for value in blob.values():
#                 found_text = _find_text(value)
#                 if found_text:
#                     return found_text
#         elif isinstance(blob, list):
#             for item in blob:
#                 found_text = _find_text(item)
#                 if found_text:
#                     return found_text
#         return None

#     audio_info = _find_audio(data)
#     if not audio_info:
#         logger.error(f"No audio data found in OpenAI response: {data}")
#         raise HTTPException(status_code=502, detail="OpenAI response missing audio content")

#     audio_b64, detected_format, detected_voice, transcript_text = audio_info
#     detected_format = (detected_format or target_format).lower()
#     if detected_format not in supported_formats:
#         detected_format = target_format

#     translated_text = _find_text(data)

#     output_dir = Path(settings.audio_output_dir).expanduser()
#     output_dir.mkdir(parents=True, exist_ok=True)

#     if request.filename:
#         filename = request.filename
#     else:
#         timestamp = int(time.time() * 1000)
#         filename = f"translation_{timestamp}.{detected_format}"

#     file_path = output_dir / filename
#     if not file_path.suffix:
#         file_path = file_path.with_suffix(f".{detected_format}")

#     try:
#         audio_bytes = base64.b64decode(audio_b64)
#     except Exception as exc:
#         logger.error(f"Failed to decode audio payload: {exc}")
#         raise HTTPException(status_code=500, detail="Could not decode audio data") from exc

#     try:
#         file_path.write_bytes(audio_bytes)
#     except Exception as exc:
#         logger.error(f"Failed to write audio file to disk: {exc}")
#         raise HTTPException(status_code=500, detail="Failed to save audio output") from exc

#     if detected_format == "mp3":
#         content_type = "audio/mpeg"
#     elif detected_format == "wav":
#         content_type = "audio/wav"
#     else:
#         content_type = "audio/pcm"

#     return TranslateTTSResponse(
#         file_path=str(file_path),
#         content_type=content_type,
#         voice_used=detected_voice or voice,
#         model=model,
#         translation_text=translated_text or transcript_text,
#         audio_base64=audio_b64 if request.include_base64_audio else None,
#     )


@router.get("/health")
async def openai_realtime_health():
    """Check OpenAI API reachability and auth/quota status with current key."""
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
        if res.status_code == 200:
            return {"status": "ok"}
        data = res.json() if res.content else {}
        return {
            "status": "error",
            "http_status": res.status_code,
            "code": (data.get("error") or {}).get("code"),
            "message": (data.get("error") or {}).get("message"),
        }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="OpenAI health check timed out")
    except Exception as e:
        logger.error(f"OpenAI health check error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI health check failed")
