"""OpenAI Realtime TTS streaming service (per-message session)."""

import asyncio
import base64
import json
from typing import AsyncIterator, Optional

import websockets

from ..core.config import settings
from ..core.logging import get_logger
from .tts_elevenlabs import elevenlabs_tts_service


logger = get_logger(__name__)


class OpenAIRealtimeTTSService:
    """Streams audio from OpenAI Realtime API for a single message."""

    def __init__(self) -> None:
        self._base_ws = "wss://api.openai.com/v1/realtime"

    async def stream_tts(
        self,
        text: str,
        lang: str,
        voice_hint: Optional[str] = None,
        audio_format: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        *,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        retries: int = 1,
    ) -> AsyncIterator[bytes]:
        """Yield audio chunks for the given text.

        Falls back to ElevenLabs if OpenAI Realtime fails before any bytes are sent.
        """

        if not settings.openai_api_key:
            logger.warning("OpenAI API key not configured; using ElevenLabs fallback for streaming")
            async for chunk in self._fallback_stream(text, lang, voice_hint):
                yield chunk
            return

        model = (settings.openai_realtime_model or "gpt-4o-realtime-preview-2024-12")
        voice = voice_hint or settings.openai_tts_voice or "alloy"
        fmt = (audio_format or settings.openai_tts_format or "mp3").lower()
        sr = int(sample_rate_hz or settings.openai_tts_sample_rate or 24000)

        logger.info(
            f"OpenAI Realtime TTS start model={model} transport={settings.openai_realtime_transport} fmt={fmt} sr={sr}"
        )

        url = f"{self._base_ws}?model={model}"
        headers = [
            ("Authorization", f"Bearer {settings.openai_api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]

        attempt = 0
        sent_any = False
        last_error: Optional[Exception] = None

        while attempt <= max(0, retries):
            attempt += 1
            try:
                # Establish WS
                async with websockets.connect(
                    url,
                    extra_headers=headers,
                    open_timeout=connect_timeout,
                    ping_interval=None,
                    max_size=None,
                ) as ws:
                    # Create a response with audio output
                    create_payload = {
                        "type": "response.create",
                        "response": {
                            "instructions": f"Speak the provided text in {lang} with a natural voice. provide us the translation if needed.",
                            "modalities": ["text","audio"],
                            "audio": {
                                "voice": voice,
                                "format": fmt,
                                "sample_rate_hz": sr,
                            },
                            "conversation": [
                                {"role": "user", "content": text}
                            ],
                        },
                    }
                    await ws.send(json.dumps(create_payload))

                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=read_timeout)
                        if isinstance(raw, (bytes, bytearray)):
                            # Some SDKs may stream audio as binary frames; pass through.
                            sent_any = True
                            yield bytes(raw)
                            continue

                        msg = json.loads(raw)
                      
                        mtype = msg.get("type")

                        if mtype == "response.output_audio.delta":
                            # Delta can be base64-encoded; support both raw bytes and base64.
                            delta = msg.get("delta")
                            if isinstance(delta, str):
                                try:
                                    chunk = base64.b64decode(delta)
                                except Exception:
                                    chunk = delta.encode("utf-8", errors="ignore")
                            elif isinstance(delta, (bytes, bytearray)):
                                chunk = bytes(delta)
                            else:
                                chunk = b""

                            if chunk:
                                sent_any = True
                                yield chunk

                        elif mtype in ("response.completed", "response.output_audio.done"):
                            return
                        elif mtype == "error":
                            raise RuntimeError(msg.get("error", "Realtime error"))

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"OpenAI Realtime TTS timeout (attempt {attempt}): {e}")
            except Exception as e:
                last_error = e
                logger.error(f"OpenAI Realtime TTS error (attempt {attempt}): {e}")

            # If we streamed any bytes, do not fallback; just stop.
            if sent_any:
                return

        # Fallback if no bytes ever sent
        logger.warning(f"Falling back to ElevenLabs after Realtime failure: {last_error}")
        async for chunk in self._fallback_stream(text, lang, voice_hint):
            yield chunk

    async def _fallback_stream(self, text: str, lang: str, voice_hint: Optional[str]) -> AsyncIterator[bytes]:
        try:
            audio_bytes, content_type, needs_fallback, _ = await elevenlabs_tts_service.synthesize_elevenlabs(
                text, lang, voice_hint, None, None
            )
            if audio_bytes:
                yield audio_bytes
            else:
                logger.warning("Fallback TTS returned no audio bytes")
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")


# Global instance
openai_realtime_tts_service = OpenAIRealtimeTTSService()
