"""WebSocket handler for real-time communication."""

import json
import base64
import asyncio
from typing import Dict, Set, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from ...db.session import AsyncSessionLocal
from ...db.crud import user_crud, conversation_crud, message_crud
from ...db.schemas import (
    WSJoinMessage, WSVoiceNoteMessage, WSMessageResponse, 
    WSPresenceResponse, WSErrorResponse, MessageCreate, MessageResponse
)
from ...db.models import MessageStatus, Message
from ...services.translate_libre import translate_service as libre_translate_service
from ...services.translate_openai import openai_translation_service
from ...services.metrics import metrics_service
from ...services.tts_openai_realtime import openai_realtime_tts_service
from ...workers.persist import schedule_background_task, persistence_worker
from ...core.logging import get_logger
from ...core.config import settings

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, str] = {}  # user_id -> connection_id
    
    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str) -> None:
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remove user mapping
        user_id_to_remove = None
        for user_id, conn_id in self.user_connections.items():
            if conn_id == connection_id:
                user_id_to_remove = user_id
                break
        
        if user_id_to_remove:
            del self.user_connections[user_id_to_remove]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_user(self, user_id: str, message: dict) -> bool:
        """Send message to specific user."""
        connection_id = self.user_connections.get(user_id)
        
        if not connection_id or connection_id not in self.active_connections:
            logger.warning(f"No active connection for user {user_id}")
            return False
        
        websocket = self.active_connections[connection_id]
        
        try:
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to user {user_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def broadcast_presence(self) -> None:
        """Broadcast online users to all connections."""
        online_users = list(self.user_connections.keys())
        presence_msg = WSPresenceResponse(
            type="presence",
            online_user_ids=online_users
        ).model_dump()
        
        for connection_id in list(self.active_connections.keys()):
            websocket = self.active_connections.get(connection_id)
            if websocket:
                try:
                    await websocket.send_text(json.dumps(presence_msg))
                except Exception as e:
                    logger.error(f"Failed to send presence to {connection_id}: {e}")
                    self.disconnect(connection_id)
    
    def register_user(self, user_id: str, connection_id: str) -> None:
        """Register user with connection."""
        self.user_connections[user_id] = connection_id
        logger.info(f"User {user_id} registered with connection {connection_id}")


# Global connection manager
manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket, connection_id: str) -> None:
    """Handle WebSocket connection."""
    await manager.connect(websocket, connection_id)
    current_user_id = None
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message_type = message_data.get("type")
                if message_type == "join":
                    await handle_join_message(message_data, connection_id)
                    current_user_id = message_data.get("user_id")
                    await manager.broadcast_presence()
                    
                elif message_type == "voice_note":
                    logger.info(f"Processing voice note message from user {current_user_id}")
                    await handle_voice_note_message(message_data)
                    logger.info(f"Voice note message processing completed for user {current_user_id}")
                elif message_type == "tts_stream_start":
                    # Enforce single realtime route if configured for WebRTC
                    if settings.use_openai_realtime and settings.openai_realtime_transport == "webrtc" and getattr(settings, "enforce_realtime_single_route", False):
                        await websocket.send_text(json.dumps({"type": "error", "message": "Realtime transport is 'webrtc'. Use WebRTC client flow."}))
                    else:
                        await handle_tts_stream_start(websocket, message_data)
                    
                elif message_type == "realtime_start":
                    await handle_realtime_start(websocket, connection_id, message_data)
                elif message_type == "realtime_audio_chunk":
                    await handle_realtime_audio_chunk(connection_id, message_data)
                elif message_type == "realtime_audio_end":
                    await handle_realtime_audio_end(websocket, connection_id)
                elif message_type == "realtime_translation_final":
                    await handle_realtime_translation_final(message_data)
                else:
                    error_msg = WSErrorResponse(
                        type="error",
                        message=f"Unknown message type: {message_type}"
                    ).model_dump()
                    await websocket.send_text(json.dumps(error_msg))
                    
            except json.JSONDecodeError:
                error_msg = WSErrorResponse(
                    type="error",
                    message="Invalid JSON format"
                ).model_dump()
                await websocket.send_text(json.dumps(error_msg))
                
            except ValidationError as e:
                error_msg = WSErrorResponse(
                    type="error",
                    message=f"Invalid message format: {str(e)}"
                ).model_dump()
                await websocket.send_text(json.dumps(error_msg))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        manager.disconnect(connection_id)
        if current_user_id:
            await manager.broadcast_presence()


async def handle_join_message(message_data: dict, connection_id: str) -> None:
    """Handle user join message."""
    try:
        join_msg = WSJoinMessage.model_validate(message_data)
        
        # Verify user exists
        async with AsyncSessionLocal() as session:
            user = await user_crud.get_by_id(session, join_msg.user_id)
            if not user:
                raise ValueError(f"User not found: {join_msg.user_id}")
        
        manager.register_user(join_msg.user_id, connection_id)
        logger.info(f"User {join_msg.user_id} joined")
        
    except Exception as e:
        logger.error(f"Failed to handle join message: {e}")
        raise


async def handle_voice_note_message(message_data: dict) -> None:
    """Handle voice note message with translation and forwarding."""
    try:
        voice_note = WSVoiceNoteMessage.model_validate(message_data)
        
        async with AsyncSessionLocal() as session:
            # Verify conversation exists
            conversation = await conversation_crud.get_by_id(session, voice_note.conversation_id)
            if not conversation:
                raise ValueError(f"Conversation not found: {voice_note.conversation_id}")
            
            # Determine recipient
            recipient_id = None
            if conversation.user_a_id == voice_note.sender_id:
                recipient_id = conversation.user_b_id
            elif conversation.user_b_id == voice_note.sender_id:
                recipient_id = conversation.user_a_id
            else:
                raise ValueError("Sender not part of conversation")
            
            # Create message record (without translation yet)
            message_create = MessageCreate(
                conversation_id=voice_note.conversation_id,
                sender_id=voice_note.sender_id,
                source_lang=voice_note.source_lang,
                target_lang=voice_note.target_lang,
                text_source=voice_note.text_source
            )
            
            message = await message_crud.create(session, message_create)
            message_id = message.id
            
            # Start TTFA tracking
            metrics_service.start_ttfa_tracking(
                message.id,
                voice_note.sender_id,
                recipient_id,
                voice_note.source_lang,
                voice_note.target_lang,
                voice_note.client_sent_at
            )
        
        # Translate text (outside DB transaction for speed)
        # Translate text using configured provider
        if settings.translation_provider_effective == "openai":
            translated_text = await openai_translation_service.translate(
                voice_note.text_source,
                voice_note.source_lang,
                voice_note.target_lang
            )
        else:
            translated_text = await libre_translate_service.translate(
                voice_note.text_source,
                voice_note.source_lang,
                voice_note.target_lang
            )
        
        # Record translation completion
        metrics_service.record_translation_completed(message.id)
        
        # Update message with translation in background
        schedule_background_task(
            persistence_worker.persist_message_translation(message.id, translated_text)
        )
        
        # Load message with sender info for response (use fresh query to avoid session issues)
        async with AsyncSessionLocal() as session:
            # Query the message fresh with sender relationship
            from sqlalchemy.orm import selectinload
            from sqlalchemy import select
            result = await session.execute(
                select(Message)
                .where(Message.id == message_id)
                .options(selectinload(Message.sender))
            )
            fresh_message = result.scalar_one()

        # Get sender gender for TTS voice selection
        sender_gender = fresh_message.sender.gender if fresh_message.sender else None
        # If no gender is set, use preferred_voice as a hint for voice selection
        if not sender_gender and fresh_message.sender and fresh_message.sender.preferred_voice:
            voice_name = fresh_message.sender.preferred_voice.lower()
            if any(male_name in voice_name for male_name in ["clyde", "david", "james", "john", "michael", "robert", "william", "thomas", "charles", "daniel"]):
                sender_gender = "male"
            elif any(female_name in voice_name for female_name in ["rachel", "valentina", "sarah", "emma", "olivia", "ava", "isabella", "sophia", "charlotte", "mia"]):
                sender_gender = "female"

        # Create recipient response with translated text
        fresh_message.text_translated = translated_text
        recipient_message_response = MessageResponse.model_validate(fresh_message)

        # Create sender response with original text (no translation)
        fresh_message.text_translated = None  # Sender sees original text
        sender_message_response = MessageResponse.model_validate(fresh_message)
        
        # Send to recipient with translated text and TTS
        play_now = {
            "lang": voice_note.target_lang,
            "text": translated_text,
            "sender_gender": sender_gender,
            "sender_id": voice_note.sender_id,
            # Attach original transcription
            "original_text": voice_note.text_source
        }
        
        recipient_ws_response = WSMessageResponse(
            type="message",
            message=recipient_message_response,
            play_now=play_now
        ).model_dump(mode='json')
        
        # Send to sender with original text (no TTS)
        sender_ws_response = WSMessageResponse(
            type="message",
            message=sender_message_response,
            play_now=None  # Sender doesn't need TTS playback
        ).model_dump(mode='json')
        
        # Record WebSocket send
        metrics_service.record_ws_sent(message.id)
        
        # Send to recipient (with translated text and TTS)
        sent_to_recipient = await manager.send_to_user(recipient_id, recipient_ws_response)
        
        # Send to sender (with original text, no TTS)
        sent_to_sender = await manager.send_to_user(voice_note.sender_id, sender_ws_response)
        
        if sent_to_recipient:
            # Update status to delivered in background
            schedule_background_task(
                persistence_worker.update_message_status(
                    message.id, 
                    MessageStatus.DELIVERED
                )
            )
            logger.info(f"Voice note delivered to recipient: {message.id}")
        else:
            # Recipient offline: keep as SENT for later delivery
            schedule_background_task(
                persistence_worker.update_message_status(
                    message.id, 
                    MessageStatus.SENT
                )
            )
            logger.warning(f"Recipient offline; queued voice note for later: {message.id}")
            
        if sent_to_sender:
            logger.info(f"Voice note confirmed to sender: {message.id}")
        else:
            logger.warning(f"Failed to confirm voice note to sender: {message.id}")
        
    except Exception as e:
        logger.error(f"Failed to handle voice note: {e}")
        # Could send error back to sender here


async def handle_tts_stream_start(websocket: WebSocket, message_data: dict) -> None:
    """Stream TTS audio to the requesting websocket as JSON chunks.
    Message schema example:
    {"type":"tts_stream_start","text":"hello","lang":"en","voice_hint":"alloy","fmt":"mp3","sr":24000}
    """
    try:
        text = message_data.get("text") or ""
        lang = message_data.get("lang") or "en"
        voice_hint = message_data.get("voice_hint")
        fmt = (message_data.get("fmt") or settings.openai_tts_format or "mp3").lower()
        sr = int(message_data.get("sr") or settings.openai_tts_sample_rate or 24000)

        await websocket.send_text(json.dumps({"type": "tts_stream_start_ack"}))
        metrics_service.record_tts_stream_start()

        bytes_out = 0
        import time as _time
        _start = _time.time()
        async for chunk in openai_realtime_tts_service.stream_tts(
            text=text,
            lang=lang,
            voice_hint=voice_hint,
            audio_format=fmt,
            sample_rate_hz=sr,
            connect_timeout=10.0,
            read_timeout=60.0,
            retries=1,
        ):
            if not chunk:
                continue
            bytes_out += len(chunk)
            await websocket.send_text(
                json.dumps({
                    "type": "tts_stream_chunk",
                    "data": base64.b64encode(chunk).decode("utf-8"),
                    "fmt": fmt
                })
            )

        elapsed = int((_time.time() - _start) * 1000)
        metrics_service.record_tts_stream_end(bytes_out, elapsed)
        await websocket.send_text(json.dumps({"type": "tts_stream_end", "bytes": bytes_out, "elapsed_ms": elapsed}))
    except Exception as e:
        metrics_service.record_tts_stream_error()
        await websocket.send_text(json.dumps({"type": "tts_stream_error", "message": str(e)}))


# --- OpenAI Realtime unified pipeline (STT -> translate -> TTS) ---

_rt_sessions: Dict[str, dict] = {}


async def handle_realtime_start(websocket: WebSocket, connection_id: str, message_data: dict) -> None:
    """Initialize a Realtime session that will accept audio chunks and stream outputs.
    Client should follow with multiple 'realtime_audio_chunk' messages and then 'realtime_audio_end'.
    message_data: {type, target_lang, source_lang?, voice?, fmt?, sr?}
    """
    try:
        target_lang = (message_data.get("target_lang") or "en").strip()
        source_lang = (message_data.get("source_lang") or "auto").strip()
        voice_hint = message_data.get("voice") or settings.openai_tts_voice or "alloy"
        fmt = (message_data.get("fmt") or settings.openai_tts_format or "mp3").lower()
        sr = int(message_data.get("sr") or settings.openai_tts_sample_rate or 24000)

        # Open Realtime WS upstream
        import websockets
        model = (settings.openai_realtime_model or "gpt-4o-realtime-preview-2024-12")
        logger.info(f"Realtime session start model={model} transport={settings.openai_realtime_transport}")
        url = f"wss://api.openai.com/v1/realtime?model={model}"
        headers = (
            ("Authorization", f"Bearer {settings.openai_api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        )
        ws = await websockets.connect(url, extra_headers=list(headers), ping_interval=None, max_size=None)

        # Store session state
        _rt_sessions[connection_id] = {
            "ws": ws,
            "voice": voice_hint,
            "fmt": fmt,
            "sr": sr,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "input_open": True,
            "forwarder": None,
        }

        await websocket.send_text(json.dumps({"type": "realtime_start_ack"}))

        # Start background forwarder that reads events and forwards to client
        async def _forward_events():
            try:
                while True:
                    raw = await _rt_sessions[connection_id]["ws"].recv()
                    if isinstance(raw, (bytes, bytearray)):
                        # Some servers may emit binary audio directly
                        await websocket.send_text(json.dumps({
                            "type": "tts_stream_chunk",
                            "data": base64.b64encode(raw).decode("utf-8"),
                            "fmt": fmt,
                        }))
                        continue
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    if mtype == "response.transcript.delta":
                        # Hypothetical event name for partial STT
                        await websocket.send_text(json.dumps({"type": "stt_delta", "text": msg.get("delta", "")}))
                    elif mtype == "response.translation.delta":
                        await websocket.send_text(json.dumps({"type": "translation_delta", "text": msg.get("delta", "")}))
                    elif mtype == "response.output_audio.delta":
                        delta = msg.get("delta")
                        if isinstance(delta, str):
                            chunk = base64.b64decode(delta)
                        elif isinstance(delta, (bytes, bytearray)):
                            chunk = bytes(delta)
                        else:
                            chunk = b""
                        if chunk:
                            await websocket.send_text(json.dumps({
                                "type": "tts_stream_chunk",
                                "data": base64.b64encode(chunk).decode("utf-8"),
                                "fmt": fmt,
                            }))
                    elif mtype in ("response.completed", "response.output_audio.done"):
                        await websocket.send_text(json.dumps({"type": "realtime_done"}))
                        break
                    elif mtype == "error":
                        await websocket.send_text(json.dumps({"type": "realtime_error", "message": str(msg.get("error"))}))
                        break
            except Exception as e:
                await websocket.send_text(json.dumps({"type": "realtime_error", "message": str(e)}))

        _rt_sessions[connection_id]["forwarder"] = asyncio.create_task(_forward_events())

    except Exception as e:
        await websocket.send_text(json.dumps({"type": "realtime_error", "message": str(e)}))


async def handle_realtime_audio_chunk(connection_id: str, message_data: dict) -> None:
    sess = _rt_sessions.get(connection_id)
    if not sess or not sess.get("input_open"):
        return
    try:
        audio_b64 = message_data.get("data")
        if not audio_b64:
            return
        # Send append buffer event upstream
        payload = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }
        await sess["ws"].send(json.dumps(payload))
    except Exception:
        pass


async def handle_realtime_audio_end(websocket: WebSocket, connection_id: str) -> None:
    sess = _rt_sessions.get(connection_id)
    if not sess:
        await websocket.send_text(json.dumps({"type": "realtime_error", "message": "no session"}))
        return
    if not sess.get("input_open"):
        return
    try:
        sess["input_open"] = False
        # Commit audio and request response with translation + TTS
        await sess["ws"].send(json.dumps({"type": "input_audio_buffer.commit"}))
        await sess["ws"].send(json.dumps({
            "type": "response.create",
            "response": {
                "instructions": (
                    "Transcribe the user's speech, translate it to "
                    f"{sess['target_lang']}, and speak the translated text."
                ),
                "modalities": ["audio","text"],
                "audio": {
                    "voice": sess["voice"],
                    "format": sess["fmt"],
                    "sample_rate_hz": sess["sr"],
                },
            },
        }))
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "realtime_error", "message": str(e)}))


async def handle_realtime_translation_final(message_data: dict) -> None:
    """On final translated text from sender, stream TTS to the recipient in real time.
    Expected message_data fields:
      - conversation_id: str
      - sender_id: str
      - target_lang: str
      - text: str (final translated text)
      - text_source?: str (original transcript)
      - source_lang?: str
      - voice_hint?: str
      - fmt?: str, sr?: int
    """
    try:
        conversation_id = message_data.get("conversation_id")
        sender_id = message_data.get("sender_id")
        target_lang = message_data.get("target_lang") or "en"
        source_lang = message_data.get("source_lang") or "auto"
        text = message_data.get("text") or ""
        text_source = message_data.get("text_source") or message_data.get("transcript") or ""
        voice_hint = message_data.get("voice_hint")
        fmt = (message_data.get("fmt") or settings.openai_tts_format or "mp3").lower()
        sr = int(message_data.get("sr") or settings.openai_tts_sample_rate or 24000)

        if not conversation_id or not sender_id or not text or not text_source:
            logger.warning("realtime_translation_final missing required fields")
            return

        recipient_id: Optional[str] = None
        message_id: Optional[str] = None

        # Find recipient from conversation
        async with AsyncSessionLocal() as session:
            conversation = await conversation_crud.get_by_id(session, conversation_id)
            if not conversation:
                logger.warning(f"Conversation not found: {conversation_id}")
                return
            if conversation.user_a_id == sender_id:
                recipient_id = conversation.user_b_id
            elif conversation.user_b_id == sender_id:
                recipient_id = conversation.user_a_id
            else:
                logger.warning("Sender not part of conversation")
                return

            message_create = MessageCreate(
                conversation_id=conversation_id,
                sender_id=sender_id,
                source_lang=source_lang,
                target_lang=target_lang,
                text_source=text_source,
            )

            message = await message_crud.create(session, message_create)
            message_id = message.id

            metrics_service.start_ttfa_tracking(
                message.id,
                sender_id,
                recipient_id,
                source_lang,
                target_lang,
                message_data.get("client_sent_at"),
            )

        if not recipient_id or not message_id:
            logger.warning("Realtime translation missing recipient_id or message_id")
            return

        # Record translation complete and persist translation
        metrics_service.record_translation_completed(message_id)
        schedule_background_task(
            persistence_worker.persist_message_translation(message_id, text)
        )

        # Load message with sender relationship
        async with AsyncSessionLocal() as session:
            from sqlalchemy.orm import selectinload
            from sqlalchemy import select

            result = await session.execute(
                select(Message)
                .where(Message.id == message_id)
                .options(selectinload(Message.sender))
            )
            fresh_message = result.scalar_one()

        sender_gender = fresh_message.sender.gender if fresh_message.sender else None
        fresh_message.text_translated = text

        recipient_message_response = MessageResponse.model_validate(fresh_message)
        recipient_ws_response = WSMessageResponse(
            type="message",
            message=recipient_message_response,
            play_now={
                "lang": target_lang,
                "text": text,
                "stream_realtime": True,
                "fmt": fmt,
                "sr": sr,
                "sender_gender": sender_gender,
                "sender_id": sender_id,
                "original_text": text_source,
            },
        ).model_dump(mode="json")

        fresh_message.text_translated = None
        sender_message_response = MessageResponse.model_validate(fresh_message)
        sender_ws_response = WSMessageResponse(
            type="message",
            message=sender_message_response,
            play_now=None,
        ).model_dump(mode="json")

        metrics_service.record_ws_sent(message_id)

        sent_to_recipient = await manager.send_to_user(recipient_id, recipient_ws_response)
        sent_to_sender = await manager.send_to_user(sender_id, sender_ws_response)

        if sent_to_recipient:
            schedule_background_task(
                persistence_worker.update_message_status(message_id, MessageStatus.DELIVERED)
            )
        else:
            schedule_background_task(
                persistence_worker.update_message_status(message_id, MessageStatus.SENT)
            )
            return

        if not sent_to_sender:
            logger.warning(f"Failed to confirm realtime message to sender: {message_id}")

        # Notify recipient to start playback
        await manager.send_to_user(recipient_id, {
            "type": "tts_stream_start",
            "message_id": message_id,
            "fmt": fmt,
            "sr": sr,
        })

        bytes_out = 0
        async for chunk in openai_realtime_tts_service.stream_tts(
            text=text,
            lang=target_lang,
            voice_hint=voice_hint,
            audio_format=fmt,
            sample_rate_hz=sr,
            connect_timeout=10.0,
            read_timeout=60.0,
            retries=1,
        ):
            if not chunk:
                continue
            bytes_out += len(chunk)
            await manager.send_to_user(recipient_id, {
                "type": "tts_stream_chunk",
                "message_id": message_id,
                "data": base64.b64encode(chunk).decode("utf-8"),
                "fmt": fmt,
            })

        await manager.send_to_user(recipient_id, {
            "type": "tts_stream_end",
            "message_id": message_id,
            "bytes": bytes_out,
            "sr": sr,
        })

    except Exception as e:
        logger.error(f"Failed realtime_translation_final streaming: {e}")
