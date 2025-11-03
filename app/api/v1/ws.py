"""WebSocket handler for real-time communication."""

import json
from typing import Dict, Set
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
                .where(Message.id == message.id)
                .options(selectinload(Message.sender))
            )
            fresh_message = result.scalar_one()
            
            # Get sender gender for TTS voice selection
            sender_gender = fresh_message.sender.gender if fresh_message.sender else None
            # If no gender is set, use preferred_voice as a hint for voice selection
            if not sender_gender and fresh_message.sender and fresh_message.sender.preferred_voice:
                # Map common voice names to gender hints
                voice_name = fresh_message.sender.preferred_voice.lower()
                if any(male_name in voice_name for male_name in ['clyde', 'david', 'james', 'john', 'michael', 'robert', 'william', 'thomas', 'charles', 'daniel']):
                    sender_gender = "male"
                elif any(female_name in voice_name for female_name in ['rachel', 'valentina', 'sarah', 'emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia']):
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
