"""Optimized Simplified FastAPI application for voice chat translation."""

import uuid
import json
import base64
import asyncio
import time
from typing import Dict, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .core.config import settings
from .core.logging import get_logger, setup_logging
from .services.translate_libre import translate_service
from .services.tts_elevenlabs import ElevenLabsTTSService
from .services.tts_openai import OpenAITTSService
from .services.stt_elevenlabs import ElevenLabsSTTService
from .services.stt_whisper import WhisperSTTService
from .services.stt_openai import OpenAISTTService

# Setup logging first
setup_logging()
logger = get_logger(__name__)

# Test logging
logger.info("🚀 Optimized Simple Voice Chat Backend starting up...")

# Thread pool for CPU-intensive operations
def get_thread_pool():
    """Get or create thread pool executor"""
    if not hasattr(get_thread_pool, 'executor'):
        get_thread_pool.executor = ThreadPoolExecutor(max_workers=4)
    return get_thread_pool.executor

async def run_in_thread(func, *args, **kwargs):
    """Run function in thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(get_thread_pool(), func, *args, **kwargs)

# Background task manager
class BackgroundTaskManager:
    def __init__(self):
        self.tasks = set()
    
    def schedule_task(self, coro):
        """Schedule a background task"""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task
    
    async def cleanup(self):
        """Clean up all background tasks"""
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

# Global background task manager
background_manager = BackgroundTaskManager()

# Retry logic with exponential backoff
async def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)

# Pydantic models for WebSocket messages
class SimpleMessage(BaseModel):
    text: str = None
    audio_data: str = None  # base64 encoded audio
    source_lang: str
    target_lang: str
    sender_id: str

class SimpleResponse(BaseModel):
    original_text: str = None
    translated_text: str
    audio_url: Optional[str] = None
    message_id: str

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
    
    async def send_message(self, websocket: WebSocket, message: dict) -> None:
        """Send message to WebSocket client."""
        try:
            # Use thread pool for JSON serialization
            message_json = await run_in_thread(json.dumps, message)
            await websocket.send_text(message_json)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

# Global connection manager
manager = ConnectionManager()

# Dynamic service factories - these check config each time they're called
def get_stt_service():
    """Get STT service based on current configuration."""
    current_provider = settings.stt_provider
    logger.info(f"Getting STT service for provider: {current_provider}")
    
    if current_provider == "elevenlabs":
        return ElevenLabsSTTService()
    elif current_provider == "openai":
        return OpenAISTTService()
    elif current_provider == "whisper":
        return WhisperSTTService()
    else:
        logger.warning(f"Unknown STT provider: {current_provider}, falling back to whisper")
        return WhisperSTTService()

def get_tts_service():
    """Get TTS service based on current configuration."""
    current_provider = settings.tts_provider
    logger.info(f"Getting TTS service for provider: {current_provider}")
    
    if current_provider == "elevenlabs":
        return ElevenLabsTTSService()
    elif current_provider == "openai":
        return OpenAITTSService()
    else:
        logger.warning(f"Unknown TTS provider: {current_provider}, falling back to openai")
        return OpenAITTSService()

# Initialize fallback service (always available)
whisper_stt_service = WhisperSTTService()  # Fallback STT service

# Log service configuration
logger.info("=" * 60)
logger.info("SIMPLE VOICE CHAT CONFIGURATION")
logger.info("=" * 60)
logger.info(f"STT Provider: {settings.stt_provider}")
logger.info(f"TTS Provider: {settings.tts_provider}")
logger.info(f"Translation Provider: {settings.translation_provider}")
logger.info(f"STT Fallback Enabled: {settings.stt_fallback_enabled}")
logger.info(f"OpenAI API Key: {'SET' if settings.openai_api_key else 'NOT SET'}")
logger.info(f"ElevenLabs API Key: {'SET' if settings.elevenlabs_api_key else 'NOT SET'}")
logger.info("=" * 60)
logger.info("Services will be dynamically selected based on configuration")
logger.info("=" * 60)

# Create FastAPI app
app = FastAPI(
    title="Simple Voice Chat Translation",
    description="Simple voice chat with translation",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("🎉 Simple Voice Chat Backend is now running!")
    logger.info(f"📡 Server running on http://127.0.0.1:8000")
    logger.info(f"🔧 Debug config available at http://127.0.0.1:8000/debug/config")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "message": "Voice Chat App is running"}

@app.get("/debug/config")
async def debug_config():
    """Debug endpoint to show current configuration and active services."""
    # Get current services dynamically
    current_stt_service = get_stt_service()
    current_tts_service = get_tts_service()
    
    # Test TTS service
    tts_test_result = "Not tested"
    try:
        if hasattr(current_tts_service, 'client') and current_tts_service.client:
            tts_test_result = "OpenAI client initialized"
        elif hasattr(current_tts_service, '_http_client') and current_tts_service._http_client:
            tts_test_result = "ElevenLabs client initialized"
        else:
            tts_test_result = "No client initialized"
    except Exception as e:
        tts_test_result = f"Error checking TTS service: {str(e)}"
    
    return {
        "stt_provider": settings.stt_provider,
        "tts_provider": settings.tts_provider,
        "translation_provider": settings.translation_provider,
        "stt_fallback_enabled": settings.stt_fallback_enabled,
        "openai_api_key_set": bool(settings.openai_api_key),
        "elevenlabs_api_key_set": bool(settings.elevenlabs_api_key),
        "current_stt_service": type(current_stt_service).__name__,
        "current_tts_service": type(current_tts_service).__name__,
        "tts_service_status": tts_test_result,
        "openai_tts_voice": settings.openai_tts_voice,
        "openai_tts_model": settings.openai_tts_model,
        "openai_stt_model": settings.openai_stt_model,
        "elevenlabs_stt_model": settings.elevenlabs_stt_model,
        "note": "Services are dynamically selected based on current configuration"
    }

# Background task functions
async def log_processing_metrics(message_id: str, processing_time: float, success: bool):
    """Log processing metrics in background"""
    try:
        logger.info(f"Metrics - Message: {message_id}, Time: {processing_time:.2f}s, Success: {success}")
        # Add analytics logging here if needed
    except Exception as e:
        logger.error(f"Metrics logging failed: {e}")

async def cache_translation_result(original: str, translated: str, source: str, target: str):
    """Cache translation result in background"""
    try:
        # Add translation caching here if needed
        logger.debug(f"Cached translation: {source}->{target}")
    except Exception as e:
        logger.error(f"Translation caching failed: {e}")

# Safe service functions with retry logic
async def safe_stt_processing(audio_bytes: bytes, source_lang: str):
    """STT processing with retry logic"""
    return await retry_with_backoff(
        lambda: get_stt_service().transcribe_audio(audio_bytes, source_lang),
        max_retries=2
    )

async def safe_translation(text: str, source_lang: str, target_lang: str):
    """Translation with retry logic"""
    return await retry_with_backoff(
        lambda: translate_service.translate(text, source_lang, target_lang),
        max_retries=2
    )

async def safe_tts_processing(text: str, lang: str, voice_hint: str, sender_gender: str, sender_id: str):
    """TTS processing with retry logic"""
    tts_service = get_tts_service()
    
    # Call the appropriate method based on service type
    if hasattr(tts_service, 'synthesize_elevenlabs'):
        return await retry_with_backoff(
            lambda: tts_service.synthesize_elevenlabs(text, lang, voice_hint, sender_gender, sender_id),
            max_retries=2
        )
    elif hasattr(tts_service, 'synthesize'):
        return await retry_with_backoff(
            lambda: tts_service.synthesize(text, lang, voice_hint, sender_gender, sender_id),
            max_retries=2
        )
    else:
        raise AttributeError(f"TTS service {type(tts_service).__name__} has no synthesize method")

# Streaming voice message processing function
async def process_voice_message_streaming(message_data: dict, websocket: WebSocket) -> dict:
    """Process voice message with streaming and progressive updates"""
    start_time = time.time()
    message_id = str(uuid.uuid4())
    
    try:
        # Check if we have audio or text input
        has_audio = 'audio_data' in message_data and message_data['audio_data']
        has_text = 'text' in message_data and message_data['text']
        
        if not has_audio and not has_text:
            return {"type": "error", "message": "No audio or text data provided"}
        
        # Send immediate acknowledgment
        stage = "decoding_audio" if has_audio else "processing_text"
        await manager.send_message(websocket, {
            "type": "processing_started",
            "data": {
                "message_id": message_id,
                "status": "processing",
                "stage": stage
            }
        })
        
        # Process audio or text input
        if has_audio:
            # Decode audio in thread pool (non-blocking)
            audio_bytes = await run_in_thread(base64.b64decode, message_data['audio_data'])
            
            # Send progress update
            await manager.send_message(websocket, {
                "type": "processing_update",
                "data": {
                    "message_id": message_id,
                    "status": "processing",
                    "stage": "speech_recognition"
                }
            })
            
            # Start STT processing
            stt_task = asyncio.create_task(safe_stt_processing(audio_bytes, message_data['source_lang']))
            
            # Wait for STT result
            stt_result = await stt_task
            original_text = stt_result.get("text", "")
            
            if not original_text:
                return {"type": "error", "message": "Speech recognition failed"}
        else:
            # Use provided text directly
            original_text = message_data['text']
            
            # Send progress update
            await manager.send_message(websocket, {
                "type": "processing_update",
                "data": {
                    "message_id": message_id,
                    "status": "processing",
                    "stage": "text_processing"
                }
            })
        
        # Send STT result immediately
        await manager.send_message(websocket, {
            "type": "stt_result",
            "data": {
                "message_id": message_id,
                "original_text": original_text,
                "status": "stt_complete"
            }
        })
        
        # Send progress update for translation
        await manager.send_message(websocket, {
            "type": "processing_update",
            "data": {
                "message_id": message_id,
                "status": "processing",
                "stage": "translation"
            }
        })
        
        # Start translation
        translation_task = asyncio.create_task(safe_translation(
            original_text, 
            message_data['source_lang'], 
            message_data['target_lang']
        ))
        
        # Wait for translation result
        translated_text = await translation_task
        
        # Send translation result immediately
        await manager.send_message(websocket, {
            "type": "translation_result",
            "data": {
                "message_id": message_id,
                "translated_text": translated_text,
                "status": "translation_complete"
            }
        })
        
        # Send progress update for TTS
        await manager.send_message(websocket, {
            "type": "processing_update",
            "data": {
                "message_id": message_id,
                "status": "processing",
                "stage": "text_to_speech"
            }
        })
        
        # Start TTS processing
        logger.info(f"Starting TTS processing for text: '{translated_text[:50]}...' in language: {message_data['target_lang']}")
        tts_task = asyncio.create_task(safe_tts_processing(
            translated_text,
            message_data['target_lang'],
            None,  # voice_hint
            None,  # sender_gender
            message_data['sender_id']
        ))
        
        # Wait for TTS result
        audio_data, content_type, needs_fallback, voice_used = await tts_task
        
        logger.info(f"TTS result: audio_data={len(audio_data) if audio_data else 'None'}, content_type={content_type}, needs_fallback={needs_fallback}, voice_used={voice_used}")
        
        # Encode audio in thread pool (non-blocking)
        audio_url = None
        if audio_data and not needs_fallback:
            logger.info(f"Encoding audio data: {len(audio_data)} bytes")
            audio_data_base64 = await run_in_thread(
                lambda data: base64.b64encode(data).decode('utf-8'), 
                audio_data
            )
            audio_url = f"data:{content_type};base64,{audio_data_base64}"
            logger.info(f"Generated audio URL: {len(audio_url)} characters")
        else:
            logger.warning(f"TTS failed or needs fallback: audio_data={bool(audio_data)}, needs_fallback={needs_fallback}")
        
        # Send final complete result
        result = {
            "type": "translation_complete",
            "data": {
                "message_id": message_id,
                "original_text": original_text,
                "translated_text": translated_text,
                "audio_url": audio_url,
                "status": "complete"
            }
        }
        
        # Schedule background tasks
        processing_time = time.time() - start_time
        background_manager.schedule_task(
            log_processing_metrics(message_id, processing_time, True)
        )
        background_manager.schedule_task(
            cache_translation_result(original_text, translated_text, 
                                   message_data['source_lang'], 
                                   message_data['target_lang'])
        )
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        background_manager.schedule_task(
            log_processing_metrics(message_id, processing_time, False)
        )
        logger.error(f"Voice message processing failed: {e}")
        return {"type": "error", "message": f"Processing failed: {str(e)}"}

# Parallel processing function for maximum speed
async def process_voice_message_parallel(message_data: dict, websocket: WebSocket) -> dict:
    """Process voice message with maximum parallelization"""
    start_time = time.time()
    message_id = str(uuid.uuid4())
    
    try:
        # Check if we have audio or text input
        has_audio = 'audio_data' in message_data and message_data['audio_data']
        has_text = 'text' in message_data and message_data['text']
        
        if not has_audio and not has_text:
            return {"type": "error", "message": "No audio or text data provided"}
        
        # Send immediate acknowledgment
        stage = "decoding_audio" if has_audio else "processing_text"
        await manager.send_message(websocket, {
            "type": "processing_started",
            "data": {
                "message_id": message_id,
                "status": "processing",
                "stage": stage
            }
        })
        
        # Process audio or text input
        if has_audio:
            # Decode audio in thread pool (non-blocking)
            audio_bytes = await run_in_thread(base64.b64decode, message_data['audio_data'])
            
            # Send progress update
            await manager.send_message(websocket, {
                "type": "processing_update",
                "data": {
                    "message_id": message_id,
                    "status": "processing",
                    "stage": "speech_recognition"
                }
            })
            
            # Start STT processing
            stt_task = asyncio.create_task(safe_stt_processing(audio_bytes, message_data['source_lang']))
            
            # Wait for STT result
            stt_result = await stt_task
            original_text = stt_result.get("text", "")
            
            if not original_text:
                return {"type": "error", "message": "Speech recognition failed"}
        else:
            # Use provided text directly
            original_text = message_data['text']
            
            # Send progress update
            await manager.send_message(websocket, {
                "type": "processing_update",
                "data": {
                    "message_id": message_id,
                    "status": "processing",
                    "stage": "text_processing"
                }
            })
        
        # Send STT result immediately
        await manager.send_message(websocket, {
            "type": "stt_result",
            "data": {
                "message_id": message_id,
                "original_text": original_text,
                "status": "stt_complete"
            }
        })
        
        # Start both translation and TTS in parallel (TTS will use original text first, then update)
        translation_task = asyncio.create_task(safe_translation(
            original_text, 
            message_data['source_lang'], 
            message_data['target_lang']
        ))
        
        # Start TTS with original text first (for immediate audio feedback)
        tts_original_task = asyncio.create_task(safe_tts_processing(
            original_text,
            message_data['source_lang'],
            None,  # voice_hint
            None,  # sender_gender
            message_data['sender_id']
        ))
        
        # Wait for translation
        translated_text = await translation_task
        
        # Send translation result
        await manager.send_message(websocket, {
            "type": "translation_result",
            "data": {
                "message_id": message_id,
                "translated_text": translated_text,
                "status": "translation_complete"
            }
        })
        
        # Start TTS with translated text
        logger.info(f"Starting TTS for translated text: '{translated_text[:50]}...' in language: {message_data['target_lang']}")
        tts_translated_task = asyncio.create_task(safe_tts_processing(
            translated_text,
            message_data['target_lang'],
            None,  # voice_hint
            None,  # sender_gender
            message_data['sender_id']
        ))
        
        # Wait for both TTS tasks
        audio_original, content_type_orig, needs_fallback_orig, voice_used_orig = await tts_original_task
        audio_translated, content_type_trans, needs_fallback_trans, voice_used_trans = await tts_translated_task
        
        logger.info(f"TTS results - Original: audio_data={len(audio_original) if audio_original else 'None'}, needs_fallback={needs_fallback_orig}")
        logger.info(f"TTS results - Translated: audio_data={len(audio_translated) if audio_translated else 'None'}, needs_fallback={needs_fallback_trans}")
        
        # Encode both audio files in parallel
        audio_tasks = []
        if audio_original and not needs_fallback_orig:
            audio_tasks.append(run_in_thread(
                lambda data: base64.b64encode(data).decode('utf-8'), 
                audio_original
            ))
        else:
            audio_tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
            
        if audio_translated and not needs_fallback_trans:
            audio_tasks.append(run_in_thread(
                lambda data: base64.b64encode(data).decode('utf-8'), 
                audio_translated
            ))
        else:
            audio_tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
        
        # Wait for both audio encoding tasks
        audio_original_b64, audio_translated_b64 = await asyncio.gather(*audio_tasks)
        
        # Prepare audio URLs
        audio_url_original = None
        audio_url_translated = None
        
        if audio_original_b64:
            audio_url_original = f"data:{content_type_orig};base64,{audio_original_b64}"
        if audio_translated_b64:
            audio_url_translated = f"data:{content_type_trans};base64,{audio_translated_b64}"
        
        # Send final complete result with both audio files
        result = {
            "type": "translation_complete",
            "data": {
                "message_id": message_id,
                "original_text": original_text,
                "translated_text": translated_text,
                "audio_url_original": audio_url_original,
                "audio_url_translated": audio_url_translated,
                "status": "complete"
            }
        }
        
        # Schedule background tasks
        processing_time = time.time() - start_time
        background_manager.schedule_task(
            log_processing_metrics(message_id, processing_time, True)
        )
        background_manager.schedule_task(
            cache_translation_result(original_text, translated_text, 
                                   message_data['source_lang'], 
                                   message_data['target_lang'])
        )
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        background_manager.schedule_task(
            log_processing_metrics(message_id, processing_time, False)
        )
        logger.error(f"Voice message processing failed: {e}")
        return {"type": "error", "message": f"Processing failed: {str(e)}"}

# Optimized voice message processing function (backward compatibility)
async def process_voice_message_optimized(message_data: dict) -> dict:
    """Process voice message with all optimizations applied (backward compatibility)"""
    start_time = time.time()
    message_id = str(uuid.uuid4())
    
    try:
        # Decode audio in thread pool (non-blocking)
        audio_bytes = await run_in_thread(base64.b64decode, message_data['audio_data'])
        
        # STT processing with retry logic
        stt_result = await safe_stt_processing(audio_bytes, message_data['source_lang'])
        original_text = stt_result.get("text", "")
        
        if not original_text:
            return {"type": "error", "message": "Speech recognition failed"}
        
        # Translation with retry logic
        translated_text = await safe_translation(
            original_text, 
            message_data['source_lang'], 
            message_data['target_lang']
        )
        
        # TTS processing with retry logic
        audio_data, content_type, needs_fallback, voice_used = await safe_tts_processing(
            translated_text,
            message_data['target_lang'],
            None,  # voice_hint
            None,  # sender_gender
            message_data['sender_id']
        )
        
        # Encode audio in thread pool (non-blocking)
        audio_url = None
        if audio_data and not needs_fallback:
            audio_data_base64 = await run_in_thread(
                lambda data: base64.b64encode(data).decode('utf-8'), 
                audio_data
            )
            audio_url = f"data:{content_type};base64,{audio_data_base64}"
        
        result = {
            "type": "translation",
            "data": {
                "message_id": message_id,
                "original_text": original_text,
                "translated_text": translated_text,
                "audio_url": audio_url
            }
        }
        
        # Schedule background tasks
        processing_time = time.time() - start_time
        background_manager.schedule_task(
            log_processing_metrics(message_id, processing_time, True)
        )
        background_manager.schedule_task(
            cache_translation_result(original_text, translated_text, 
                                   message_data['source_lang'], 
                                   message_data['target_lang'])
        )
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        background_manager.schedule_task(
            log_processing_metrics(message_id, processing_time, False)
        )
        logger.error(f"Voice message processing failed: {e}")
        return {"type": "error", "message": f"Processing failed: {str(e)}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication with streaming."""
    connection_id = str(uuid.uuid4())
    await manager.connect(websocket, connection_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            # Parse JSON in thread pool (non-blocking)
            message_data = await run_in_thread(json.loads, data)
            
            # Validate message
            try:
                message = SimpleMessage(**message_data)
            except Exception as e:
                await manager.send_message(websocket, {
                    "type": "error",
                    "message": f"Invalid message format: {str(e)}"
                })
                continue
            
            # Process message with streaming function for maximum responsiveness
            result = await process_voice_message_streaming(message_data, websocket)
            await manager.send_message(websocket, result)
                
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(connection_id)

@app.websocket("/ws/parallel")
async def websocket_parallel_endpoint(websocket: WebSocket):
    """WebSocket endpoint for maximum parallel processing."""
    connection_id = str(uuid.uuid4())
    await manager.connect(websocket, connection_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            # Parse JSON in thread pool (non-blocking)
            message_data = await run_in_thread(json.loads, data)
            
            # Validate message
            try:
                message = SimpleMessage(**message_data)
            except Exception as e:
                await manager.send_message(websocket, {
                    "type": "error",
                    "message": f"Invalid message format: {str(e)}"
                })
                continue
            
            # Process message with parallel function for maximum speed
            result = await process_voice_message_parallel(message_data, websocket)
            await manager.send_message(websocket, result)
                
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(connection_id)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "optimized-simple-voice-chat"}

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down optimized voice chat backend...")
    await background_manager.cleanup()
    get_thread_pool().shutdown(wait=True)
    logger.info("Cleanup completed")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Simple Voice Chat Translation API"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        "app.main_simple:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info",
        access_log=True
    )
