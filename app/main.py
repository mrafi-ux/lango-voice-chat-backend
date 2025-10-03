"""FastAPI application main module."""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging import get_logger
from .db.seed import init_database
from .api.v1 import routes_users, routes_conversations, routes_messages, routes_tts, routes_stt, routes_capabilities
from .api.v1 import routes_auth
from .api.v1.ws import handle_websocket

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting VoiceCare backend...")
    
    # Initialize database with tables and seed data
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Don't fail startup if database init fails
    
    yield
    
    # Shutdown
    logger.info("Shutting down VoiceCare backend...")


# Create FastAPI app
app = FastAPI(
    title="VoiceCare API",
    description="Patient-Nurse Voice Translation System",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(
    routes_users.router,
    prefix="/api/v1/users",
    tags=["users"]
)

app.include_router(
    routes_conversations.router,
    prefix="/api/v1/conversations",
    tags=["conversations"]
)

app.include_router(
    routes_messages.router,
    prefix="/api/v1/messages",
    tags=["messages"]
)

app.include_router(
    routes_tts.router,
    prefix="/api/v1/tts",
    tags=["tts"]
)

app.include_router(
    routes_stt.router,
    prefix="/api/v1/stt",
    tags=["stt"]
)

app.include_router(
    routes_capabilities.router,
    prefix="/api/v1/capabilities",
    tags=["capabilities"]
)

app.include_router(
    routes_auth.router,
    prefix="/api/v1/auth",
    tags=["auth"]
)


# WebSocket endpoint
@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    connection_id = str(uuid.uuid4())
    await handle_websocket(websocket, connection_id)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "voicecare-backend"}

# Debug endpoint for WebSocket connections
@app.get("/debug/connections")
async def debug_connections():
    """Debug endpoint to check WebSocket connections."""
    from .api.v1.ws import manager
    return {
        "active_connections": len(manager.active_connections),
        "user_connections": dict(manager.user_connections),
        "connection_ids": list(manager.active_connections.keys())
    }


# Metrics endpoint (for monitoring)
@app.get("/api/v1/metrics")
async def get_metrics():
    """Get performance metrics."""
    return {
        "ttfa_stats": metrics_service.get_ttfa_stats(),
        "translation_stats": metrics_service.get_translation_stats(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True
    )
