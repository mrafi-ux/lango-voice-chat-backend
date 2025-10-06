"""Background workers for async tasks."""

import asyncio
from typing import Optional

from ..db.session import AsyncSessionLocal
from ..db.crud import message_crud
from ..db.models import MessageStatus
from ..core.logging import get_logger

logger = get_logger(__name__)


class PersistenceWorker:
    """Worker for handling async persistence tasks."""
    
    @staticmethod
    async def persist_message_translation(
        message_id: str,
        translated_text: str
    ) -> None:
        """Persist message translation in background."""
        try:
            async with AsyncSessionLocal() as session:
                await message_crud.update_translation(
                    session, message_id, translated_text
                )
            logger.debug(f"Persisted translation for message {message_id}")
        except Exception as e:
            logger.error(f"Failed to persist translation for {message_id}: {e}")
    
    @staticmethod
    async def update_message_status(
        message_id: str,
        status: MessageStatus,
        ttfa_ms: Optional[int] = None
    ) -> None:
        """Update message status in background."""
        try:
            async with AsyncSessionLocal() as session:
                await message_crud.update_status(
                    session, message_id, status, ttfa_ms
                )
            logger.debug(f"Updated status for message {message_id} to {status}")
        except Exception as e:
            logger.error(f"Failed to update status for {message_id}: {e}")


# Global worker instance
persistence_worker = PersistenceWorker()


def schedule_background_task(coro):
    """Schedule a coroutine to run in the background."""
    loop = asyncio.get_event_loop()
    loop.create_task(coro)
