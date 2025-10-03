"""Message management API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.session import get_session
from ...db.crud import message_crud, conversation_crud
from ...db.schemas import MessagesResponse, MessageResponse
from ...core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/{conversation_id}", response_model=MessagesResponse)
async def get_messages(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: Optional[str] = Query(default=None)
) -> MessagesResponse:
    """Get messages for a conversation with pagination."""
    try:
        # Verify conversation exists
        conversation = await conversation_crud.get_by_id(session, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages, has_more, next_cursor = await message_crud.get_by_conversation(
            session, conversation_id, limit, cursor
        )
        
        # Convert to response models
        message_responses = []
        for message in messages:
            # Load sender relationship
            await session.refresh(message, ["sender"])
            message_responses.append(MessageResponse.model_validate(message))
        
        return MessagesResponse(
            messages=message_responses,
            has_more=has_more,
            next_cursor=next_cursor
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get messages for conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve messages")
