"""Conversation management API routes."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.session import get_session
from ...db.crud import conversation_crud, user_crud
from ...db.schemas import ConversationCreate, ConversationResponse
from ...core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[ConversationResponse])
async def get_conversations(
    session: AsyncSession = Depends(get_session)
) -> List[ConversationResponse]:
    """Get all conversations."""
    try:
        conversations = await conversation_crud.get_all(session)
        
        # Load user relationships for all conversations
        for conversation in conversations:
            await session.refresh(conversation, ["user_a", "user_b"])
        
        logger.info(f"Retrieved {len(conversations)} conversations")
        return [ConversationResponse.model_validate(conv) for conv in conversations]
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")


@router.post("/", response_model=ConversationResponse, status_code=201)
async def create_or_get_conversation(
    conversation_data: ConversationCreate,
    session: AsyncSession = Depends(get_session)
) -> ConversationResponse:
    """Create or get existing conversation between two users."""
    try:
        # Validate users exist
        user_a = await user_crud.get_by_id(session, conversation_data.user_a_id)
        user_b = await user_crud.get_by_id(session, conversation_data.user_b_id)
        
        if not user_a:
            raise HTTPException(status_code=404, detail=f"User A not found: {conversation_data.user_a_id}")
        if not user_b:
            raise HTTPException(status_code=404, detail=f"User B not found: {conversation_data.user_b_id}")
        
        if conversation_data.user_a_id == conversation_data.user_b_id:
            raise HTTPException(status_code=400, detail="Cannot create conversation with same user")
        
        conversation = await conversation_crud.create_or_get(
            session, 
            conversation_data.user_a_id, 
            conversation_data.user_b_id
        )
        
        # Load user relationships
        await session.refresh(conversation, ["user_a", "user_b"])
        
        logger.info(f"Created/retrieved conversation {conversation.id} between {user_a.name} and {user_b.name}")
        return ConversationResponse.model_validate(conversation)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session)
) -> ConversationResponse:
    """Get conversation by ID."""
    try:
        conversation = await conversation_crud.get_by_id(session, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        # Load user relationships
        await session.refresh(conversation, ["user_a", "user_b"])
        
        return ConversationResponse.model_validate(conversation)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")
