"""CRUD operations for database models."""

from typing import List, Optional, Tuple
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload

from .models import User, Conversation, Message, MessageStatus, UserProfile
from .schemas import UserCreate, MessageCreate
from ..core.security import generate_user_id, generate_conversation_id, generate_message_id


class UserCRUD:
    """CRUD operations for User model."""
    
    @staticmethod
    async def create(session: AsyncSession, user_data: UserCreate) -> User:
        """Create a new user."""
        user = User(
            id=generate_user_id(),
            name=user_data.name,
            role=user_data.role,
            gender=getattr(user_data, 'gender', None),
            preferred_lang=user_data.preferred_lang,
            preferred_voice=user_data.preferred_voice
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user
    
    @staticmethod
    async def get_by_id(session: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all(session: AsyncSession) -> List[User]:
        """Get all users."""
        result = await session.execute(select(User).order_by(User.created_at.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def get_by_email(session: AsyncSession, email: str) -> Optional[User]:
        """Get user by email via profile relation."""
        result = await session.execute(
            select(User).join(UserProfile).where(UserProfile.email == email)
        )
        return result.scalar_one_or_none()


class ConversationCRUD:
    """CRUD operations for Conversation model."""
    
    @staticmethod
    async def create_or_get(
        session: AsyncSession, 
        user_a_id: str, 
        user_b_id: str
    ) -> Conversation:
        """Create or get existing conversation between two users."""
        # Ensure consistent ordering for uniqueness
        min_id, max_id = (user_a_id, user_b_id) if user_a_id < user_b_id else (user_b_id, user_a_id)
        
        # Try to find existing conversation
        result = await session.execute(
            select(Conversation).where(
                and_(
                    Conversation.user_a_id == min_id,
                    Conversation.user_b_id == max_id
                )
            ).options(selectinload(Conversation.user_a), selectinload(Conversation.user_b))
        )
        conversation = result.scalar_one_or_none()
        
        if conversation:
            return conversation
        
        # Create new conversation
        conversation = Conversation(
            id=generate_conversation_id(),
            user_a_id=min_id,
            user_b_id=max_id
        )
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
        return conversation
    
    @staticmethod
    async def get_by_id(session: AsyncSession, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        result = await session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.user_a), selectinload(Conversation.user_b))
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_all(session: AsyncSession) -> List[Conversation]:
        """Get all conversations."""
        result = await session.execute(
            select(Conversation)
            .options(selectinload(Conversation.user_a), selectinload(Conversation.user_b))
            .order_by(Conversation.created_at.desc())
        )
        return list(result.scalars().all())


class MessageCRUD:
    """CRUD operations for Message model."""
    
    @staticmethod
    async def create(session: AsyncSession, message_data: MessageCreate) -> Message:
        """Create a new message."""
        message = Message(
            id=generate_message_id(),
            conversation_id=message_data.conversation_id,
            sender_id=message_data.sender_id,
            source_lang=message_data.source_lang,
            target_lang=message_data.target_lang,
            text_source=message_data.text_source
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message
    
    @staticmethod
    async def get_by_conversation(
        session: AsyncSession,
        conversation_id: str,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> Tuple[List[Message], bool, Optional[str]]:
        """Get messages by conversation with pagination."""
        query = select(Message).where(Message.conversation_id == conversation_id)
        
        if cursor:
            # Parse cursor (timestamp)
            try:
                cursor_time = datetime.fromisoformat(cursor)
                query = query.where(Message.created_at < cursor_time)
            except ValueError:
                pass  # Invalid cursor, ignore
        
        query = query.options(selectinload(Message.sender)).order_by(desc(Message.created_at)).limit(limit + 1)
        result = await session.execute(query)
        messages = list(result.scalars().all())
        
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]
        
        next_cursor = None
        if has_more and messages:
            next_cursor = messages[-1].created_at.isoformat()
        
        # Reverse to show oldest first
        messages.reverse()
        
        return messages, has_more, next_cursor
    
    @staticmethod
    async def update_translation(
        session: AsyncSession,
        message_id: str,
        translated_text: str
    ) -> Optional[Message]:
        """Update message with translation."""
        result = await session.execute(select(Message).where(Message.id == message_id))
        message = result.scalar_one_or_none()
        
        if message:
            message.text_translated = translated_text
            await session.commit()
            await session.refresh(message)
        
        return message
    
    @staticmethod
    async def update_status(
        session: AsyncSession,
        message_id: str,
        status: MessageStatus,
        ttfa_ms: Optional[int] = None
    ) -> Optional[Message]:
        """Update message status and TTFA."""
        result = await session.execute(select(Message).where(Message.id == message_id))
        message = result.scalar_one_or_none()
        
        if message:
            message.status = status
            if ttfa_ms is not None:
                message.ttfa_ms = ttfa_ms
            await session.commit()
            await session.refresh(message)
        
        return message


# Create CRUD instances
user_crud = UserCRUD()
conversation_crud = ConversationCRUD()
message_crud = MessageCRUD()


class UserProfileCRUD:
    """CRUD operations for UserProfile model."""

    @staticmethod
    async def get_by_email(session: AsyncSession, email: str) -> Optional[UserProfile]:
        result = await session.execute(select(UserProfile).where(UserProfile.email == email))
        return result.scalar_one_or_none()

    @staticmethod
    async def create(session: AsyncSession, user_id: str, email: str, password_hash: str) -> UserProfile:
        profile = UserProfile(user_id=user_id, email=email, password_hash=password_hash)
        session.add(profile)
        await session.commit()
        await session.refresh(profile)
        return profile


user_profile_crud = UserProfileCRUD()
