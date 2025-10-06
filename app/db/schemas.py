"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, ConfigDict

from .models import UserRole, MessageStatus


# User schemas
class UserCreate(BaseModel):
    """Schema for creating a user."""
    name: str = Field(min_length=1, max_length=100)
    role: UserRole
    gender: Optional[str] = Field(default=None, max_length=20)
    preferred_lang: str = Field(min_length=2, max_length=5)
    preferred_voice: Optional[str] = Field(default=None, max_length=100)


class UserRegister(BaseModel):
    """Schema for user registration (with credentials)."""
    name: str = Field(min_length=1, max_length=100)
    role: UserRole
    gender: Optional[str] = Field(default=None, max_length=20)
    preferred_lang: str = Field(min_length=2, max_length=5)
    preferred_voice: Optional[str] = Field(default=None, max_length=100)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=6, max_length=128)


class UserResponse(BaseModel):
    """Schema for user response."""
    id: str
    name: str
    role: UserRole
    gender: Optional[str]
    tts_gender: Optional[str]
    preferred_lang: str
    preferred_voice: Optional[str]
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        }
    )


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=6, max_length=128)


class AuthResponse(BaseModel):
    """Schema for authentication response."""
    user: "UserResponse"
    token: str


# Conversation schemas
class ConversationCreate(BaseModel):
    """Schema for creating a conversation."""
    user_a_id: str
    user_b_id: str


class ConversationResponse(BaseModel):
    """Schema for conversation response."""
    id: str
    user_a_id: str
    user_b_id: str
    created_at: datetime
    user_a: Optional[UserResponse] = None
    user_b: Optional[UserResponse] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        }
    )


# Message schemas
class MessageCreate(BaseModel):
    """Schema for creating a message."""
    conversation_id: str
    sender_id: str
    source_lang: str = Field(min_length=2, max_length=5)
    target_lang: str = Field(min_length=2, max_length=5)
    text_source: str = Field(min_length=1)


class MessageResponse(BaseModel):
    """Schema for message response."""
    id: str
    conversation_id: str
    sender_id: str
    source_lang: str
    target_lang: str
    text_source: str
    text_translated: Optional[str]
    status: MessageStatus
    ttfa_ms: Optional[int]
    created_at: datetime
    sender: Optional[UserResponse] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda dt: dt.isoformat() if dt else None
        }
    )


class MessagesResponse(BaseModel):
    """Schema for paginated messages response."""
    messages: List[MessageResponse]
    has_more: bool
    next_cursor: Optional[str]


# WebSocket schemas
class WSJoinMessage(BaseModel):
    """WebSocket join message."""
    type: str = "join"
    user_id: str


class WSVoiceNoteMessage(BaseModel):
    """WebSocket voice note message."""
    type: str = "voice_note"
    conversation_id: str
    sender_id: str
    source_lang: str
    target_lang: str
    text_source: str
    client_sent_at: datetime


class WSMessageResponse(BaseModel):
    """WebSocket message response."""
    type: str = "message"
    message: MessageResponse
    play_now: Optional[dict] = None  # {lang: str, text: str, sender_gender: str}


class WSPresenceResponse(BaseModel):
    """WebSocket presence response."""
    type: str = "presence"
    online_user_ids: List[str]


class WSErrorResponse(BaseModel):
    """WebSocket error response."""
    type: str = "error"
    message: str
    code: Optional[str] = None
