"""Database models using SQLAlchemy."""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import Enum as SQLEnum

Base = declarative_base()


class UserRole(str, Enum):
    """User roles."""
    PATIENT = "patient"
    NURSE = "nurse"
    ADMIN = "admin"


class MessageStatus(str, Enum):
    """Message status."""
    SENT = "sent"
    DELIVERED = "delivered"
    PLAYED = "played"
    FAILED = "failed"


class User(Base):
    """User model."""
    __tablename__ = "user"
    
    id = Column(String, primary_key=True)
    name = Column(String(100), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    gender = Column(String(20), nullable=True)
    tts_gender = Column(String(20), nullable=True)  # Assigned TTS gender for voice selection
    preferred_lang = Column(String(5), nullable=False)  # BCP-47 format
    preferred_voice = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sent_messages = relationship("Message", back_populates="sender", foreign_keys="Message.sender_id")
    profile = relationship("UserProfile", back_populates="user", uselist=False)


class UserProfile(Base):
    """Authentication profile for a user (email + password)."""
    __tablename__ = "user_profile"

    # Use user_id as the primary key for 1:1 relation
    user_id = Column(String, ForeignKey("user.id"), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="profile", uselist=False)


class Conversation(Base):
    """Conversation model."""
    __tablename__ = "conversation"
    
    id = Column(String, primary_key=True)
    user_a_id = Column(String, ForeignKey("user.id"), nullable=False)
    user_b_id = Column(String, ForeignKey("user.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user_a = relationship("User", foreign_keys=[user_a_id])
    user_b = relationship("User", foreign_keys=[user_b_id])


class Message(Base):
    """Message model."""
    __tablename__ = "message"
    
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("conversation.id"), nullable=False)
    sender_id = Column(String, ForeignKey("user.id"), nullable=False)
    text_source = Column(Text, nullable=False)
    text_translated = Column(Text, nullable=True)
    source_lang = Column(String(5), nullable=False)
    target_lang = Column(String(5), nullable=False)
    status = Column(SQLEnum(MessageStatus), default=MessageStatus.SENT)
    created_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    played_at = Column(DateTime, nullable=True)
    
    # Performance metrics
    client_sent_at = Column(DateTime, nullable=True)
    ttfa_ms = Column(Integer, nullable=True)  # Time to first audio
    
    # Relationships
    conversation = relationship("Conversation")
    sender = relationship("User", back_populates="sent_messages")
