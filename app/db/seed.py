"""Database seeding script to create demo users and data."""

import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from .session import AsyncSessionLocal, create_tables
from .models import User, UserRole, Conversation
from .crud import user_crud, conversation_crud
from .schemas import UserCreate, ConversationCreate
from ..core.logging import get_logger
from ..core.security import generate_conversation_id

logger = get_logger(__name__)


async def seed_demo_users() -> None:
    """Create demo users for testing."""
    async with AsyncSessionLocal() as session:
        try:
            # Check if users already exist
            existing_users = await user_crud.get_all(session)
            if existing_users:
                logger.info(f"Found {len(existing_users)} existing users, skipping seed")
                return
            
            # Create demo users with fixed IDs to match frontend expectations
            demo_users = [
                {
                    "id": "1",
                    "name": "Admin User",
                    "role": UserRole.ADMIN,
                    "gender": "female",
                    "preferred_lang": "en",
                    "preferred_voice": "Rachel"
                },
                {
                    "id": "2", 
                    "name": "Ana Rodriguez",
                    "role": UserRole.PATIENT,
                    "gender": "female",
                    "preferred_lang": "es",
                    "preferred_voice": "Valentina"
                },
                {
                    "id": "3",
                    "name": "Ben Smith", 
                    "role": UserRole.NURSE,
                    "gender": "male",
                    "preferred_lang": "en",
                    "preferred_voice": "Clyde"
                }
            ]
            
            created_users = []
            for user_data in demo_users:
                # Create user directly with fixed ID
                user = User(
                    id=user_data["id"],
                    name=user_data["name"],
                    role=user_data["role"],
                    gender=user_data.get("gender"),
                    preferred_lang=user_data["preferred_lang"],
                    preferred_voice=user_data["preferred_voice"],
                    created_at=datetime.utcnow()
                )
                session.add(user)
                created_users.append(user)
                logger.info(f"Created demo user: {user.name} ({user.role.value}) with ID: {user.id}")
            
            await session.commit()
            
            # Create a demo conversation between Ana and Ben
            try:
                conversation = Conversation(
                    id=generate_conversation_id(),
                    user_a_id="2",  # Ana
                    user_b_id="3",  # Ben
                    created_at=datetime.utcnow()
                )
                session.add(conversation)
                await session.commit()
                logger.info(f"Created demo conversation: {conversation.id}")
            except Exception as conv_error:
                logger.warning(f"Failed to create demo conversation: {conv_error}")
            
            logger.info(f"Successfully seeded {len(created_users)} demo users")
            
        except Exception as e:
            logger.error(f"Failed to seed demo users: {e}")
            await session.rollback()
            raise


async def init_database() -> None:
    """Initialize database with tables and seed data."""
    try:
        logger.info("Initializing database...")
        
        # Create tables
        await create_tables()
        logger.info("Database tables created")
        
        # Seed demo data
        await seed_demo_users()
        logger.info("Database seeding completed")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(init_database()) 