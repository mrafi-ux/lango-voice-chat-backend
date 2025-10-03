"""Database session management."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from .models import Base
from ..core.config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def create_tables() -> None:
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Lightweight SQLite migration: ensure new nullable columns exist
        try:
            url = str(settings.database_url)
            if url.startswith("sqlite"):
                # Ensure 'gender' column on user table
                res = await conn.exec_driver_sql("PRAGMA table_info(user)")
                cols = {row[1] for row in res.fetchall()}  # name is second column
                if 'gender' not in cols:
                    await conn.exec_driver_sql("ALTER TABLE user ADD COLUMN gender VARCHAR(20)")
        except Exception:
            # Don't fail startup if migration step fails
            pass


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
