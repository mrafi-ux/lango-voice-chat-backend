"""User management API routes."""

from typing import List
import uuid
import hashlib

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...db.session import get_session
from ...db.crud import user_crud
from ...db.schemas import UserCreate, UserResponse, UserRegister, LoginRequest, AuthResponse
from ...db.models import User as UserModel, UserProfile
from ...core.logging import get_logger
from ...core.security import get_current_user, create_access_token

logger = get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=UserResponse, status_code=201)
async def create_user(
    user_data: UserCreate,
    session: AsyncSession = Depends(get_session)
) -> UserResponse:
    """Create a new user."""
    try:
        user = await user_crud.create(session, user_data)
        logger.info(f"Created user: {user.name} ({user.role})")
        return UserResponse.model_validate(user)
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=400, detail="Failed to create user")


@router.get("/", response_model=List[UserResponse])
async def get_users(
    session: AsyncSession = Depends(get_session)
) -> List[UserResponse]:
    """Get all users."""
    try:
        users = await user_crud.get_all(session)
        return [UserResponse.model_validate(user) for user in users]
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    session: AsyncSession = Depends(get_session)
) -> UserResponse:
    """Get user by ID."""
    try:
        user = await user_crud.get_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return UserResponse.model_validate(user)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user")


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register_user(
    user_data: UserRegister,
    session: AsyncSession = Depends(get_session)
) -> AuthResponse:
    """Register a new user with email and password."""
    try:
        # Check if email already exists
        result = await session.execute(
            select(UserProfile).where(UserProfile.email == user_data.email)
        )
        existing_profile = result.scalar_one_or_none()
        
        if existing_profile:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Create user
        user_id = str(uuid.uuid4())
        user = UserModel(
            id=user_id,
            name=user_data.name,
            role=user_data.role,
            gender=user_data.gender,
            preferred_lang=user_data.preferred_lang,
            preferred_voice=user_data.preferred_voice
        )
        
        # Create user profile with hashed password
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        user_profile = UserProfile(
            user_id=user_id,
            email=user_data.email,
            password_hash=password_hash
        )
        
        session.add(user)
        session.add(user_profile)
        await session.commit()
        await session.refresh(user)
        
        # Create JWT token
        token = create_access_token(user_id)
        
        logger.info(f"Registered user: {user.name} ({user.email})")
        
        return AuthResponse(
            user=UserResponse.model_validate(user),
            token=token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register user: {e}")
        raise HTTPException(status_code=400, detail="Failed to register user")


@router.post("/login", response_model=AuthResponse)
async def login_user(
    login_data: LoginRequest,
    session: AsyncSession = Depends(get_session)
) -> AuthResponse:
    """Login user with email and password."""
    try:
        # Find user profile by email
        result = await session.execute(
            select(UserProfile).where(UserProfile.email == login_data.email)
        )
        user_profile = result.scalar_one_or_none()
        
        if not user_profile:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Verify password
        password_hash = hashlib.sha256(login_data.password.encode()).hexdigest()
        if user_profile.password_hash != password_hash:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Get user details
        user = await user_crud.get_by_id(session, user_profile.user_id)
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        # Create JWT token
        token = create_access_token(user.id)
        
        logger.info(f"User logged in: {user.name} ({user_profile.email})")
        
        return AuthResponse(
            user=UserResponse.model_validate(user),
            token=token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to login user: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserModel = Depends(get_current_user)) -> UserResponse:
    """Get the currently authenticated user."""
    return UserResponse.model_validate(current_user)
