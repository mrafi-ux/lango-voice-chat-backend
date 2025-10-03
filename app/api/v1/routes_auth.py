"""Authentication routes: register, login, and me."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...db.session import get_session
from ...db.crud import user_crud, user_profile_crud
from ...db.schemas import UserRegister, LoginRequest, AuthResponse, UserResponse
from ...core.security import hash_password, verify_password, create_token, get_current_user
from ...core.logging import get_logger
from ...db.models import User

router = APIRouter()
logger = get_logger(__name__)


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(
    data: UserRegister,
    session: AsyncSession = Depends(get_session)
) -> AuthResponse:
    """Register a new user with credentials."""
    # Check if email already exists
    existing_profile = await user_profile_crud.get_by_email(session, data.email)
    if existing_profile:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create base user
    user = await user_crud.create(
        session,
        user_data=data,  # compatible fields with UserCreate
    )

    # Create profile with hashed password
    pwd_hash = hash_password(data.password)
    await user_profile_crud.create(session, user.id, data.email, pwd_hash)

    token = create_token(user.id)
    logger.info(f"Registered new user {user.id} with email {data.email}")
    return AuthResponse(user=UserResponse.model_validate(user), token=token)


@router.post("/login", response_model=AuthResponse)
async def login(
    credentials: LoginRequest,
    session: AsyncSession = Depends(get_session)
) -> AuthResponse:
    """Authenticate a user and return token."""
    profile = await user_profile_crud.get_by_email(session, credentials.email)
    if not profile or not verify_password(credentials.password, profile.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Load user
    result = await session.execute(select(User).where(User.id == profile.user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    token = create_token(user.id)
    logger.info(f"User {user.id} logged in")
    return AuthResponse(user=UserResponse.model_validate(user), token=token)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated user."""
    return UserResponse.model_validate(current_user)

