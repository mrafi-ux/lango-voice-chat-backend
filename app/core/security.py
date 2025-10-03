"""Security utilities: IDs, password hashing, and simple HMAC tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.config import settings
from ..db.session import get_session
from ..db.models import User


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return str(uuid.uuid4())


def generate_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


def generate_message_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


# Password hashing using PBKDF2-HMAC (no external deps)
def hash_password(password: str, *, iterations: int = 200_000) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256.

    Returns a string: "pbkdf2_sha256$iterations$salt_hex$hash_hex".
    """
    if not isinstance(password, str) or password == "":
        raise ValueError("password must be a non-empty string")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    try:
        scheme, iterations_str, salt_hex, hash_hex = hashed.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# Minimal JWT-like token with HS256 HMAC signing (no external deps)
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(user_id: str, *, expires_minutes: int | None = None) -> str:
    """Create a JWT access token for the given user ID."""
    return create_token(user_id, expires_minutes=expires_minutes)


def create_token(user_id: str, *, expires_minutes: int | None = None) -> str:
    """Create a signed token for the given user ID."""
    if expires_minutes is None:
        expires_minutes = settings.auth_token_expire_minutes
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "exp": int(time.time()) + int(expires_minutes) * 60,
    }
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(settings.auth_secret_key.encode(), signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(sig)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def verify_token(token: str) -> Optional[str]:
    """Verify a token and return the user ID if valid, else None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(settings.auth_secret_key.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, _b64url_decode(signature_b64)):
            return None
        payload = json.loads(_b64url_decode(payload_b64).decode())
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return str(payload.get("sub"))
    except Exception:
        return None


async def get_current_user(
    authorization: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> User:
    """FastAPI dependency to get the current authenticated user."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
