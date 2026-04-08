# backend/auth/utils.py

from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("JWT_SECRET", "secret")
ALGORITHM = "HS256"


def hash_password(password: str):
    if len(password) > 72:
        password = password[:72]  # bcrypt limit

    return pwd_context.hash(password)


def verify_password(password: str, hashed: str):
    if len(password) > 72:
        password = password[:72]

    return pwd_context.verify(password, hashed)


def create_token(user_id: str):
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=1)
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)