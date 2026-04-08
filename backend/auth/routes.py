# backend/auth/routes.py

from fastapi import APIRouter
from auth.service import signup_user, login_user

router = APIRouter()


@router.post("/signup")
def signup(data: dict):
    return signup_user(data)


@router.post("/login")
def login(data: dict):
    return login_user(data)