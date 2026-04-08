# backend/auth/service.py

from fastapi import HTTPException
from auth.models import users_collection
from auth.utils import hash_password, verify_password, create_token


def signup_user(data):
    existing = users_collection.find_one({"email": data["email"]})

    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user = {
        "name": data["name"],
        "email": data["email"],
        "password": hash_password(data["password"])
    }

    users_collection.insert_one(user)

    return {"msg": "User created"}


def login_user(data):
    user = users_collection.find_one({"email": data["email"]})

    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not verify_password(data["password"], user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_token(str(user["_id"]))

    return {
        "token": token,
        "user": {
            "name": user["name"],
            "email": user["email"]
        }
    }