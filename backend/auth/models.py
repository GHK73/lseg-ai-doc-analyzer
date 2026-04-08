# backend/auth/models.py

from pymongo import MongoClient
import os


client = MongoClient(os.getenv("MONGO_URI"))
db = client["rag_app"]

users_collection = db["users"]