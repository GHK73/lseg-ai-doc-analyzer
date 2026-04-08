# backend/config.py

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")  # ✅ FIX

    return Groq(
        api_key=GROQ_API_KEY,
        timeout=10  # prevent hanging requests
    )


# -------- Lazy Init --------
_client = None


def get_client():
    global _client

    if _client is None:
        _client = get_groq_client()

    return _client