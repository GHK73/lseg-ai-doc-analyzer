# bakcend/document/routes.py

from fastapi import APIRouter, UploadFile, File, Depends
from document.service import upload_document,ask_question
from auth.deps import get_current_user


router = APIRouter()

@router.post("/upload")
def upload(file: UploadFile, user_id=Depends(get_current_user)):
    return upload_document(user_id, file)

@router.get("/ask")
def ask(query: str, user_id=Depends(get_current_user)):
    return ask_question(user_id, query)