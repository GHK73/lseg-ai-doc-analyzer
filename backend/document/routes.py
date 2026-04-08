from fastapi import APIRouter, UploadFile, File, Depends

from document.service import (
    upload_document,
    ask_question,
    list_documents,
    delete_document
)

from auth.deps import get_current_user

router = APIRouter()


# -------- Upload --------
@router.post("/upload")
def upload(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    user_id = current_user
    return upload_document(user_id, file)


# -------- Ask --------
@router.get("/ask")
def ask(
    query: str,
    current_user=Depends(get_current_user)
):
    user_id = current_user
    return ask_question(user_id, query)


# -------- List Documents --------
@router.get("/documents")
def get_documents(
    current_user=Depends(get_current_user)
):
    user_id = current_user
    return list_documents(user_id)


# -------- Delete Document --------
@router.delete("/documents/{doc_id}")
def remove_document(
    doc_id: str,
    current_user=Depends(get_current_user)
):
    user_id = current_user
    return delete_document(user_id, doc_id)