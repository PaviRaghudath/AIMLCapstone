from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.rag_engine import collection, embedding_model
from backend.rule_based_qa_parser import answer_question

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
def rag_answer(fastApi_request: Request, request: ChatRequest):
    return {"answer": answer_question(fastApi_request, request.question)}
