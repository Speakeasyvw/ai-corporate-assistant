from enum import Enum
from typing import Optional
from pydantic import BaseModel


class DocumentCategory(str, Enum):
    financial = "financial"
    technical = "technical"
    operations = "operations"
    compliance = "compliance"


class ChatRequest(BaseModel):
    question: str
    k: int = 3
    document_filter: Optional[DocumentCategory] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 3
    document_filter: Optional[DocumentCategory] = None