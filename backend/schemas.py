from enum import Enum
from typing import Optional
from pydantic import BaseModel
import uuid


class DocumentCategory(str, Enum):
    financial = "financial"
    technical = "technical"
    operations = "operations"
    compliance = "compliance"


class ChatRequest(BaseModel):
    question: str
    k: int = 3
    document_filter: Optional[DocumentCategory] = None
    session_id: str = None

    def model_post_init(self, __context):
        # si no mandan session_id, generamos uno automáticamente
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())


class SearchRequest(BaseModel):
    query: str
    k: int = 3
    document_filter: Optional[DocumentCategory] = None