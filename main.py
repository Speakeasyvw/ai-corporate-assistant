from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from backend.schemas import ChatRequest, SearchRequest
from backend.rag_service import run_rag, run_rag_stream, run_retrieval, conversation_store
from backend.config import API_KEYS

app = FastAPI(title="Corporate AI Assistant API")

# Header donde se espera la API key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verifica que la API key sea válida.
    Devuelve los datos del usuario si es válida, lanza 401 si no.
    """
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return API_KEYS[api_key]


def verify_category_access(user_data: dict, category: str | None):
    """
    Verifica que el usuario tenga acceso a la categoría solicitada.
    Lanza 403 si no tiene permisos.
    """
    if category and category not in user_data["allowed_categories"]:
        raise HTTPException(
            status_code=403,
            detail=f"User '{user_data['user']}' does not have access to category '{category}'"
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest, user_data: dict = Security(verify_api_key)):
    category = request.document_filter.value if request.document_filter else None
    verify_category_access(user_data, category)

    return run_rag(
        request.question,
        k=request.k,
        document_filter=category,
        session_id=request.session_id
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest, user_data: dict = Security(verify_api_key)):
    category = request.document_filter.value if request.document_filter else None
    verify_category_access(user_data, category)

    return StreamingResponse(
        run_rag_stream(
            request.question,
            k=request.k,
            document_filter=category,
            session_id=request.session_id
        ),
        media_type="text/event-stream"
    )


@app.post("/search")
def search(request: SearchRequest, user_data: dict = Security(verify_api_key)):
    category = request.document_filter.value if request.document_filter else None
    verify_category_access(user_data, category)

    result = run_retrieval(
        request.query,
        k=request.k,
        document_filter=category
    )
    return {
        "results": result["results"]
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str, user_data: dict = Security(verify_api_key)):
    if session_id in conversation_store:
        del conversation_store[session_id]
        return {"status": "session cleared", "session_id": session_id}
    return {"status": "session not found", "session_id": session_id}