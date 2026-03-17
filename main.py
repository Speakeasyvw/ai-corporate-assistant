from fastapi import FastAPI
from backend.schemas import ChatRequest, SearchRequest
from backend.rag_service import run_rag, run_retrieval, conversation_store

app = FastAPI(title="Corporate AI Assistant API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    return run_rag(
        request.question,
        k=request.k,
        document_filter=request.document_filter.value if request.document_filter else None,
        session_id=request.session_id
    )


@app.post("/search")
def search(request: SearchRequest):
    result = run_retrieval(
        request.query,
        k=request.k,
        document_filter=request.document_filter.value if request.document_filter else None
    )
    return {
        "results": result["results"]
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Elimina el historial de una sesión específica."""
    if session_id in conversation_store:
        del conversation_store[session_id]
        return {"status": "session cleared", "session_id": session_id}
    return {"status": "session not found", "session_id": session_id}