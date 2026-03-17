from fastapi import FastAPI
from backend.schemas import ChatRequest, SearchRequest
from backend.rag_service import run_rag, run_retrieval

app = FastAPI(title="Corporate AI Assistant API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    return run_rag(
        request.question,
        k=request.k,
        document_filter=request.document_filter.value if request.document_filter else None
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