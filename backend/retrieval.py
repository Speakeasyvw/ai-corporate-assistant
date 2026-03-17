from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from backend.config import VECTOR_STORE_PATH, EMBEDDING_MODEL

# Okay aca cargamos el cross-encoder una sola vez al iniciar
# este modelo es liviano y efectivo para reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_vector_store():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


def rerank_docs(query: str, docs: list, top_k: int) -> list:
    """
    Reordena los documentos candidatos usando un cross-encoder.
    El cross-encoder evalúa la relevancia de cada chunk
    en relación a la pregunta directamente, no por similitud vectorial.
    """
    if not docs:
        return docs

    # Creamos pares (pregunta, chunk) para que el cross-encoder los evalúe
    pairs = [(query, doc.page_content) for doc in docs]

    # El cross-encoder devuelve un score de relevancia para cada par
    scores = reranker.predict(pairs)

    # Ordenamos los docs de mayor a menor score
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]


def search_docs(query: str, k: int = 4, document_filter: str | None = None):
    vector_store = load_vector_store()

    # Traemos más candidatos para que el reranker tenga más donde elegir
    initial_k = max(k * 8, 20)
    docs = vector_store.similarity_search(query, k=initial_k)

    if document_filter:
        docs = [
            doc for doc in docs
            if doc.metadata.get("category") == document_filter
        ]

    # Reranking: reordenamos por relevancia semántica real
    docs = rerank_docs(query, docs, top_k=k)

    return docs