from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from backend.config import VECTOR_STORE_PATH, EMBEDDING_MODEL


def load_vector_store():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


def search_docs(query: str, k: int = 4, document_filter: str | None = None):
    vector_store = load_vector_store()

    # Recuperamos más resultados para después filtrar
    initial_k = max(k * 4, 10)
    docs = vector_store.similarity_search(query, k=initial_k)

    if document_filter:
        docs = [
            doc for doc in docs
            if doc.metadata.get("category") == document_filter
        ]

    return docs[:k]