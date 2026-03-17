from langchain_openai import ChatOpenAI

from backend.config import CHAT_MODEL, DEFAULT_TOP_K, MAX_SNIPPET_LENGTH
from backend.retrieval import search_docs

# Historial de conversaciones por session_id
# { "session-123": [ {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."} ] }
conversation_store: dict = {}


def format_context(docs):
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content.strip()

        context_parts.append(
            f"[Source {i} | File: {source} | Page: {page}]\n{content}"
        )

    return "\n\n".join(context_parts)


def format_history(history: list) -> str:
    """Convierte el historial en texto legible para el prompt."""
    if not history:
        return "No previous conversation."

    lines = []
    for message in history:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")

    return "\n".join(lines)


def build_prompt(question: str, context: str, history: list):
    history_text = format_history(history)

    return f"""
You are an enterprise AI assistant.

Answer the user's question using ONLY the provided context.
You may use the conversation history to understand references like "it", "that", "the same topic", etc.

Rules:
1. Do not invent information.
2. If the answer is not clearly supported by the context, say:
   "I could not find enough evidence in the provided documents."
3. Keep the answer clear and concise.
4. When possible, reference the source file and page.
5. Prefer the most relevant evidence.

Conversation history:
{history_text}

User question:
{question}

Context:
{context}
"""


def format_sources(docs):
    formatted = []

    for doc in docs:
        formatted.append({
            "file": doc.metadata.get("source_file"),
            "page": doc.metadata.get("page"),
            "category": doc.metadata.get("category"),
            "snippet": doc.page_content[:MAX_SNIPPET_LENGTH].replace("\n", " ")
        })

    return formatted


def run_retrieval(query: str, k: int = DEFAULT_TOP_K, document_filter: str | None = None):
    docs = search_docs(query, k=k, document_filter=document_filter)
    return {
        "results": format_sources(docs),
        "raw_docs": docs
    }


def run_rag(question: str, k: int = DEFAULT_TOP_K, document_filter: str | None = None, session_id: str = None):
    # Obtenemos o creamos el historial de esta sesión
    history = conversation_store.get(session_id, [])

    docs = search_docs(question, k=k, document_filter=document_filter)
    context = format_context(docs)
    prompt = build_prompt(question, context, history)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0
    )

    response = llm.invoke(prompt)
    answer = response.content

    # Guardamos la pregunta y respuesta en el historial
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    conversation_store[session_id] = history

    return {
        "answer": answer,
        "sources": format_sources(docs),
        "session_id": session_id
    }