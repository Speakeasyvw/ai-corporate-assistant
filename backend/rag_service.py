from langchain_openai import ChatOpenAI

from backend.config import CHAT_MODEL, DEFAULT_TOP_K, MAX_SNIPPET_LENGTH
from backend.retrieval import search_docs


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


def build_prompt(question: str, context: str):
    return f"""
You are an enterprise AI assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent information.
2. If the answer is not clearly supported by the context, say:
   "I could not find enough evidence in the provided documents."
3. Keep the answer clear and concise.
4. When possible, reference the source file and page.
5. Prefer the most relevant evidence.

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


def run_rag(question: str, k: int = DEFAULT_TOP_K, document_filter: str | None = None):
    docs = search_docs(question, k=k, document_filter=document_filter)
    context = format_context(docs)
    prompt = build_prompt(question, context)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": format_sources(docs)
    }