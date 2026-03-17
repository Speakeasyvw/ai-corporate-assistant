import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_TOP_K = 3
MAX_SNIPPET_LENGTH = 250

# API Keys con sus categorías permitidas
# en producción esto vendría de una base de datos obvio
API_KEYS = {
    "admin-key-001": {
        "user": "admin",
        "allowed_categories": ["financial", "technical", "operations", "compliance"]
    },
    "financial-key-002": {
        "user": "financial_analyst",
        "allowed_categories": ["financial"]
    },
    "legal-key-003": {
        "user": "legal_team",
        "allowed_categories": ["compliance"]
    },
    "tech-key-004": {
        "user": "tech_team",
        "allowed_categories": ["technical", "operations"]
    }
}