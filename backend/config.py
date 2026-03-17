from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_TOP_K = 3
MAX_SNIPPET_LENGTH = 250