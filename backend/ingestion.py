from pathlib import Path
from annotated_types import doc
from dotenv import load_dotenv
from document_registry import DOCUMENT_CATEGORIES
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


load_dotenv()

DOCS_PATH = Path("data/documents")
VECTOR_STORE_PATH = "vector_store/faiss_index"


def load_pdfs():
    documents = []

    pdf_files = list(DOCS_PATH.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No se encontraron archivos PDF en data/documents")

    for pdf_file in pdf_files:
        print(f"Leyendo PDF: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        pdf_docs = loader.load()

        for doc in pdf_docs:
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["category"] = DOCUMENT_CATEGORIES.get(pdf_file.name, "unknown")

        documents.extend(pdf_docs)

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return text_splitter.split_documents(documents)


def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

    return vector_store


def main():
    print("1. Cargando PDFs...")
    documents = load_pdfs()
    print(f"Se cargaron {len(documents)} páginas/documentos")

    print("\n2. Dividiendo en chunks...")
    chunks = split_documents(documents)
    print(f"Se generaron {len(chunks)} chunks")

    print("\n3. Construyendo índice FAISS...")
    build_vector_store(chunks)
    print("Índice guardado correctamente en vector_store/faiss_index")


if __name__ == "__main__":
    main()