from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")


def load_documents():
    docs = []
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data/docs/")

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pdf_docs = loader.load()
        docs.extend(pdf_docs)

    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180
    )
    return splitter.split_documents(documents)


def build_index():
    print("Loading PDFs...")
    documents = load_documents()

    print(f"Loaded {len(documents)} pages")

    print("Splitting into chunks...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print(f"FAISS index saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()