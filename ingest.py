import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DOCS_DIR   = "./docs"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

def load_documents(docs_dir):
    documents = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"[ERROR] Docs folder '{docs_dir}' not found. Create it and add your files.")
        return documents

    files = list(docs_path.iterdir())
    if not files:
        print(f"[ERROR] No files found in '{docs_dir}'. Add your PDFs, Word docs, or CSV files.")
        return documents

    for file_path in files:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"  Loaded PDF:  {file_path.name} ({len(docs)} pages)")

            elif suffix in (".docx", ".doc"):
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"  Loaded DOCX: {file_path.name} ({len(docs)} sections)")

            elif suffix == ".csv":
                # CSV rows become structured text documents with column labels,
                # which generally improves retrieval for tabular catalog data.
                loader = CSVLoader(str(file_path), autodetect_encoding=True)
                docs = loader.load()
                documents.extend(docs)
                print(f"  Loaded CSV:  {file_path.name} ({len(docs)} rows)")

            else:
                print(f"  Skipped:     {file_path.name} (unsupported type)")

        except Exception as e:
            print(f"  [ERROR] Could not load {file_path.name}: {e}")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks):
    print(f"\nLoading embedding model '{EMBED_MODEL}' ...")
    print("(First run downloads ~90 MB — this is normal)\n")

    # Rebuild from scratch so removed/old source files do not leave stale vectors.
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Embedding {len(chunks)} chunks and saving to ChromaDB ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done. Vector store saved to '{CHROMA_DIR}'")
    return vectorstore


def main():
    print("=" * 50)
    print("  RAG Ingestion Pipeline")
    print("=" * 50)

    print(f"\nStep 1 — Loading documents from '{DOCS_DIR}' ...\n")
    documents = load_documents(DOCS_DIR)
    if not documents:
        return
    print(f"\n  Total pages/sections loaded: {len(documents)}")

    print(f"\nStep 2 — Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    chunks = split_documents(documents)
    print(f"  Total chunks created: {len(chunks)}")

    print(f"\nStep 3 — Building vector store ...")
    build_vectorstore(chunks)

    print("\n" + "=" * 50)
    print(f"  Ingestion complete!")
    print(f"  {len(chunks)} chunks stored in ChromaDB.")
    print(f"  Now run:  streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()