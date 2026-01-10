import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "rag_corpus"

def load_documents():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"'{DATA_DIR}/' folder not found. Create it and add documents.")

    docs = []

    # TXT
    docs += DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    ).load()

    # MD
    docs += DirectoryLoader(
        DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    ).load()

    # PDF
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                docs += PyPDFLoader(path).load()

    return docs

def main():
    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in data/. Add .txt/.md/.pdf files.")

    print(f"Loaded documents: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created chunks: {len(chunks)}")

    # Local embeddings (no API key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Rebuild DB from scratch every time (clean for assignment)
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    vectordb.add_documents(chunks)
    vectordb.persist()

    print(f"✅ Vector DB saved to: {DB_DIR}/ (collection: {COLLECTION_NAME})")

if __name__ == "__main__":
    main()
