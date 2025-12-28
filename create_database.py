from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from document_loader import DocumentLoader
from dotenv import load_dotenv
import os
import shutil

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/docs"


def main():
    generate_data_store()


def generate_data_store():
    """Generate the vector database from documents"""
    documents = load_documents()
    if not documents:
        print("No documents found. Please add documents to the data/docs folder.")
        return

    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    """Load documents using the DocumentLoader class"""
    loader = DocumentLoader(DATA_PATH)
    documents = loader.load_documents()
    return documents


def split_text(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if chunks:
        # Show example chunk
        document = chunks[0] if len(chunks) > 0 else None
        if document:
            print("\nExample chunk:")
            print(document.page_content[:200] + "...")
            print(f"Metadata: {document.metadata}")

    return chunks


def save_to_chroma(chunks: list[Document]):
    """Save chunks to ChromaDB with HuggingFace embeddings"""
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize HuggingFace embeddings
    # Using a lightweight, fast model good for embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Create a new DB from the documents
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()