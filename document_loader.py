from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
import os
from typing import List


class DocumentLoader:
    """Handles loading of multiple document types"""

    def __init__(self, data_path: str = "data/docs"):
        self.data_path = data_path

    def load_documents(self) -> List[Document]:
        """Load documents from multiple file types"""
        documents = []

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"Created directory: {self.data_path}")
            return documents

        # Load all file types individually for better error handling
        file_extensions = {
            ".md": self._load_text_file,
            ".txt": self._load_text_file,
            ".pdf": self._load_pdf_file,
            ".docx": self._load_docx_file
        }

        for filename in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()

                if file_ext in file_extensions:
                    try:
                        docs = file_extensions[file_ext](file_path)
                        documents.extend(docs)
                        print(f"✅ Loaded: {filename}")
                    except Exception as e:
                        print(f"❌ Error loading {filename}: {e}")
                else:
                    print(f"⚠️ Unsupported file type: {filename}")

        print(f"Total documents loaded: {len(documents)}")
        return documents

    def _load_text_file(self, file_path: str) -> List[Document]:
        """Load text/markdown files with proper encoding handling"""
        try:
            # Try UTF-8 first
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        except UnicodeDecodeError:
            try:
                # Try with latin-1 encoding
                loader = TextLoader(file_path, encoding='latin-1')
                return loader.load()
            except Exception:
                # Try with system default encoding
                loader = TextLoader(file_path, encoding=None)
                return loader.load()

    def _load_pdf_file(self, file_path: str) -> List[Document]:
        """Load PDF files"""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _load_docx_file(self, file_path: str) -> List[Document]:
        """Load DOCX files"""
        loader = Docx2txtLoader(file_path)
        return loader.load()