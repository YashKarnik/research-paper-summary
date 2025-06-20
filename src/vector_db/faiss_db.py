from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import AppConfig
from src.workflow.utils import get_relevant_docs, embedding_model


class FaissVectorStore:
    def __init__(self):
        self._vector_database = None
        self._document_chunks = None
        self._document_path = None
        self._filename = None

    def load_in_memory(self, filename: str):
        print("building vector store")
        self._filename = filename
        self._document_path = AppConfig.get_file_upload_path(self._filename)
        self._document_chunks = self._chunk_documents()
        self._vector_database = self._initialize_vector_database()
        return self

    def commit_to_disk(self):
        return self._save_vector_database()

    def _chunk_documents(self) -> List[Document]:
        pdf_loader = PyPDFLoader(self._document_path)
        documents = pdf_loader.load()
        pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = pdf_text_splitter.split_documents(documents)
        return get_relevant_docs(split_docs)

    def _initialize_vector_database(self) -> FAISS:
        return FAISS.from_documents(documents=self._document_chunks,
                                    embedding=embedding_model)

    def _save_vector_database(self):
        vector_index_name = Path(self._filename).stem
        vector_db_local_path = AppConfig.get_index_upload_path(vector_index_name)
        self._vector_database.save_local(vector_db_local_path)
        return vector_index_name

    @staticmethod
    def get_local_vector_db(vector_index_name: str) -> FAISS:
        return FAISS.load_local(AppConfig.get_index_upload_path(vector_index_name),
                                embeddings=embedding_model,
                                allow_dangerous_deserialization=True)

    def get_documents(self):
        return self._document_chunks

