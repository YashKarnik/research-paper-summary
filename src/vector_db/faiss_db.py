from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.workflow.utils import get_relevant_docs


class FaissVectorStore:
    def __init__(self, document_path: str):
        print("buildign vrcotr store")
        self._document_path = document_path
        self._relevant_docs = self._fetch_documents()
        self._vector_database = self._initialize_vector_database()

    def _fetch_documents(self) -> List[Document]:
        pdf_loader = PyPDFLoader(self._document_path)
        documents = pdf_loader.load()
        pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = pdf_text_splitter.split_documents(documents)
        return get_relevant_docs(split_docs)

    def _initialize_vector_database(self) -> FAISS:
        embedding_db = OllamaEmbeddings(model="nomic-embed-text:latest")
        return FAISS.from_documents(documents=self._relevant_docs, embedding=embedding_db)

    def get_vector_db(self) -> FAISS:
        return self._vector_database

    def get_documents(self) -> List[Document]:
        return self._relevant_docs

# fdb = FaissVectorStore('E:/agenticai/data/attention_is_all_you_need.pdf')
# print(fdb.)
