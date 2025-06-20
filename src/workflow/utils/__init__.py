from typing import List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from src.workflow.state import AgentClass


def get_last_message(state: AgentClass) -> BaseMessage:
    return state['messages'][-1]


base_llm_model = ChatOllama(model="gemma:2b")


def get_relevant_docs(documents: List[Document]):
    """Removes first and last pages"""
    # todo: fix later
    # relevant_docs = []
    # for i in range(len(documents)):
    #     if documents[i].metadata['page'] <= 1 or documents[i].metadata['page'] > len(documents) - 5:
    #         continue
    #     relevant_docs.append(documents[i])
    return documents
