from typing import TypedDict

from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send

from src.workflow.state import AgentClass
from src.workflow.utils import base_llm_model

_prompt = """
    You are an AI assistant. Your job is to summarize the below topic.
    Topic: {context}
"""


class _WorkerState(TypedDict):
    topic: str


class Worker:

    def __init__(self, vector_database: FAISS):
        self._template = ChatPromptTemplate.from_template(_prompt)
        self._document_chain = create_stuff_documents_chain(base_llm_model, self._template)
        self._retrieval_chain = create_retrieval_chain(vector_database.as_retriever(), self._document_chain)

    @staticmethod
    def worker(state: AgentClass):
        pass

    @staticmethod
    def worker_orchestrator(state: AgentClass):
        topic_summarizer_task = [Send("topic_summarizer", {"topic": topic}) for topic in state['topics']]
        web_info_getter_task = [Send("web_info_getter", {"topic": topic}) for topic in state['topics']]
        return topic_summarizer_task + web_info_getter_task

    def topic_summarizer(self, state: _WorkerState):
        topic_title = state['topic']
        result = self._retrieval_chain.invoke({"input": topic_title})
        print("IN topic_summarizer")
        # return {"summary": [{"title": topic_title, "content": "random_result"}]}
        return {"summary": [{"title": topic_title, "content": result['answer']}]}

    @staticmethod
    def web_info_getter(state: _WorkerState):
        search_tool = DuckDuckGoSearchRun()
        topic_title = state['topic']
        result = ""
        try:
            result += search_tool.invoke(topic_title)
            # print("IN web_info_getter")
        except DuckDuckGoSearchException:
            result += "ERROR_RATE_LIMIT_REACHED"
        return {"web_results": [{"title": topic_title, "content": result}]}
