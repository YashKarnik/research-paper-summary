import operator
from typing import TypedDict, List, Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class _TopicInformation(TypedDict):
    title: str
    content: str


class AgentClass(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: List[Document]
    topics: Annotated[List[str], operator.add]
    summary: Annotated[List[_TopicInformation], operator.add]
    web_results: Annotated[List[_TopicInformation], operator.add]

