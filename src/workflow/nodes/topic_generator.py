import random
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.vector_db.faiss_db import FaissVectorStore
from src.workflow.state import AgentClass
from src.workflow.utils import base_llm_model


class _TopicGeneratorWorkerState(TypedDict):
    document: Document


class _DocumentRelevantTopicStructuredState(BaseModel):
    topic: str = Field(description="Summarized topic name from given input")


class TopicGenerator:
    template = ChatPromptTemplate.from_template("""
        You are an AI assistant. Analyse at the content given below and generate a relevant topic name that summarizes given content.
        Content: {context}
    """)

    def __init__(self, vector_store: FaissVectorStore):
        self.vector_store = vector_store
        self.llm_with_structured_output_for_topic_generation = base_llm_model.with_structured_output(
            _DocumentRelevantTopicStructuredState)

    # def _create_structured_llm(self):
    #     self.llm_with_structured_output_for_topic_generation =

    # nodes===>

    def generate_topics(self, state: AgentClass):
        relevant_docs = self.vector_store.get_documents()
        random_docs = random.sample(relevant_docs, k=3)
        return {"documents": random_docs}

    @staticmethod
    def topic_generator_orchestrator(state: AgentClass):
        return [Send("topic_generator_worker", {"document": document}) for document in state['documents']]

    def topic_generator_worker(self, topic_generator_worker_state: _TopicGeneratorWorkerState):
        chain = self.template | self.llm_with_structured_output_for_topic_generation
        result = chain.invoke({"context": topic_generator_worker_state['document'].page_content})
        print("IN TOPIC GENERATOR WORKER")
        print(result)
        return {"topics": [result.topic]}
