import random
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.workflow.state import AgentClass
from src.workflow.utils import base_llm_model, get_last_message

_prompt = """
    You are an AI assistant. You are to look at the query given below and determine if the user has
    either
        Asked for explanation on any topics
    or
        Given any topics

    and
        Respond with a list of properly formatted topics

    Eg. Query: Please explain me what is calculus and limits?
        Response: ['Calculus','Limits']

    Eg. Query: Calculus
        Response: ['Calculus']

    Query: {context}
"""


class _DocumentRelevantTopicStructuredState(BaseModel):
    topics: List[str] = Field(description="List of topics generated from the users query")


class GetUserTopics:
    _template = ChatPromptTemplate.from_template(_prompt)
    _llm_with_structured_output_for_multiple_topic_generation = base_llm_model.with_structured_output(
        _DocumentRelevantTopicStructuredState)
    _chain = _template | _llm_with_structured_output_for_multiple_topic_generation

    def get_user_topics(self, state: AgentClass):
        last_message = get_last_message(state).content
        print("IN get_user_topics")
        result = self._chain.invoke({"context": last_message})
        topics = random.sample(result.topics, k=min(len(result.topics), 3))
        return {"topics": topics}
