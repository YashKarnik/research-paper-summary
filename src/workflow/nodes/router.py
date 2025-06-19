from typing import Literal

from src.workflow.state import AgentClass
from src.workflow.utils import get_last_message


def router(state: AgentClass) -> Literal["generate_topics", "get_user_topics"]:
    """ Checks if user has given any specific topic """
    print("IN ROUTER")
    user_query = get_last_message(state).content
    print("user_query")
    print(user_query)

    if user_query == "":
        return "generate_topics"
    return "get_user_topics"
