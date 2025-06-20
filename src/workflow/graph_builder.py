from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.vector_db.faiss_db import FaissVectorStore
from src.workflow.nodes.aggregator import Aggregator
from src.workflow.nodes.get_user_topics import GetUserTopics
from src.workflow.nodes.parse_and_format_topics import ParseAndFormatTopics
from src.workflow.nodes.router import router
from src.workflow.nodes.topic_generator import TopicGenerator
from src.workflow.nodes.worker import Worker
from src.workflow.state import AgentClass


class GraphBuilder:
    def __init__(self, vector_store: FaissVectorStore, index_name:str):
        # self._query: str = query
        self._vector_store: FaissVectorStore = vector_store
        self.graph_builder: StateGraph = None
        self.topic_generator = TopicGenerator(self._vector_store)
        self.get_user_topics = GetUserTopics()
        self.parse_and_format_topics = ParseAndFormatTopics()
        self.worker = Worker(self._vector_store.get_local_vector_db(index_name))
        self.aggregator = Aggregator()

    def build(self) -> CompiledStateGraph:
        print("Building graph")

        self.graph_builder = StateGraph(AgentClass)
        # Build Nodes
        self._build_nodes()

        # Build Edges
        self._build_edges()
        return self.graph_builder.compile()

    def _build_nodes(self):
        self._build_topic_generator_node()
        self._build_get_user_topics_node()
        self._build_parse_and_format_topics_node()
        self._build_worker_node()
        self._build_aggregator_node()

    def _build_topic_generator_node(self):
        self.graph_builder.add_node("generate_topics", self.topic_generator.generate_topics)
        self.graph_builder.add_node("topic_generator_worker", self.topic_generator.topic_generator_worker)

    def _build_get_user_topics_node(self):
        self.graph_builder.add_node("get_user_topics", self.get_user_topics.get_user_topics)

    def _build_parse_and_format_topics_node(self):
        self.graph_builder.add_node("parse_and_format_topics", self.parse_and_format_topics.parse_and_format_topics)

    def _build_worker_node(self):
        self.graph_builder.add_node("worker", self.worker.worker)
        self.graph_builder.add_node("topic_summarizer", self.worker.topic_summarizer)
        self.graph_builder.add_node("web_info_getter", self.worker.web_info_getter)

    def _build_aggregator_node(self):
        self.graph_builder.add_node("aggregator", self.aggregator.aggregator)

    # Build edges
    def _build_edges(self):
        self.graph_builder.add_conditional_edges(START, router,
                                                 {"generate_topics": "generate_topics",
                                                  "get_user_topics": "get_user_topics"})

        self.graph_builder.add_conditional_edges("generate_topics", self.topic_generator.topic_generator_orchestrator,
                                                 ["topic_generator_worker"])

        self.graph_builder.add_edge("topic_generator_worker", "parse_and_format_topics")
        self.graph_builder.add_edge("get_user_topics", "parse_and_format_topics")
        self.graph_builder.add_edge("parse_and_format_topics", "worker")
        self.graph_builder.add_conditional_edges("worker", self.worker.worker_orchestrator,
                                                 ["topic_summarizer", "web_info_getter"])
        self.graph_builder.add_edge("topic_summarizer", "aggregator")
        self.graph_builder.add_edge("web_info_getter", "aggregator")
        self.graph_builder.add_edge("aggregator", END)
