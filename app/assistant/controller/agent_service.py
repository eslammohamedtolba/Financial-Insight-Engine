from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from .state_service import FinancialAnalysisState
from .model_service import ModelService
from .retrieval_service import RetrievalService
from .cache_service import CacheService
from app.helpers import settings
import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self):
        self.workflow = None
        embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.model_service = ModelService()
        self.retrieval_service = RetrievalService(embedding_model)
        self.cache_service = CacheService(embedding_model)

    def _setup_checkpointer_tables_sync(self):
        with PostgresSaver.from_conn_string(settings.langgraph_database_url) as checkpointer:
            checkpointer.setup()
        logger.info("LangGraph checkpointer tables verified/created.")

    async def initialize_services(self):
        """
        Initializes all services. Although the services themselves are now sync,
        we run their initialization in threads to keep startup fast.
        """
        logger.info("Initializing RAG services...")
        
        # Use asyncio.to_thread to run each synchronous initialization
        init_tasks = [
            asyncio.to_thread(self.model_service.initialize),
            asyncio.to_thread(self.retrieval_service.initialize),
            asyncio.to_thread(self.cache_service.initialize),
            asyncio.to_thread(self._setup_checkpointer_tables_sync)
        ]
        await asyncio.gather(*init_tasks)

        workflow = StateGraph(FinancialAnalysisState)
        
        workflow.add_node("query_construct", self.query_construct)
        workflow.add_node("check_cache", self.check_cache)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.set_entry_point("query_construct")
        workflow.add_edge("query_construct", "check_cache")
        workflow.add_conditional_edges(
            "check_cache",
            lambda state: "hit" if state.get("cache_hit") else "miss",
            {"hit": END, "miss": "retrieve"}
        )
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        self.workflow = workflow
        logger.info("RAG services initialized successfully.")

    def query_construct(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        messages = state['messages']
        query = messages[-1].content
        conversation_context = "This is the first question from the user."
        if len(messages) > 1:
            context_messages = messages[-3:-1]
            conversation_context = "\n".join(
                [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                 for m in context_messages]
            )
        # Call the new synchronous method in the model service
        structured_query = self.model_service.analyze_query_sync(query, conversation_context)
        state['structured_query'] = structured_query
        return state

    def check_cache(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        refined_query = state['structured_query'].refined_query
        # Call the new synchronous method in the cache service
        cached_response = self.cache_service.get_cached_response_sync(refined_query)
        if cached_response:
            state['messages'].append(AIMessage(content=cached_response))
            state['cache_hit'] = True
        else:
            state['cache_hit'] = False
        return state

    def retrieve(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        refined_query = state['structured_query'].refined_query
        metadata_filter = state['structured_query'].filter.model_dump()
        # Call the new synchronous method in the retrieval service
        docs = self.retrieval_service.hybrid_search_sync(refined_query, metadata_filter)
        if not docs:
            state['source_documents'] = []
        else:
            reranked_scores = self.model_service.rerank_documents_sync(
                query=refined_query,
                documents=[doc.page_content for doc in docs]
            )
            scored_docs = sorted(zip(reranked_scores, docs), key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:2]]
            state['source_documents'] = [doc.page_content for doc in top_docs]
        return state

    def generate_answer(self, state: FinancialAnalysisState) -> FinancialAnalysisState:
        refined_query = state['structured_query'].refined_query
        context = "\n\n".join(state['source_documents']) if state['source_documents'] else "No relevant documents were found."
        original_query = state['messages'][-1].content
        answer = self.model_service.run_phi3_inference_sync(original_query, context)
        if not answer:
            answer = "I apologize, but I was unable to generate a response for this query."
        self.cache_service.add_to_cache_sync(refined_query, answer)
        state['messages'].append(AIMessage(content=answer))
        return state

    # --- Main Synchronous Process Method ---
    def process_sync(self, query: str, thread_id: uuid.UUID) -> FinancialAnalysisState:
        if not self.workflow:
            raise RuntimeError("AgentService not initialized. Call initialize_services() first.")

        state: FinancialAnalysisState = {"messages": [HumanMessage(content=query)]}
        config: dict = {"configurable": {"thread_id": str(thread_id)}}
        
        with PostgresSaver.from_conn_string(settings.langgraph_database_url) as checkpointer:
            graph = self.workflow.compile(checkpointer=checkpointer)
            final_state = graph.invoke(state, config=config)
            return final_state

    def get_conversation_history_sync(self, thread_id: str):
        if not self.workflow:
            raise RuntimeError("AgentService not initialized. Call initialize_services() first.")
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            with PostgresSaver.from_conn_string(settings.langgraph_database_url) as checkpointer:
                graph = self.workflow.compile(checkpointer=checkpointer)
                current_state = graph.get_state(config)
                
                if not current_state or not current_state.values:
                    return []
                
                return current_state.values.get("messages", [])
        except Exception as e:
            logger.error(f"Error retrieving history for thread {thread_id}: {e}")
            return []

    async def cleanup_services(self):
        # Cleanup can remain async if needed, but we'll make it sync for consistency
        def _cleanup():
            self.model_service.cleanup()
            self.retrieval_service.cleanup()
            self.cache_service.cleanup()
        
        await asyncio.to_thread(_cleanup)
        logger.info("RAG services cleanup complete.")




