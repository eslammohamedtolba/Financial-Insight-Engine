from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from db_utils import get_db_connection
from nodes import check_cache, query_construct, retrieve, generate_answer
from state import FinancialAnalysisState

# --- Define conditional edge function ---
def route_after_cache(state: FinancialAnalysisState):
    """
    Determines the next step based on whether a cache hit occurred.
    If a hit is found, the graph ends. Otherwise, it proceeds to retrieve documents.
    """
    if state['cache_hit']:
        return "end"
    else:
        return "retrieve"

# --- Define the graph ---
def create_graph():
    """
    Creates and compiles the LangGraph agent for financial analysis.
    The flow is:
    1. Construct/refine the query.
    2. Check the cache with the refined query.
    3. If cache miss, retrieve documents.
    4. Generate the final answer.
    """
    workflow = StateGraph(FinancialAnalysisState)

    # --- Add nodes ---
    workflow.add_node("query_construct", query_construct)
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # --- Add edges ---
    # The entry point is now query construction.
    workflow.set_entry_point("query_construct")

    # After constructing the query, check the cache.
    workflow.add_edge("query_construct", "check_cache")

    # Add the conditional edge after the cache check.
    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {
            "retrieve": "retrieve", 
            "end": END
        }
    )

    # Edges for the main RAG chain (if cache is missed)
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # The official checkpointer handles the connection and table setup
    checkpointer = PostgresSaver(
        conn=get_db_connection(autocommit=True)
    )
    
    app = workflow.compile(checkpointer=checkpointer)
    
    return app
