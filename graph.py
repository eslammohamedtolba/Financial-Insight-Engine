from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from db_utils import get_db_connection
from nodes import check_cache, query_construct, retrieve, generate_answer
from state import FinancialAnalysisState

# --- Define conditional edge function ---
def route_after_cache(state: FinancialAnalysisState):
    """
    Determines the next step based on whether a cache hit occurred.
    """
    if state['cache_hit']:
        return "end"
    else:
        return "query_construct"

# --- Define the graph ---
def create_graph():
    workflow = StateGraph(FinancialAnalysisState)

    # --- Add nodes ---
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("query_construct", query_construct)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # --- Add edges ---
    # connect the entry point to the cache check
    workflow.set_entry_point("check_cache")

    # Conditional edge from the cache check node
    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {
            "query_construct": "query_construct",
            "end": END
        }
    )

    # Edges for the main RAG chain
    workflow.add_edge("query_construct", "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # The official checkpointer handles the connection and table setup
    checkpointer = PostgresSaver(
        conn=get_db_connection(autocommit=True)
    )
    
    return workflow.compile(checkpointer=checkpointer)

