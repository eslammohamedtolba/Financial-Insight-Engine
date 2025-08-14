# graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3 

from nodes import check_cache, query_construct, retrieve, generate_answer
from state import FinancialAnalysisState

# --- Define the graph ---

def create_graph():
    workflow = StateGraph(FinancialAnalysisState)

    # --- Add nodes ---
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("query_construct", query_construct)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # --- Define conditional edge function ---
    def route_after_cache(state: FinancialAnalysisState):
        """
        Determines the next step based on whether a cache hit occurred.
        """
        if state['cache_hit']:
            print("---ROUTE: Cache hit, ending workflow.---")
            return "end"
        else:
            print("---ROUTE: No cache hit, proceeding to query construction.---")
            return "query_construct"

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

    # --- Compile the graph with memory ---
    sqlite_conn = sqlite3.connect("Data\\graph_memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn = sqlite_conn)
    
    return workflow.compile(checkpointer=memory)

