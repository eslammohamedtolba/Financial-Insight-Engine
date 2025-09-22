from typing import TypedDict, Annotated, Sequence, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# --- Pydantic Models for Schema-Guided Generation ---

class Metadata(BaseModel):
    """
    A Pydantic model to represent the metadata for filtering vector database queries.
    This schema is used to guide an LLM to extract structured information from a user query.
    """
    company: Optional[Literal["AAPL", "MSFT", "GOOG", "AMZN", "META"]] = Field(
        default=None,
        description="The stock ticker for a specific company to filter by. Must be one of the provided uppercase values: 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'. Leave None if no specific company is mentioned."
    )
    category: Optional[Literal["risks", "management_dis"]] = Field(
        default=None,
        description="The category of the SEC filing to search within. 'risks' for risk factors, 'management_dis' for management discussion. Leave None if no specific category is mentioned."
    )

class QueryConstruct(BaseModel):
    """
    A Pydantic model to represent the structured output for query construction.
    This is the output schema for the LLM that analyzes the user's intent and conversation history.
    """
    filter: Metadata = Field(
        default_factory=Metadata,
        description="A structured filter to be applied to the vector database query based on the user's latest query."
    )
    refined_query: str = Field(
        ...,
        description=(
            "A rewritten, self-contained query that incorporates context from the conversation. "
            "This query is optimized for vector database retrieval."
        )
    )

# --- State Definition ---

class FinancialAnalysisState(TypedDict):
    """
    Represents the state of our financial analysis RAG pipeline.
    This is passed between nodes in the LangGraph.
    """
    # Conversation history with the user.
    # The `add_messages` annotation ensures new messages are appended to the list.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # The result of the query construction node.
    # It contains the refined query and the metadata filter.
    structured_query: QueryConstruct
    
    # The documents retrieved from the vector database.
    # This is the context used by the LLM to formulate the final answer.
    source_documents: list[str]
    
    # A boolean to indicate whether a query was found in the cache.
    # This guides the conditional logic in the LangGraph.
    cache_hit: bool