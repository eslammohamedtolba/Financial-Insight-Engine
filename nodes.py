from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from config import cache_store, chroma_store, llm
from state import QueryConstruct, FinancialAnalysisState, Metadata
from langchain_core.prompts import ChatPromptTemplate

# --- Node 1: Check Cache ---
def check_cache(state: FinancialAnalysisState):
    """
    Checks if the user's query is in the Redis cache.
    """
    print("---NODE: CHECK CACHE---")
    print(f"Length of messages: {len(state['messages'])}\n")
    # Get the last human message
    query = state['messages'][-1].content
    
    # Perform a similarity search on the cache with a threshold
    # Note: `similarity_search_with_score` returns a list of tuples (document, score)
    results = cache_store.similarity_search_with_score(query=query, k=1)
    
    print(f"The results: {results}\n\n")
    
    if results and (1 - results[0][1] >= 0.9):  # Check for a cache hit with a 0.9 similarity threshold
        # Cache hit, retrieve the stored answer
        cached_answer = results[0][0].metadata.get("response")
        state['cache_hit'] = True
        state['messages'].append(AIMessage(content=cached_answer))
        print("Cache hit detected.")
    else:
        # No cache hit
        state['cache_hit'] = False
        print("No cache hit.")
    
    return state

# --- Node 2: Query Construction ---
def query_construct(state: FinancialAnalysisState):
    """
    Uses an LLM to extract a search query and a metadata filter from the user's message.
    """
    print("---NODE: QUERY CONSTRUCTION---")
    query = state['messages'][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert at extracting a structured metadata filter from a user's question about financial filings. "
         "Analyze the user's query and extract the company ticker and category if they are explicitly mentioned. "
         "Company tickers: AAPL (Apple), MSFT (Microsoft), GOOG (Google), AMZN (Amazon), META (Meta/Facebook). "
         "Categories: 'risks' for risk factors, 'management_dis' for management discussion and analysis. "
         "If a company or category is not explicitly mentioned or cannot be determined, set it to null. "
         "Examples:\n"
         "- 'What are the risks for Apple?' -> company: 'AAPL', category: 'risks'\n"
         "- 'Tell me about management discussion for Google' -> company: 'GOOG', category: 'management_dis'\n"
         "- 'How has Amazon performed?' -> company: 'AMZN', category: null\n"
         "- 'What are the risks of tech companies?' -> company: null, category: 'risks'\n"
         "- 'Tell me about financial performance' -> company: null, category: null"),
        ("human", "User query: {query}")
    ])
    
    try:
        structured_llm = llm.with_structured_output(QueryConstruct)
        
        # Generate the structured filter
        query_construct_output = structured_llm.invoke(prompt.format_messages(query=query))
        
        # If the output is None or invalid, create a default QueryConstruct
        if query_construct_output is None:
            print("LLM returned None, creating default QueryConstruct")
            query_construct_output = QueryConstruct(filter=Metadata())
        
        # Update the state with the constructed filter
        state['query_construction'] = query_construct_output
        
        print(f"Constructed Filter: {query_construct_output.filter.model_dump()}")
        
    except Exception as e:
        print(f"Error in query construction: {e}")
        print("Creating default QueryConstruct")
        # Create a fallback QueryConstruct if structured output fails
        state['query_construction'] = QueryConstruct(filter=Metadata())
        print(f"Default Filter: {state['query_construction'].filter.model_dump()}")
    
    return state

# --- Node 3: Retriever ---
def retrieve(state: FinancialAnalysisState):
    """
    Retrieves documents from the ChromaDB vector store based on the constructed query and filter.
    """
    print("---NODE: RETRIEVER---")
    query_text = state['messages'][-1].content
    metadata_filter = state['query_construction'].filter.model_dump()
    
    # Remove filters that have None values
    cleaned_filter = {k: v for k, v in metadata_filter.items() if v is not None}
    
    # Perform a filtered similarity search
    if len(cleaned_filter) > 1:
        where_clause = {"$and": [{key: {"$eq": value}} for key, value in cleaned_filter.items()]}
    elif len(cleaned_filter) == 1:
        key, value = next(iter(cleaned_filter.items()))
        where_clause = {key: {"$eq": value}}
    else:
        where_clause = None  # No filter if no valid metadata is found

    print(f"Retrieving with filter: {where_clause}")

    try:
        # Use 'filter' parameter instead of 'where'
        retrieved_docs = chroma_store.similarity_search(
            query=query_text,
            filter=where_clause,
            k=3  # Retrieve top 3 documents
        )
        
        state['source_documents'] = [doc.page_content for doc in retrieved_docs]
        print(f"Retrieved {len(retrieved_docs)} documents.")
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        # If retrieval fails, set empty documents
        state['source_documents'] = []
        print("Set empty document list due to retrieval error.")
    
    return state

# --- Node 4: Answer Question and Store in Cache ---
def generate_answer(state: FinancialAnalysisState):
    """
    Generates a final answer using the LLM and retrieved documents as context,
    and stores the answer in the cache.
    """
    print("---NODE: GENERATE ANSWER & CACHE---")
    # Get the original human query
    original_query = state['messages'][-1].content
    
    # Get the retrieved documents for context
    documents = state['source_documents']
    
    if documents:
        context = "\n\n".join(documents)
        print(f"Using context from {len(documents)} documents.")
    else:
        context = "No relevant documents were found."
        print("No documents available for context.")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant that answers questions based on provided documents. "
         "Your answer must be concise, accurate, and directly reference the provided documents when available. "
         "If the documents do not contain the answer, politely state that you cannot find the information in the available documents. "
         "\n\nContext:\n{context}"),
        ("human", "Question: {question}")
    ])
    
    try:
        chain = prompt | llm
        answer_message = chain.invoke({"context": context, "question": original_query})
        
        # Update messages with the AI's response
        state['messages'].append(answer_message)
        
        # Store the query and answer in the cache for future use
        new_doc = Document(page_content=original_query, metadata={"response": answer_message.content})
        cache_store.add_documents([new_doc])
        
        print("Answer generated and stored in cache.")
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        # Create a fallback response
        fallback_response = AIMessage(content=f"I apologize, but I encountered an error while processing your query: '{original_query}'. Please try again.")
        state['messages'].append(fallback_response)
        print("Added fallback response due to error.")
    
    return state