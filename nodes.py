from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from config import cache_store, chroma_store, bm25_retriever, llm, reranker_model
from state import QueryConstruct, FinancialAnalysisState, Metadata
from langchain_core.prompts import ChatPromptTemplate

# Check Cache Node
def check_cache(state: FinancialAnalysisState):
    """
    Checks if the user's query is in the Redis cache.
    """
    # Get the last human message
    query = state['messages'][-1].content
    
    # Perform a similarity search on the cache with a threshold
    # Note: `similarity_search_with_score` returns a list of tuples (document, score)
    results = cache_store.similarity_search_with_score(query=query, k=1)
    
    if results and (1 - abs(results[0][1]) >= 0.85):  # Check for a cache hit with a 0.85 similarity threshold
        # Cache hit, retrieve the stored answer
        cached_answer = results[0][0].metadata.get("response")
        state['cache_hit'] = True
        state['messages'].append(AIMessage(content=cached_answer))
    else:
        # No cache hit
        state['cache_hit'] = False
    
    return state

# Query Construction Node
def query_construct(state: FinancialAnalysisState):
    """
    Uses an LLM to extract a search query and a metadata filter from the user's message.
    """
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
            query_construct_output = QueryConstruct(filter=Metadata())
        
        # Update the state with the constructed filter
        state['query_construction'] = query_construct_output
        
    except Exception as e:
        # Create a fallback QueryConstruct if structured output fails
        state['query_construction'] = QueryConstruct(filter=Metadata())
    
    return state

# Retriever Node
def retrieve(state: FinancialAnalysisState):
    """
    Retrieves documents using a hybrid search approach, then reranks them for relevance.
    1. Performs a filtered semantic search with ChromaDB.
    2. Performs a keyword search with BM25.
    3. Combines and de-duplicates the results.
    4. Reranks the unique documents using a Cross-Encoder model.
    5. Selects the top N documents to create a rich, focused context.
    """

    query_text = state['messages'][-1].content
    metadata_filter = state['query_construction'].filter.model_dump()
    
    # Build the filter for ChromaDB
    cleaned_filter = {k: v for k, v in metadata_filter.items() if v is not None}
    if len(cleaned_filter) > 1:
        where_clause = {"$and": [{key: {"$eq": value}} for key, value in cleaned_filter.items()]}
    elif len(cleaned_filter) == 1:
        key, value = next(iter(cleaned_filter.items()))
        where_clause = {key: {"$eq": value}}
    else:
        where_clause = None

    try:
        # Get documents from semantic search (ChromaDB)
        semantic_docs = chroma_store.similarity_search(
            query=query_text,
            filter=where_clause,
            k=3
        )

        # Get documents from keyword search (BM25)
        keyword_docs = bm25_retriever.invoke(query_text)

        # Combine and de-duplicate the results
        combined_docs = semantic_docs + keyword_docs
        seen_contents = set()
        unique_docs = []
        for doc in combined_docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)

        # Reranking documents
        if unique_docs:
            # Create pairs of [query, document_content] for the reranker model
            rerank_pairs = [[query_text, doc.page_content] for doc in unique_docs]
            
            # Get relevance scores from the model
            scores = reranker_model.predict(rerank_pairs)
            
            # Combine documents with their scores and sort
            scored_docs = list(zip(scores, unique_docs))
            scored_docs.sort(reverse=True)

            # Select the top 2 documents for the final context
            top_n = 2
            reranked_final_docs = [doc for score, doc in scored_docs[:top_n]]
            state['source_documents'] = [doc.page_content for doc in reranked_final_docs]
        else:
            state['source_documents'] = []
            
    except Exception as e:
        state['source_documents'] = []
    
    return state

# Answer Question Node
def generate_answer(state: FinancialAnalysisState):
    """
    Generates a final answer using the LLM and retrieved documents as context,
    and stores the answer in the cache.
    """
    # Get the original human query
    original_query = state['messages'][-1].content
    
    # Get the retrieved documents for context
    documents = state['source_documents']
    
    if documents:
        context = "\n\n".join(documents)
    else:
        context = "No relevant documents were found."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert Financial Analyst Assistant. Your purpose is to answer user questions about the financial performance and risks of major tech companies, using excerpts from their latest 10-K filings."
         "\n\n"
         "Follow these instructions carefully:\n"
         "1. Analyze the provided 'Context' section, which contains relevant text from the filings.\n"
         "2. Synthesize the information to form a concise, accurate, and insightful answer.\n"
         "3. Adopt a professional and direct tone. Answer the question as if you have the knowledge yourself.\n"
         "4. **Crucially, DO NOT** start your response with phrases like 'Based on the documents provided,' or 'According to the context.' The user already knows the information comes from these documents.\n"
         "5. If the context does not contain the information needed to answer the question, state that you cannot find the relevant details in the available filings.\n"
         "\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])
    
    try:
        chain = prompt | llm
        answer_message = chain.invoke({"context": context, "question": original_query})
        
        # Check if the response from the LLM is empty (due to safety filters, etc.)
        if not answer_message.content:
            fallback_response = AIMessage(content="I was unable to generate a response for this query, possibly due to a content filter. Please try rephrasing your question.")
            state['messages'].append(fallback_response)
        else:
            # Update messages with the AI's response
            state['messages'].append(answer_message)
            
            # Store the query and answer in the cache for future use
            new_doc = Document(page_content=original_query, metadata={"response": answer_message.content})
            cache_store.add_documents([new_doc])
        
    except Exception as e:
        # Create a fallback response
        fallback_response = AIMessage(content=f"I apologize, but I encountered an error while processing your query: '{original_query}'. Please try again.")
        state['messages'].append(fallback_response)
    
    return state