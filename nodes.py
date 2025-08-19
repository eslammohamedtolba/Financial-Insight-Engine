from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from config import cache_store, chroma_store, bm25_retriever, llm, reranker_model
from state import QueryConstruct, FinancialAnalysisState, Metadata
from langchain_core.prompts import ChatPromptTemplate

# Query Construction Node
def query_construct(state: FinancialAnalysisState):
    """
    Analyzes the user's message and conversation history to create a structured query.
    It extracts metadata filters and refines the user's query into a standalone question
    optimized for retrieval. This is the first step to understand user intent.
    """
    messages = state['messages']
    query = messages[-1].content
    
    # Extract the last two messages (if they exist) to provide context.
    if len(messages) > 1:
        previous_messages = messages[:-1]
        context_messages = previous_messages[-2:]
        conversation_context = "\n".join(
            [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in context_messages]
        )
    else:
        # If this is the first message, there is no context.
        conversation_context = "This is the first question from the user."
    
    # Create Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at query analysis and refinement for a financial RAG system. "
         "Your task is to analyze the user's latest query in the context of the recent conversation history. "
         "You must produce two outputs in a structured format:\n"
         "1. `filter`: Extract a metadata filter (company ticker and category) from the user's **latest query only**. "
         "   - Company tickers: AAPL, MSFT, GOOG, AMZN, META. "
         "   - Categories: 'risks', 'management_dis'. "
         "   - If not mentioned, set the value to null.\n"
         "2. `refined_query`: Rewrite the user's latest query into a clear, self-contained question. "
         "   - Use the 'Conversation Context' to resolve pronouns (e.g., 'it', 'they') or follow-up questions (e.g., 'what about for them?'). "
         "   - If the new query is on a completely different topic, ignore the context and create a standalone query. "
         "   - The refined query should be optimized for a vector database search.\n\n"
         "Examples:\n"
         "--- (Example 1: Follow-up question) ---\n"
         "Conversation Context:\nUser: What are the main risks for Apple?\nAssistant: [Provides answer about Apple's risks]\n"
         "User Query: What about for Microsoft?\n"
         "Your Output: {{'filter': {{'company': 'MSFT', 'category': 'risks'}}, 'refined_query': 'What are the main risks for Microsoft?'}}\n\n"
         "--- (Example 2: Standalone question) ---\n"
         "Conversation Context:\nUser: How was Amazon's revenue last year?\nAssistant: [Provides answer about AMZN revenue]\n"
         "User Query: Tell me about Google's management discussion.\n"
         "Your Output: {{'filter': {{'company': 'GOOG', 'category': 'management_dis'}}, 'refined_query': 'What is discussed in Google's management discussion and analysis section?'}}\n\n"
         "--- (Example 3: First question) ---\n"
         "Conversation Context:\nThis is the first question from the user.\n"
         "User Query: meta risks\n"
         "Your Output: {{'filter': {{'company': 'META', 'category': 'risks'}}, 'refined_query': 'What are the main business and operational risks for Meta?'}}"),
        ("human", "Conversation Context:\n{conversation_context}\n\nUser Query: {query}")
    ])
    
    try:
        structured_llm = llm.with_structured_output(QueryConstruct)
        
        # Generate the structured filter and refined query
        query_construct_output = structured_llm.invoke(
            prompt.format_messages(conversation_context=conversation_context, query=query)
        )
        
        if query_construct_output is None:
            # If the LLM fails to produce structured output, create a fallback.
            query_construct_output = QueryConstruct(
                filter=Metadata(),
                refined_query=query
            )
        
        # Update the state with the constructed object
        state['structured_query'] = query_construct_output
        
    except Exception as e:
        # Create a fallback QueryConstruct if structured output fails
        fallback_output = QueryConstruct(
            filter=Metadata(),
            refined_query=query
        )
        state['structured_query'] = fallback_output
    
    return state

# Check Cache Node
def check_cache(state: FinancialAnalysisState):
    """
    Checks if the refined query is in the Redis cache. This now runs *after* query construction.
    """
    # Use the self-contained, refined query for the cache check.
    refined_query = state['structured_query'].refined_query
    
    # Perform a similarity search on the cache with a threshold
    results = cache_store.similarity_search_with_score(query=refined_query, k=1)
    
    if results and (1 - abs(results[0][1]) >= 0.90):  # Using a slightly higher threshold 0.9 for refined queries
        # Cache hit, retrieve the stored answer
        cached_answer = results[0][0].metadata.get("response")
        state['cache_hit'] = True
        # Append the cached answer as a new AI message
        state['messages'].append(AIMessage(content=cached_answer))
    else:
        # No cache hit
        state['cache_hit'] = False
    
    return state

# Retriever Node
def retrieve(state: FinancialAnalysisState):
    """
    Retrieves documents using a hybrid search approach based on the refined query.
    """
    query_text = state['structured_query'].refined_query
    metadata_filter = state['structured_query'].filter.model_dump()
    
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
            rerank_pairs = [[query_text, doc.page_content] for doc in unique_docs]
            scores = reranker_model.predict(rerank_pairs)
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
    and stores the answer in the cache against the refined query.
    """
    # Use the original human query for the final prompt context
    original_query = state['messages'][-1].content
    # Use the refined query for caching
    refined_query = state['structured_query'].refined_query
    
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
        # The LLM answers the original question, but using the context retrieved by the refined query
        answer_message = chain.invoke({"context": context, "question": original_query})
        
        if not answer_message.content:
            fallback_response = AIMessage(content="I was unable to generate a response for this query, possibly due to a content filter. Please try rephrasing your question.")
            state['messages'].append(fallback_response)
        else:
            state['messages'].append(answer_message)
            
            # Store the refined query and its answer in the cache
            new_doc = Document(page_content=refined_query, metadata={"response": answer_message.content})
            cache_store.add_documents([new_doc])
            
    except Exception as e:
        fallback_response = AIMessage(content=f"I apologize, but I encountered an error while processing your query: '{original_query}'. Please try again.")
        state['messages'].append(fallback_response)
    
    return state
