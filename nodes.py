from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from state import QueryConstruct, FinancialAnalysisState, Metadata

# --- Import all initialization functions from our new config ---
from config import (
    get_gemini_llm,
    get_redis_cache,
    get_chroma_store,
    get_bm25_retriever,
    get_reranker_model,
    load_phi3_model,
    phi3_inference
)

# --- Initialize all components once when the application starts ---
gemini_llm = get_gemini_llm()
redis_cache = get_redis_cache()
chroma_vector_store = get_chroma_store()
bm25 = get_bm25_retriever()
reranker = get_reranker_model()

# Load our powerful, fine-tuned local model ONCE.
phi3_model, phi3_tokenizer = load_phi3_model()


# --- Node Definitions ---

def query_construct(state: FinancialAnalysisState):
    """Analyzes the user's message using Gemini to create a structured query."""
    messages = state['messages']
    query = messages[-1].content
    
    # Context extraction logic remains the same
    if len(messages) > 1:
        previous_messages = messages[:-1]
        context_messages = previous_messages[-2:]
        conversation_context = "\n".join(
            [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in context_messages]
        )
    else:
        conversation_context = "This is the first question from the user."
    
    # Prompt template for Gemini
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
        structured_llm = gemini_llm.with_structured_output(QueryConstruct)
        query_construct_output = structured_llm.invoke(
            prompt.format_messages(conversation_context=conversation_context, query=query)
        )
        state['structured_query'] = query_construct_output or QueryConstruct(filter=Metadata(), refined_query=query)
        
    except Exception as e:
        state['structured_query'] = QueryConstruct(filter=Metadata(), refined_query=query)
    
    return state


def check_cache(state: FinancialAnalysisState):
    """Checks the Redis cache using the refined query."""
    refined_query = state['structured_query'].refined_query
    
    results = redis_cache.similarity_search_with_score(query=refined_query, k=1)
    
    if results and (1 - abs(results[0][1]) >= 0.90):
        cached_answer = results[0][0].metadata.get("response")
        state['cache_hit'] = True
        state['messages'].append(AIMessage(content=cached_answer))
    else:
        state['cache_hit'] = False
    
    return state


def retrieve(state: FinancialAnalysisState):
    """Retrieves documents using a hybrid search approach."""
    query_text = state['structured_query'].refined_query
    metadata_filter = state['structured_query'].filter.model_dump()
    
    cleaned_filter = {k: v for k, v in metadata_filter.items() if v is not None}
    where_clause = None
    if len(cleaned_filter) > 1:
        where_clause = {"$and": [{key: {"$eq": value}} for key, value in cleaned_filter.items()]}
    elif cleaned_filter:
        key, value = next(iter(cleaned_filter.items()))
        where_clause = {key: {"$eq": value}}

    try:
        semantic_docs = chroma_vector_store.similarity_search(query=query_text, filter=where_clause, k=3)

        keyword_docs = bm25.invoke(query_text)

        combined_docs = semantic_docs + keyword_docs
        seen_contents = set()
        unique_docs = [doc for doc in combined_docs if doc.page_content not in seen_contents and not seen_contents.add(doc.page_content)]

        if unique_docs:
            rerank_pairs = [[query_text, doc.page_content] for doc in unique_docs]
            scores = reranker.predict(rerank_pairs)
            scored_docs = sorted(zip(scores, unique_docs), reverse=True)
            
            top_n = 2 
            reranked_final_docs = [doc for score, doc in scored_docs[:top_n]]
            state['source_documents'] = [doc.page_content for doc in reranked_final_docs]
        else:
            state['source_documents'] = []
            
    except Exception as e:
        state['source_documents'] = []
    
    return state


def generate_answer(state: FinancialAnalysisState):
    """Generates a final answer using the fine-tuned Phi-3 model and retrieved documents."""
    original_query = state['messages'][-1].content
    documents = state['source_documents']
    context = "\n\n".join(documents) if documents else "No relevant documents were found."
    
    try:
        answer_text = phi3_inference(
            question=original_query,
            context=context,
            model=phi3_model,
            tokenizer=phi3_tokenizer
        )
        
        if not answer_text:
            answer_message = AIMessage(content="I was unable to generate a response for this query. Please try rephrasing your question.")
        else:
            answer_message = AIMessage(content=answer_text)
            
            # Store the result in the cache
            new_doc = Document(page_content=state['structured_query'].refined_query, metadata={"response": answer_message.content})
            redis_cache.add_documents([new_doc])

        state['messages'].append(answer_message)
            
    except Exception as e:
        fallback_response = AIMessage(content=f"I apologize, but I encountered an error while processing your query. Please try again.")
        state['messages'].append(fallback_response)
    
    return state