from unsloth import FastLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from sentence_transformers import CrossEncoder
from settings import settings
import pickle
import torch

# --- Local Fine-Tuned LLM (Phi-3) ---

def load_phi3_model():
    """
    Loads the fine-tuned Phi-3 model and tokenizer from the local disk using Unsloth.
    This is a heavy operation and should only be run once.
    """
    MODEL_PATH = "Data/phi3_finetuned_model"
    
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. Cannot load the local Phi-3 model.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    
    model.eval()
    return model, tokenizer

def phi3_inference(question: str, context: str, model, tokenizer, max_new_tokens: int = 512):
    """
    Runs inference using the fine-tuned Phi-3 model by applying its native chat template.
    """
    # Create the message structure that the model was fine-tuned on.
    messages = [
        {
            "role": "system",
            "content": "You are an expert financial analyst. Answer the user's question based only on the provided context."
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        },
    ]

    # Use the tokenizer's built-in chat template for the correct, native prompt format.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    decoded_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # Clean up any potential leftover special tokens
    return decoded_response.replace("<|end|>", "").strip()


# --- Google Generative AI LLM (Gemini for Query Construction) ---

def get_gemini_llm():
    """
    Initializes and returns the ChatGoogleGenerativeAI model (Gemini).
    Used for the query construction node.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.google_api_key,
        temperature=0, # Set to 0 for deterministic, structured output
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )


# --- Embedding and Reranker Models ---

def get_embedding_model():
    """Initializes and returns the FastEmbed embedding model."""
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def get_reranker_model():
    """Initializes and returns the CrossEncoder reranking model."""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# --- Vector Stores and Retrievers ---

def get_chroma_store():
    """Initializes and returns the ChromaDB vector store for document retrieval."""
    chroma_path = "Data/chroma_db"
    return Chroma(
        collection_name="financial_filings",
        embedding_function=get_embedding_model(),
        persist_directory=chroma_path
    )

def get_bm25_retriever():
    """Loads and returns the pre-built BM25 retriever from a pickle file."""
    bm25_path = "Data/bm25_retriever.pkl"
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 3  # Set the number of results to return
    return bm25_retriever

def get_redis_cache():
    """Initializes and returns the Redis vector store for caching."""
    ttl_seconds = 3600 * 24  # Remove caching after one day

    return RedisVectorStore(
        get_embedding_model(),
        ttl=ttl_seconds,
        config=RedisConfig(
            index_name="cached_contents",
            redis_url=str(settings.redis_url),
            distance_metric="COSINE",
            metadata_schema=[
                {"name": "response", "type": "text"}
            ],
        )
    )