from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from sentence_transformers import CrossEncoder
from unsloth import FastLanguageModel
import torch
from dotenv import load_dotenv
import pickle

load_dotenv()

# --- Local Fine-Tuned LLM (Llama 3) ---

# This function loads the heavy model and should be cached in the main app
def load_llama3_model():
    """
    Loads the fine-tuned Llama 3 model and tokenizer from the local disk using Unsloth.
    """
    MODEL_PATH = "Data/complete_finetuned_model"
    
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. Cannot load the local Llama 3 model.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    
    model.eval()
    return model, tokenizer

def llama3_inference(question: str, context: str, model, tokenizer, max_new_tokens: int = 512):
    """
    Runs inference using the fine-tuned Llama 3 model with the specific prompt template it was trained on.
    This function will be imported and used in the 'generate_answer' node.
    """
    # The prompt template must exactly match the one used during fine-tuning
    ft_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Below is a user question, paired with retrieved context. Write a response that appropriately answers the question,
include specific details in your response. <|eot_id|>

<|start_header_id|>user<|end_header_id|>

### Question:
{}

### Context:
{}

<|eot_id|>

### Response: <|start_header_id|>assistant<|end_header_id|>
{}"""

    prompt = ft_prompt.format(question, context, "")
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
    
    return decoded_response.replace("<|eot_id|>", "").strip()


# --- Google Generative AI LLM (Gemini) ---

def get_gemini_llm():
    """
    Initializes and returns the ChatGoogleGenerativeAI model (Gemini).
    Used for the query construction node.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
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
    url = "redis://localhost:6379"  # docker run --name my-redis -p 6379:6379 -v redis-data:/data -d redis/redis-stack:latest
    ttl_seconds = 3600 * 24  # Remove caching after one day

    return RedisVectorStore(
        get_embedding_model(),
        ttl=ttl_seconds,
        config=RedisConfig(
            index_name="cached_contents",
            redis_url=url,
            distance_metric="COSINE",
            metadata_schema=[
                {"name": "response", "type": "text"}
            ],
        )
    )