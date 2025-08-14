from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- Large Language Model ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7
)

# --- Embeddings Model ---
# A single embedding model is used for both the ChromaDB vector store
# and the Redis cache to ensure vector consistency.
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# --- Vector Store for Retrieval (ChromaDB) ---
chroma_path = "Data/chroma_db"

chroma_store = Chroma(
    collection_name="financial_filings",
    embedding_function=embeddings,
    persist_directory=chroma_path
)


# --- Vector Store for Caching (Redis) ---
url = "redis://localhost:6379" # docker run --name my-redis -p 6379:6379 -v redis-data:/data -d redis/redis-stack:latest
ttl_seconds = 3600 * 24  # Remove caching after one day

# Create Redis vector store instance
cache_store = RedisVectorStore(
    embeddings,
    ttl=ttl_seconds,
    config=RedisConfig(
        index_name="cached_contents",   # Custom index name
        redis_url=url,
        distance_metric="COSINE",       # For semantic similarity
        metadata_schema=[               # Schema for storing metadata (like LLM response)
            {"name": "response", "type": "text"}
        ],
    )
)