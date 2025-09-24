from typing import Optional
from langchain_core.documents import Document
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import logging
from .base_service import BaseService
from app.helpers import settings

logger = logging.getLogger(__name__)

class CacheService(BaseService):
    def __init__(self, embedding_model: FastEmbedEmbeddings):
        self._embedding_model = embedding_model
        self._redis_store: Optional[RedisVectorStore] = None
        self._cache_hit_threshold = 0.90
        self._ttl_seconds = 3600 * 24

    def initialize(self) -> None:
        """Initializes the Redis store synchronously."""
        self._redis_store = RedisVectorStore(
            embeddings=self._embedding_model,
            ttl=self._ttl_seconds,
            config=RedisConfig(
                index_name="cached_contents",
                redis_url=settings.redis_url,
                distance_metric="COSINE",
                metadata_schema=[{"name": "response", "type": "text"}],
            )
        )
        logger.info("CacheService initialized successfully")

    def get_cached_response_sync(self, query: str) -> Optional[str]:
        """Retrieves a cached response synchronously."""
        search_results = self._redis_store.similarity_search_with_score(query=query, k=1)
        
        if not search_results:
            return None
        
        similarity_score = 1 - search_results[0][1]
        
        if similarity_score >= self._cache_hit_threshold:
            return search_results[0][0].metadata.get("response")
        
        return None

    def add_to_cache_sync(self, query: str, response: str) -> None:
        """Adds a new query-response pair to the cache synchronously."""
        doc = Document(page_content=query, metadata={"response": response})
        self._redis_store.add_documents([doc])
        logger.info(f"Cached response for query: {query[:50]}...")

    def cleanup(self) -> None:
        self._redis_store = None
        logger.info("CacheService cleaned up successfully")
