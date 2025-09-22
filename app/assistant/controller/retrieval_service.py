from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from pathlib import Path
import pickle
import logging
from .base_service import BaseService

logger = logging.getLogger(__name__)

class RetrievalService(BaseService):
    def __init__(self, embedding_model: FastEmbedEmbeddings):
        self._embedding_model = embedding_model
        self._chroma_store: Optional[Chroma] = None
        self._bm25_retriever: Optional[Any] = None
        self._chroma_path = str(Path("Data") / "chroma_db")
        self._bm25_path = str(Path("Data") / "bm25_retriever.pkl")

    def initialize(self) -> None:
        """Initializes Chroma and BM25 synchronously."""
        self._chroma_store = Chroma(
            collection_name="financial_filings",
            embedding_function=self._embedding_model,
            persist_directory=self._chroma_path
        )
        with open(self._bm25_path, "rb") as f:
            self._bm25_retriever = pickle.load(f)
            self._bm25_retriever.k = 3
        logger.info("RetrievalService initialized successfully")

    def hybrid_search_sync(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Performs a hybrid search synchronously."""
        semantic_docs = self._semantic_search_sync(query, metadata_filter)
        keyword_docs = self._keyword_search_sync(query)
        
        unique_docs = self._deduplicate_documents(semantic_docs + keyword_docs)
        logger.info(f"Hybrid search returned {len(unique_docs)} unique documents.")
        return unique_docs

    def _semantic_search_sync(self, query: str, metadata_filter: Optional[Dict[str, Any]]) -> List[Document]:
        where_clause = self._build_where_clause(metadata_filter)
        return self._chroma_store.similarity_search(query=query, filter=where_clause, k=3)

    def _keyword_search_sync(self, query: str) -> List[Document]:
        return self._bm25_retriever.invoke(query)

    def _build_where_clause(self, metadata_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # This helper function remains the same
        if not metadata_filter:
            return None
        cleaned_filter = {k: v for k, v in metadata_filter.items() if v is not None}
        if not cleaned_filter:
            return None
        if len(cleaned_filter) > 1:
            return {"$and": [{key: {"$eq": value}} for key, value in cleaned_filter.items()]}
        else:
            key, value = next(iter(cleaned_filter.items()))
            return {key: {"$eq": value}}

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        # This helper function remains the same
        seen_contents = set()
        unique_docs = []
        for doc in documents:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs

    def cleanup(self) -> None:
        self._chroma_store = None
        self._bm25_retriever = None
        logger.info("RetrievalService cleaned up successfully")
