from unsloth import FastLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Any, List
from .base_service import BaseService
from .state_service import QueryConstruct, Metadata
from sentence_transformers import CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from pathlib import Path
from app.helpers.settings import settings
import torch
import gc
import logging
import threading

logger = logging.getLogger(__name__)


class ModelService(BaseService):
    """
    Service for all ML model operations (LLMs, Rerankers).
    Uses threading locks to ensure thread-safe access to local models.
    """

    def __init__(self):
        self._phi3_model: Optional[Any] = None
        self._phi3_tokenizer: Optional[Any] = None
        self._gemini_llm: Optional[ChatGoogleGenerativeAI] = None
        self._reranker_model: Optional[CrossEncoder] = None
        self._phi3_model_path = str(Path("Data") / "phi3_finetuned_model")
        self._reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        
        # These are critical to prevent race conditions on the local models.
        self._phi3_lock = threading.Lock()
        self._reranker_lock = threading.Lock()

    def initialize(self) -> None:
        """Initializes all models synchronously."""
        logger.info("Initializing ModelService...")
        self._load_phi3_model_sync()
        self._load_reranker_model_sync()
        self._initialize_gemini_sync()
        if self._phi3_model:
            self._phi3_model.eval()
        logger.info("ModelService initialized successfully.")

    def _load_phi3_model_sync(self) -> None:
        if not torch.cuda.is_available():
            raise SystemError("CUDA is not available. Cannot load Phi-3 model.")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self._phi3_model_path,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        self._phi3_model = model
        self._phi3_tokenizer = tokenizer

    def _load_reranker_model_sync(self) -> None:
        self._reranker_model = CrossEncoder(self._reranker_model_name)

    def _initialize_gemini_sync(self) -> None:
        self._gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.google_api_key,
            temperature=0,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    def run_phi3_inference_sync(self, question: str, context: str, max_new_tokens: int = 512) -> str:
        """
        Runs inference in a thread-safe manner using a lock.
        """
        with self._phi3_lock:
            try:
                messages = [
                    {"role": "system", "content": "You are an expert financial analyst. Answer the user's question based only on the provided context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                ]
                prompt = self._phi3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self._phi3_tokenizer([prompt], return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = self._phi3_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        pad_token_id=self._phi3_tokenizer.eos_token_id,
                    )
                response_tokens = outputs[0][inputs.input_ids.shape[-1]:]
                decoded_response = self._phi3_tokenizer.decode(response_tokens, skip_special_tokens=True)
                return decoded_response.replace("<|end|>", "").strip()
            except Exception as e:
                logger.error(f"Error during Phi-3 inference: {e}", exc_info=True)
                return "I apologize, but I encountered an error while generating a response."


    def analyze_query_sync(self, query: str, conversation_context: str) -> QueryConstruct:
        """
        Analyzes the user query. The Gemini API is already thread-safe, so no lock is needed.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at query analysis for a financial RAG system. Your task is to analyze the user's latest query in the context of the recent conversation history. You must produce two outputs in a structured format: 'filter' and 'refined_query'. The 'filter' should extract company tickers (AAPL, MSFT, GOOG, AMZN, META) and categories ('risks', 'management_dis') from the user's latest query only. If not mentioned, set the value to null. The 'refined_query' should be a rewritten, self-contained question optimized for a vector database search, using the 'Conversation Context' to resolve pronouns or follow-up questions."),
            ("human", "Conversation Context:\n{conversation_context}\n\nUser Query: {query}")
        ])
        
        structured_llm = self._gemini_llm.with_structured_output(QueryConstruct)
        
        try:
            return structured_llm.invoke(
                prompt.format_messages(conversation_context=conversation_context, query=query)
            )
        except Exception as e:
            logger.warning(f"Structured query generation failed: {e}. Falling back to original query.")
            return QueryConstruct(filter=Metadata(), refined_query=query)

    def rerank_documents_sync(self, query: str, documents: List[str]) -> List[float]:
        """
        Reranks documents in a thread-safe manner using a lock.
        """
        with self._reranker_lock:
            try:
                pairs = [[query, doc] for doc in documents]
                scores = self._reranker_model.predict(pairs)
                return scores.tolist() if hasattr(scores, 'tolist') else scores
            except Exception as e:
                logger.error(f"Error during document reranking: {e}", exc_info=True)
                # Return a neutral score if reranking fails
                return [0.5] * len(documents)

    def cleanup(self) -> None:
        """Cleans up models and releases GPU memory."""
        self._phi3_model = None
        self._phi3_tokenizer = None
        self._gemini_llm = None
        self._reranker_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("ModelService cleaned up successfully")

