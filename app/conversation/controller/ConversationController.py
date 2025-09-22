from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    HarmBlockThreshold, 
                                    HarmCategory)
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.models.Conversation import Conversation
from sqlmodel import select
import uuid
from typing import List
from app.helpers.settings import settings
import asyncio
import logging
from datetime import datetime
from typing import Optional
import psycopg

logger = logging.getLogger(__name__)

class ConversationController:
    """Handles the core business logic for managing conversations."""

    def __init__(self):
        """Initializes a dedicated, lightweight LLM for title generation."""
        try:
            self._title_generation_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.google_api_key,
                temperature=0.3, # A little creativity for titles
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
                max_output_tokens=20  # Force concise responses
            )
            logger.info("ConversationController's title generation LLM initialized.")
        except Exception as e:
            self._title_generation_llm = None
            logger.error(f"Failed to initialize title generation LLM: {e}")

    def _delete_langgraph_checkpoints_sync(self, thread_id: str):
        """
        Synchronous, low-level function to delete LangGraph checkpoints using psycopg.
        This runs in a separate thread to avoid blocking the async event loop.
        """
        try:
            conn_str = str(settings.langgraph_database_url)
            with psycopg.connect(conn_str, autocommit=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                    logger.info(f"Successfully deleted LangGraph checkpoints for thread_id: {thread_id}")
        except psycopg.Error as e:
            logger.error(f"Database error during checkpoint deletion for thread {thread_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during checkpoint deletion for thread {thread_id}: {e}")

    async def delete_conversation(self, conversation_to_delete: Conversation, session: AsyncSession) -> None:
        """
        Deletes a conversation thread and its associated LangGraph messages.
        This method assumes the conversation object has already been retrieved and verified.
        """
        conv_id_str = str(conversation_to_delete.id)
        
        await session.delete(conversation_to_delete)
        await session.commit()
        logger.info(f"Deleted conversation record {conv_id_str} from main database.")

        await asyncio.to_thread(self._delete_langgraph_checkpoints_sync, conv_id_str)

    async def rename_conversation(self, conversation_to_rename: Conversation, new_name: str, session: AsyncSession) -> Conversation:
        """
        Renames a conversation thread.
        This method assumes the conversation object has already been retrieved and verified.
        """
        conversation_to_rename.name = new_name
        conversation_to_rename.updated_at = datetime.now()
        session.add(conversation_to_rename)
        await session.commit()
        await session.refresh(conversation_to_rename)
        logger.info(f"Renamed conversation {conversation_to_rename.id} to '{new_name}'.")
        
        return conversation_to_rename

    async def generate_conversation_name(self, query: str) -> Optional[str]:
        """
        Uses a dedicated Gemini model to generate a concise, 4-word name for a conversation.
        """
        if not self._title_generation_llm:
            logger.warning("Title generation LLM not available.")
            return None

        prompt = ChatPromptTemplate.from_template(
            "Create a concise title, four words maximum, for a conversation starting with this user query: '{query}'. "
            "The title should capture the main topic. Do not use quotes. Provide only the title text."
        )
        chain = prompt | self._title_generation_llm

        try:
            response = await chain.ainvoke({"query": query})
            raw_name = response.content.strip().strip('"').strip("'")

            if not raw_name:
                logger.warning("LLM generated an empty title.")
                return None

            words = raw_name.split()
            return " ".join(words[:4]) if words else None

        except Exception as e:
            logger.error(f"LLM error during title generation: {e}", exc_info=True)
            return None

    async def create_conversation(self, user_id: uuid.UUID, session: AsyncSession) -> Conversation:
        """
        Creates and returns a new empty conversation for a specific user.
        """
        new_conversation = Conversation(user_id=user_id)
        session.add(new_conversation)
        await session.commit()
        await session.refresh(new_conversation)
        return new_conversation

    async def get_conversations_by_user(self, user_id: uuid.UUID, session: AsyncSession) -> List[Conversation]:
        """
        Retrieves all conversations for a specific user from the database.
        """
        statement = select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.created_at.desc())
        results = await session.exec(statement)
        return results.all()
