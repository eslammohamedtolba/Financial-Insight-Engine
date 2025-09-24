from fastapi import APIRouter, Depends, Path, HTTPException, status, Body
from langchain_core.messages import HumanMessage, AIMessage
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from typing import List
import uuid
import logging
import asyncio

from app.core.schemas import ConversationMessage
from app.core.models import User, Conversation
from app.authentication.dependencies import get_current_user
from app.db import get_session
from .controller import agent_service
from app.conversation.controller import ConversationController

logger = logging.getLogger(__name__)

chat_router = APIRouter(
    prefix="/api/v1/agent",
    tags=['Assistant']
)

# Initialize conversation controller
conversation_controller = ConversationController()

@chat_router.post("/chat/{conversation_id}", response_model=dict)
async def chat(
    query: str = Body(..., embed=True),
    conversation_id: uuid.UUID = Path(...),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Handles a chat interaction by running the synchronous RAG pipeline in a thread.
    """
    # Security Check to ensure the user owns the conversation
    statement = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    conversation = (await session.exec(statement)).one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or permission denied.")
    
    # Check if conversation name needs to be generated
    needs_name_generation = conversation.name == "New Conversation"
    
    # Offload the entire synchronous process to a worker thread to keep the server responsive
    final_state = await asyncio.to_thread(
        agent_service.process_sync,
        query=query,
        thread_id=conversation_id
    )
    
    assistant_response = final_state['messages'][-1]
    
    # Generate conversation name if needed
    if needs_name_generation:
        try:
            generated_name = await conversation_controller.generate_conversation_name(query)
            if generated_name:
                await conversation_controller.rename_conversation(
                    conversation_to_rename=conversation,
                    new_name=generated_name,
                    session=session
                )
                logger.info(f"Auto-generated name '{generated_name}' for conversation {conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to generate conversation name: {e}")
            # Continue without name generation - it's not critical
    
    if isinstance(assistant_response, AIMessage):
        return {"response": assistant_response.content}
    else:
        # Handle cases where the last message might not be from the AI (e.g., error state)
        return {"response": "An unexpected error occurred."}

@chat_router.get("/messages/{conversation_id}", response_model=List[ConversationMessage])
async def get_messages(
    conversation_id: uuid.UUID = Path(...),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieves conversation history by running the synchronous method in a thread.
    """
    # Security Check
    statement = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    if not (await session.exec(statement)).one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found or permission denied.")

    # Offload the synchronous history retrieval to a worker thread
    langchain_messages = await asyncio.to_thread(
        agent_service.get_conversation_history_sync,
        thread_id=str(conversation_id)
    )

    if not langchain_messages:
        return []

    # Convert the LangChain message objects to our Pydantic API response models
    response_messages = [
        ConversationMessage(
            role='user' if isinstance(msg, HumanMessage) else 'assistant',
            content=msg.content
        )
        for msg in langchain_messages
    ]
    
    return response_messages