from fastapi import APIRouter, Depends, Path, Body, status, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession
from typing import List
import uuid

from app.db.session import get_session
from app.authentication.dependencies import get_current_user
from app.core.models.User import User
from app.core.models.Conversation import Conversation
from app.core.schemas.Conversation import ConversationPublic, ConversationUpdate
from .controller.ConversationController import ConversationController
from sqlmodel import select

conversation_router = APIRouter(
    prefix="/api/v1/conversations",
    tags=['Conversations']
)

# Create the controller
conversation_controller = ConversationController()


@conversation_router.post("", response_model=ConversationPublic, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """Creates a new, empty conversation for the authenticated user."""
    new_conversation = await conversation_controller.create_conversation(user_id=user.id, session=session)
    
    return new_conversation


@conversation_router.get("", response_model=List[ConversationPublic])
async def list_conversations(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """Retrieves all conversations for the authenticated user."""
    results = await conversation_controller.get_conversations_by_user(user_id=user.id, session=session)
    
    return results

@conversation_router.patch("/{conversation_id}", response_model=ConversationPublic)
async def update_conversation(
    conversation_id: uuid.UUID = Path(...),
    update_data: ConversationUpdate = Body(...),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """Renames a specific conversation after verifying ownership."""
    
    # Security Check Layer
    statement = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    conversation = (await session.exec(statement)).one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or permission denied.")

    return await conversation_controller.rename_conversation(
        conversation_to_rename=conversation,
        new_name=update_data.name,
        session=session
    )

@conversation_router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID = Path(...),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """Deletes a specific conversation and all its messages after verifying ownership."""
    
    # Security Check Layer
    statement = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    conversation = (await session.exec(statement)).one_or_none()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or permission denied.")
    
    await conversation_controller.delete_conversation(
        conversation_to_delete=conversation,
        session=session
    )
    return None

