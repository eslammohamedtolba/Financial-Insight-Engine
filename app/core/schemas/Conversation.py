from sqlmodel import SQLModel
from datetime import datetime
import uuid

# Public-facing schema for returning conversation info.
class ConversationPublic(SQLModel):
    id: uuid.UUID
    name: str
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

# Schema for updating a conversation's name.
class ConversationUpdate(SQLModel):
    name: str

class ConversationMessage(SQLModel):
    role: str
    content: str