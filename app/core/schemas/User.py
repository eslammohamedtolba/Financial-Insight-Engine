from sqlmodel import SQLModel
from typing import Optional, List
import uuid
from .Conversation import ConversationPublic

# Base schema with common user fields.
class BaseUser(SQLModel):
    email: str
    username: str
    first_name: Optional[str] = None
    second_name: Optional[str] = None

# Schema for login existing user
class LoginUser(SQLModel):
    email: str
    password: str

# Schema for creating a new user (receives a password).
class UserCreate(BaseUser):
    password: str

# Schema for reading/returning a user (never includes password).
class UserPublic(BaseUser):
    id: uuid.UUID

# Schema for a user with all their conversations.
class UserPublicWithConversations(UserPublic):
    conversations: List[ConversationPublic] = []
