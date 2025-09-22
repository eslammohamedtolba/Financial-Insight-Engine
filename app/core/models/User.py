from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy.orm import Mapped
from typing import List, TYPE_CHECKING
from datetime import datetime
import uuid

if TYPE_CHECKING:
    from .Conversation import Conversation

class User(SQLModel, table=True):
    """
    Represents the 'users' table in the database.
    """
    __tablename__ = 'users'

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    email: str = Field(unique=True, index=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    first_name: str
    second_name: str
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(
        default_factory=datetime.now,
        sa_column_kwargs={"onupdate": datetime.now}
    )

    conversations: Mapped[List["Conversation"]] = Relationship(back_populates="owner")