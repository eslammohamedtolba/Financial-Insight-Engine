from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy.orm import Mapped
from typing import TYPE_CHECKING
from datetime import datetime
import uuid

if TYPE_CHECKING:
    from .User import User

class Conversation(SQLModel, table=True):
    """
    Represents the 'conversations' table in the database.
    """
    __tablename__ = 'conversations'
    
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    name: str = Field(default="New Conversation", index=True, nullable=False)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True)
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(
        default_factory=datetime.now,
        sa_column_kwargs={"onupdate": datetime.now}
    )
    
    owner: Mapped["User"] = Relationship(back_populates="conversations")