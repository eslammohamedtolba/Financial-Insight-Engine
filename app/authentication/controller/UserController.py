from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select
from app.core.models.User import User
from app.core.schemas.User import UserCreate
from .BaseController import BaseController
import uuid

class UserController(BaseController):
    def __init__(self):
        super().__init__()
    
    async def get_user_by_email(self, email: str, session: AsyncSession) -> User | None:
        """
        Retrieves a single user from the database based on their email.
        """ 
        statement = select(User).where(User.email == email)
        
        result = await session.exec(statement)
        return result.first()

    async def get_user_by_id(self, user_id: uuid.UUID, session: AsyncSession) -> User | None:
        """
        Retrieves a single user from the database based on their unique ID.
        This is the primary and most efficient way to fetch a user.
        """
        user = await session.get(User, user_id)
        return user

    async def user_exists(self, email: str, session: AsyncSession) -> bool:
        """
        Checks if a user with the given email already exists.
        """
        user = await self.get_user_by_email(email, session)
        return True if user is not None else False
    
    async def create_user(self, user_data: UserCreate, session: AsyncSession) -> User:
        """
        Creates a new user, hashes their password, and saves them to the database.
        """
        user_data_dict = user_data.model_dump()
        
        password = user_data_dict.pop("password")
        new_user = User(**user_data_dict)
        new_user.hashed_password = self.generate_passwd_hash(password)
        
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        
        return new_user
        
    # Not implemented yet
    async def update_user(self, id: int, session: AsyncSession):
        pass
    # Not implemented yet
    async def delete_user(self, session: AsyncSession):
        pass
