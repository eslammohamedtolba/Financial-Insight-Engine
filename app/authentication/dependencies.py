from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlmodel.ext.asyncio.session import AsyncSession
from jose import jwt, JWTError
import uuid

from app.db.session import get_session
from app.helpers.settings import settings
from app.core.models.User import User
from app.core.schemas.Authentication import TokenData
from .controller.UserController import UserController

reusable_oauth2 = HTTPBearer(
    scheme_name="Authorization"
)

async def get_current_user(
    token: str = Depends(reusable_oauth2), 
    session: AsyncSession = Depends(get_session)
) -> User:
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token.credentials, 
            settings.JWT_SECRET, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Extract the 'sub' claim
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
            
        # Validate it with our Pydantic schema
        token_data = TokenData(sub=user_id_str)

    except (JWTError, ValueError): # Catch JWT errors and potential validation errors
        raise credentials_exception

    user = await UserController().get_user_by_id(user_id=uuid.UUID(token_data.sub), session=session)
    
    if user is None:
        raise credentials_exception
        
    return user
