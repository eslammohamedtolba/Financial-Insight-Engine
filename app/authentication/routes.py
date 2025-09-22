from fastapi import APIRouter, HTTPException, Depends, status
from app.helpers.settings import settings
from .controller.UserController import UserController
from app.core.schemas.User import UserCreate, LoginUser, UserPublic
from app.core.schemas.Authentication import Token
from app.db.session import get_session
from sqlmodel.ext.asyncio.session import AsyncSession
from datetime import timedelta

auth_router = APIRouter(
    prefix="/api/v1/auth",
    tags=['Authentication']
)

user_controller = UserController()


@auth_router.post('/register', response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(
        user_data: UserCreate, 
        session: AsyncSession = Depends(get_session)
    ):
    exists = await user_controller.user_exists(user_data.email, session)

    if exists:
        raise HTTPException(
            detail="User with this email already exists", 
            status_code=status.HTTP_409_CONFLICT
        )
    
    new_user = await user_controller.create_user(user_data, session)
    return new_user


@auth_router.post('/login', response_model=Token)
async def login_user(
        user_data: LoginUser, 
        session: AsyncSession = Depends(get_session)
    ):
    user = await user_controller.get_user_by_email(user_data.email, session)

    if not user or not user_controller.verify_passwd(password=user_data.password, hash=user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=int(settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    access_token = UserController.create_secret_token(
        user_data=str(user.id),
        expiry=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@auth_router.post('/logout')
async def logout_user():
    """
    Logs out the user by instructing the client to remove the JWT.

    This endpoint provides a standard RESTful interface for logout.
    With stateless JWTs, true logout is a client-side responsibility (deleting the token).

    For a more secure production environment, this endpoint should be extended
    to add the token's unique identifier (jti) to a Redis "blocklist" with a TTL
    matching the token's expiry. The `get_current_user` dependency would then
    be updated to check this blocklist on every request.
    """
    return {"message": "Successfully logged out."}