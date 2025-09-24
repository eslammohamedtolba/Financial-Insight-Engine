from fastapi import APIRouter

from app.authentication import auth_router
from app.assistant import chat_router
from app.conversation import conversation_router

# Create a single main router to include all module-specific routers
api_router = APIRouter()

# Include routers from all modules
api_router.include_router(auth_router)
api_router.include_router(chat_router)
api_router.include_router(conversation_router)