import logging
from .agent_service import AgentService

logger = logging.getLogger(__name__)

# Singleton Instance
agent_service = AgentService()

# --- FastAPI Event Handlers ---
async def startup_event_handler():
    logger.info("Application startup: Initializing AgentService...")
    await agent_service.initialize_services()
    logger.info("AgentService initialized successfully.")

async def shutdown_event_handler():
    logger.info("Application shutdown: Cleaning up AgentService components...")
    await agent_service.cleanup_services() 
    logger.info("AgentService cleanup complete.")