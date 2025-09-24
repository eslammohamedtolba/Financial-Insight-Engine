from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from .helpers import settings
from app.assistant.controller.agent_controller import startup_event_handler, shutdown_event_handler
from app import api_router


app = FastAPI(title=settings.PROJECT_NAME)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# Register the lifecycle events
app.add_event_handler("startup", startup_event_handler)
app.add_event_handler("shutdown", shutdown_event_handler)

# Include API Routers
app.include_router(api_router)

# Mount static files (for frontend assets and icons)
web_ui_path = Path(__file__).parent.parent / "web-ui"
if web_ui_path.exists():
    # Mount CSS and JS folders
    app.mount("/css", StaticFiles(directory=str(web_ui_path / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(web_ui_path / "js")), name="js")
    app.mount("/static", StaticFiles(directory=str(web_ui_path / "static")), name="static")

@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the frontend application"""
    frontend_file = web_ui_path / "index.html"
    if frontend_file.exists():
        return FileResponse(str(frontend_file))
    else:
        return {"message": "Frontend not found. Please ensure web-ui/index.html exists."}

@app.get("/health", tags=["Health Check"])
def health_check():
    """API health check endpoint"""
    return {"status": "ok", "project": settings.PROJECT_NAME}