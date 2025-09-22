from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    """
    Defines the application's configuration model.
    Pydantic-settings will automatically load variables from the .env file.
    """
    # Postgresql Database Configuration
    database_url: str       
    langgraph_database_url: str

    # Redis Cache Configuration
    redis_url: str

    # Google API Configuration
    google_api_key: str

    # LangSmith Configuration
    langsmith_tracing: str
    langsmith_endpoint: AnyHttpUrl
    langsmith_api_key: str
    langsmith_project: str

    # Application settings
    PROJECT_NAME: str
    API_V1_STR: str

    # Security
    JWT_SECRET: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: str
    
    # Concurrency Control
    MAX_CONCURRENT_REQUESTS: int = 3

    class Config:
        # Specifies the name of the file to load environment variables from.
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a single, importable instance of the settings.
settings = Settings()