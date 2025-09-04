from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, RedisDsn

class Settings(BaseSettings):
    """
    Defines the application's configuration model.
    Pydantic-settings will automatically load variables from the .env file.
    """
    # Postgresql Database Configuration
    database_url: PostgresDsn

    # Redis Cache Configuration
    redis_url: RedisDsn
    
    # Google API Configuration
    google_api_key: str

    class Config:
        # Specifies the name of the file to load environment variables from.
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create a single, importable instance of the settings.
settings = Settings()
