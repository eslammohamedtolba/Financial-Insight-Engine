from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.helpers.settings import settings

# Create an asynchronous engine
async_engine = create_async_engine(
    str(settings.database_url),
    pool_pre_ping=True
)

# Create SQLModel async session factory (not SQLAlchemy's)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,  # This is the key change - use SQLModel's AsyncSession
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

async def get_session() -> AsyncSession: # type: ignore
    """
    An async dependency that yields a SQLModel async session.
    """
    async with AsyncSessionLocal() as session:
        yield session