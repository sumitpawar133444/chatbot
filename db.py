from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import ssl
from dotenv import load_dotenv


# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

# Get database credentials from environment
username = os.getenv('DB_USERNAME')
password = os.getenv('PASSWORD')
neon_url = os.getenv('NEON_URL')
database = os.getenv('DATABASE')

if not all([username, password, neon_url, database]):
    raise ValueError("Missing one or more environment variables for the database connection.")

# Async database URL (no sslmode in the URL)
SQLALCHEMY_DATABASE_URL = f"postgresql+asyncpg://{username}:{password}@{neon_url}/{database}"

# SSL context for asyncpg
ssl_context = ssl.create_default_context()

# Create async engine with SSL
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,
    connect_args={"ssl": ssl_context},
)

# Async sessionmaker
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for ORM models
Base = declarative_base()

# Dependency for FastAPI routes
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session  # Automatically handles session cleanup