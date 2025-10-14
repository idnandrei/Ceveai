from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from core.config import settings
from logger import app_log

# Replace with your actual database URL (e.g., SQLite for local dev or PostgreSQL in prod)
DATABASE_URL = settings.DATABASE_URL  # Use PostgreSQL URI in production

# Create engine with connection verification
try:
    app_log.info("Attempting to connect to PostgreSQL database...")
    engine = create_engine(DATABASE_URL)
    # Test the connection
    with engine.connect() as connection:
        app_log.info("Successfully connected to PostgreSQL database")
except OperationalError as e:
    app_log.error(f"Failed to connect to PostgreSQL database: {str(e)}")
    app_log.error("Please ensure PostgreSQL is installed and running")
    raise

# Session local for dependency injection
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models to inherit from
Base = declarative_base()


def create_tables():
    try:
        app_log.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        app_log.info("Database tables created successfully")
    except Exception as e:
        app_log.error(f"Error creating database tables: {str(e)}")
        raise
