"""
FastAPI Application Entry Point

Registers all routers, configures CORS, and sets up startup/shutdown hooks.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import engine, Base
from app.logging_config import setup_logging, get_logger
from app.api.stocks import router as stocks_router
from app.api.models import router as models_router, predict_router
from app.api.sentiment import router as sentiment_router
from app.api.backtest import router as backtest_router

# Configure structured logging before anything else
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create all tables on startup (Alembic is used for migrations in production)."""
    logger.info("Starting AI Stock Forecasting Platform", version=settings.VERSION)
    # Ensure all model directories exist
    settings.MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    settings.sample_data_dir.mkdir(parents=True, exist_ok=True)

    # Create tables (idempotent — won't drop existing tables)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "AI-powered stock forecasting and market sentiment analysis platform. "
        "Predictions are experimental and NOT financial advice."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(stocks_router)
app.include_router(models_router)
app.include_router(predict_router)
app.include_router(sentiment_router)
app.include_router(backtest_router)


@app.get("/")
def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "disclaimer": (
            "Predictions are experimental machine-learning forecasts for educational "
            "and research purposes only and are not financial advice."
        ),
    }
