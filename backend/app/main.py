"""
FastAPI application — Plant Disease Detection API.
MobileNetV3 classifier with Gemini AI cross-validation.

Start:
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.ml.predictor import get_predictor
from backend.app.ml.gemini_validator import get_validator
from backend.app.services.supabase_service import supabase_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load MobileNetV3 model and check Gemini API on startup."""
    print("Starting Plant Disease Detection API...")

    # Initialize Supabase
    supabase_service.initialize()

    # Load MobileNetV3
    predictor = get_predictor()
    models_dir = os.getenv("MODELS_DIR", "checkpoints")

    if os.path.exists(models_dir):
        try:
            predictor.load(models_dir)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("   API will start but predictions won't work until model is available.")
    else:
        print(f"⚠️  Models directory not found: {models_dir}")
        print("   API will start but predictions won't work until models are loaded.")

    # Check Gemini API availability
    validator = get_validator()
    validator.check_availability()
    if not validator.is_available:
        print("⚠️  Gemini API not available — predictions will work without cross-validation.")
        print("   Check your GEMINI_API_KEY in .env")

    yield
    print("Shutting down...")


app = FastAPI(
    title="Plant Disease Detection API",
    description="MobileNetV3 plant disease classifier with Gemini AI cross-validation for reliable diagnosis.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "Plant Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# Optional: Prometheus metrics
try:
    from prometheus_client import make_asgi_app
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
except ImportError:
    pass

# Optional: Sentry
try:
    import sentry_sdk
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)
except ImportError:
    pass
