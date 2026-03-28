"""
FastAPI application — Plant Disease Detection API.
MobileNetV3 classifier with Gemini AI cross-validation.

Start:
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import router
from backend.app.ml.predictor import get_predictor
from backend.app.ml.gemini_validator import get_validator
from backend.app.services.supabase_service import supabase_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load MobileNetV3 model and check Gemini API on startup."""
    print("Starting Plant Disease Detection API...")
    print(f"PORT: {os.environ.get('PORT', 'not set')}")

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

    print("✓ App startup complete, ready for requests")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Plant Disease Detection API",
    description="MobileNetV3 plant disease classifier with Gemini AI cross-validation for reliable diagnosis.",
    version="1.0.0",
    lifespan=lifespan,
)

# Trust all hosts - required for Railway healthcheck from healthcheck.railway.app
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # Allow all hosts including healthcheck.railway.app
)

# CORS - allow all origins for healthchecks and API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/health")
async def root_health():
    """Simple health check for load balancers (always returns OK if app is running)."""
    return JSONResponse(content={"status": "ok"}, status_code=200)


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
