"""API routes for prediction, auth, and history."""
import io
import os
import time
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Header, Query
from PIL import Image

from backend.app.ml.predictor import get_predictor
from backend.app.ml.gemini_validator import get_validator
from backend.app.models.schemas import (
    PredictionResponse, HealthResponse, GeminiValidation
)
from backend.app.services.supabase_service import supabase_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    predictor = get_predictor()
    validator = get_validator()
    return HealthResponse(
        status="healthy" if predictor._loaded else "models_not_loaded",
        model_loaded=predictor._loaded,
        gemini_available=validator.is_available,
        device=str(predictor.device),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=38),
    authorization: Optional[str] = Header(default=None),
):
    """
    Predict plant disease from leaf image using MobileNetV3.
    Includes Gemini AI cross-validation with visual analysis of the leaf image.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG, PNG, etc.)")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    predictor = get_predictor()
    if not predictor._loaded:
        raise HTTPException(503, "Model not loaded yet")

    start = time.time()
    result = predictor.predict(image, top_k=top_k)
    image_metadata = result.pop("image_metadata", {})
    latency_ms = (time.time() - start) * 1000
    print(f"MobileNet inference: {latency_ms:.0f}ms")

    # Run Gemini validation synchronously (multimodal: image + text)
    gemini_validation = None
    validator = get_validator()
    if validator.is_available:
        gemini_start = time.time()
        validation_result = await validator.validate_async(
            dict(result), dict(image_metadata), image_bytes=contents
        )
        gemini_ms = (time.time() - gemini_start) * 1000
        print(f"Gemini validation: {gemini_ms:.0f}ms")
        if validation_result:
            gemini_validation = GeminiValidation(**validation_result)

    # Save to history if authenticated
    if authorization and supabase_service.is_available:
        token = authorization.replace("Bearer ", "")
        user = supabase_service.verify_token(token)
        if user:
            image_url = supabase_service.upload_image(
                contents, file.filename or "image.jpg", user["id"]
            )
            supabase_service.save_prediction(
                user["id"], result, image_url,
                gemini_validation=gemini_validation.model_dump() if gemini_validation else None
            )

    return PredictionResponse(
        **{"class": result["class"]},
        probability=result["probability"],
        uncertainty=result["uncertainty"],
        top_k=[{"class": p["class"], "probability": p["probability"]} for p in result["top_k"]],
        gemini_validation=gemini_validation,
    )


@router.post("/auth/signup")
async def signup(email: str, password: str):
    result = supabase_service.sign_up(email, password)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/auth/signin")
async def signin(email: str, password: str):
    result = supabase_service.sign_in(email, password)
    if "error" in result:
        raise HTTPException(401, result["error"])
    return result


@router.get("/history")
async def get_history(
    authorization: str = Header(...),
    limit: int = Query(default=20, ge=1, le=100),
):
    token = authorization.replace("Bearer ", "")
    user = supabase_service.verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")
    history = supabase_service.get_history(user["id"], limit)
    return {"predictions": history}


@router.get("/stats")
async def get_stats(authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    user = supabase_service.verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")
    stats = supabase_service.get_stats(user["id"])
    return stats


@router.get("/classes")
async def get_classes():
    """Return list of supported disease classes."""
    predictor = get_predictor()
    return {"classes": predictor.class_names, "count": len(predictor.class_names)}
