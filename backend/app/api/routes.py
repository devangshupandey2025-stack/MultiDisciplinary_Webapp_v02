"""API routes for prediction, auth, and history."""
import io
import os
import time
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Header, Query, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
import httpx
import base64

from backend.app.ml.predictor import get_predictor
from backend.app.ml.gemini_validator import get_validator
from backend.app.ml.finetuner import get_finetuner
from backend.app.models.schemas import (
    PredictionResponse, HealthResponse, GeminiValidation
)
from backend.app.services.supabase_service import supabase_service

router = APIRouter()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "anushka")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
# Max chars per TTS request (bulbul:v2=1500, v3=2500)
SARVAM_MAX_CHARS = 1500


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    pace: float = 1.0


class FeedbackRequest(BaseModel):
    predicted_class: str
    actual_class: str
    is_correct: bool
    image_url: str = ""
    prediction_id: Optional[str] = None


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
    image_url = None
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
        image_url=image_url,
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


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Submit user feedback on a prediction (for model training)."""
    user_id = None
    if authorization:
        token = authorization.replace("Bearer ", "")
        user = supabase_service.verify_token(token)
        if user:
            user_id = user["id"]

    if not user_id:
        user_id = "anonymous"

    result = supabase_service.save_feedback(
        user_id=user_id,
        prediction_id=request.prediction_id,
        image_url=request.image_url,
        predicted_class=request.predicted_class,
        actual_class=request.actual_class,
        is_correct=request.is_correct,
    )
    if result.get("status") == "error":
        raise HTTPException(500, result.get("error", "Failed to save feedback"))
    return result


@router.get("/feedback/stats")
async def feedback_stats():
    """Get feedback statistics and model training status."""
    stats = supabase_service.get_feedback_stats()
    finetuner = get_finetuner()
    stats["is_training"] = finetuner.is_training
    stats["last_training"] = finetuner.last_training
    stats["can_retrain"] = stats["total"] >= 5
    return stats


@router.post("/admin/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    authorization: str = Header(...),
):
    """Trigger model fine-tuning using accumulated feedback data."""
    token = authorization.replace("Bearer ", "")
    user = supabase_service.verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")

    finetuner = get_finetuner()
    if finetuner.is_training:
        raise HTTPException(409, "Training already in progress")

    feedback_data = supabase_service.get_feedback()
    if len(feedback_data) < 5:
        raise HTTPException(400, f"Need at least 5 feedback samples, got {len(feedback_data)}")

    predictor = get_predictor()

    def run_training():
        result = finetuner.finetune(predictor, feedback_data, supabase_service)
        if result["status"] == "success":
            predictor.model.eval()
            print(f"Model fine-tuned: v{result['version']}, {result['samples_used']} samples, acc={result['final_accuracy']:.2%}")

    background_tasks.add_task(run_training)
    return {"status": "training_started", "samples_queued": len(feedback_data)}


@router.get("/admin/retrain/status")
async def retrain_status(authorization: str = Header(...)):
    """Check current training status."""
    token = authorization.replace("Bearer ", "")
    user = supabase_service.verify_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")

    finetuner = get_finetuner()
    return {
        "is_training": finetuner.is_training,
        "last_training": finetuner.last_training,
        "total_runs": len(finetuner.training_history),
    }


# Language code mapping for Sarvam TTS
LANG_TO_SARVAM = {
    "en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN",
    "bn": "bn-IN", "mr": "mr-IN", "kn": "kn-IN", "gu": "gu-IN",
}


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using Sarvam AI TTS (Indian language support)."""
    if not SARVAM_API_KEY:
        raise HTTPException(503, "TTS service not configured (SARVAM_API_KEY missing)")

    text = request.text.strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty")

    target_lang = LANG_TO_SARVAM.get(request.language, "en-IN")
    pace = max(0.5, min(2.0, request.pace))

    # Truncate text to max chars (Sarvam limit)
    if len(text) > SARVAM_MAX_CHARS:
        text = text[:SARVAM_MAX_CHARS]

    payload = {
        "text": text,
        "target_language_code": target_lang,
        "speaker": SARVAM_TTS_SPEAKER,
        "model": SARVAM_TTS_MODEL,
        "pace": pace,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                SARVAM_TTS_URL,
                json=payload,
                headers={
                    "api-subscription-key": SARVAM_API_KEY,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        audios = data.get("audios", [])
        if not audios:
            raise HTTPException(500, "No audio returned from TTS service")

        audio_bytes = base64.b64decode(audios[0])
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline"},
        )
    except httpx.HTTPStatusError as e:
        print(f"Sarvam TTS error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(502, f"TTS service error: {e.response.status_code}")
    except Exception as e:
        print(f"TTS error: {e}")
        raise HTTPException(500, "Failed to generate speech")
