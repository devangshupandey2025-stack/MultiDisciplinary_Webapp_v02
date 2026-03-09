"""Pydantic models for API request/response schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str = Field(..., alias="class", description="Disease class name")
    probability: float = Field(..., ge=0, le=1, description="Calibrated probability")

    class Config:
        populate_by_name = True


class GeminiValidation(BaseModel):
    """Gemini AI cross-validation report for MobileNet predictions."""
    agrees: bool = Field(..., description="Whether Gemini agrees with MobileNet prediction")
    confidence_assessment: str = Field(..., description="Gemini's assessment of prediction confidence")
    reasoning: str = Field(..., description="Explanation based on visual inspection of the leaf image")
    alternative_suggestions: List[str] = Field(default_factory=list, description="Alternative diseases to consider")
    treatment_advice: str = Field(default="", description="Recommended treatment or next steps")
    summary: str = Field(..., description="Brief one-line validation summary")

    class Config:
        json_schema_extra = {
            "example": {
                "agrees": True,
                "confidence_assessment": "High — the visible symptoms align well with the predicted disease.",
                "reasoning": "The leaf image shows brown necrotic lesions with concentric rings on tomato leaves, which is highly consistent with Early Blight caused by Alternaria solani.",
                "alternative_suggestions": ["Tomato___Septoria_leaf_spot"],
                "treatment_advice": "Apply copper-based fungicide. Remove affected leaves. Improve air circulation around plants. Avoid overhead watering.",
                "summary": "Prediction confirmed: Early Blight on Tomato with high confidence."
            }
        }


class PredictionResponse(BaseModel):
    """Main prediction response with Gemini AI validation."""
    class_name: str = Field(..., alias="class", description="Predicted disease class")
    probability: float = Field(..., ge=0, le=1, description="Model probability")
    uncertainty: float = Field(..., ge=0, le=1, description="Uncertainty score (0=confident, 1=uncertain)")
    top_k: List[PredictionItem] = Field(default_factory=list, description="Top-k predictions")
    gemini_validation: Optional[GeminiValidation] = Field(default=None, description="Gemini AI cross-validation report")
    image_url: Optional[str] = Field(default=None, description="Supabase storage URL of the uploaded image")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "class": "Tomato___Early_blight",
                "probability": 0.92,
                "uncertainty": 0.08,
                "top_k": [
                    {"class": "Tomato___Early_blight", "probability": 0.92},
                    {"class": "Tomato___Late_blight", "probability": 0.05},
                    {"class": "Tomato___Bacterial_spot", "probability": 0.02},
                ],
                "gemini_validation": None,
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gemini_available: bool
    device: str


class UserProfile(BaseModel):
    id: str
    email: str
    display_name: Optional[str] = None


class PredictionHistory(BaseModel):
    id: Optional[str] = None
    user_id: str
    image_url: Optional[str] = None
    prediction_class: str
    probability: float
    uncertainty: float
    top_k: Optional[List[dict]] = None
    created_at: Optional[str] = None
