"""Supabase service for authentication and data storage."""
import os
import logging
from typing import Optional, List, Dict
from datetime import datetime
import json

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from backend directory (handles running from project root)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv()  # Also check CWD

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")


class SupabaseService:
    """Supabase client wrapper for auth, database, and storage."""

    def __init__(self):
        self.client = None
        self._initialized = False

    def initialize(self):
        """Initialize Supabase client."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("⚠️  Supabase not configured. Set SUPABASE_URL and SUPABASE_KEY env vars.")
            print("   The app will work without Supabase (no auth/history).")
            return

        try:
            from supabase import create_client
            # Anon client for auth (token verification)
            self.client = create_client(SUPABASE_URL, SUPABASE_KEY)
            # Service role client for DB/storage (bypasses RLS)
            if SUPABASE_SERVICE_KEY:
                self._admin_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            else:
                self._admin_client = self.client
                print("⚠️  No SUPABASE_SERVICE_KEY — using anon key for DB ops (RLS may block writes)")
            self._initialized = True
            print("✓ Supabase connected")
        except Exception as e:
            print(f"⚠️  Supabase init failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._initialized and self.client is not None

    # --- Auth ---
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify a Supabase JWT token and return user info."""
        if not self.is_available:
            logger.warning("verify_token called but Supabase not available")
            return None
        try:
            user = self.client.auth.get_user(token)
            return {"id": user.user.id, "email": user.user.email}
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None

    def sign_up(self, email: str, password: str) -> dict:
        if not self.is_available:
            return {"error": "Supabase not configured"}
        try:
            res = self.client.auth.sign_up({"email": email, "password": password})
            return {"user_id": res.user.id, "email": res.user.email}
        except Exception as e:
            return {"error": str(e)}

    def sign_in(self, email: str, password: str) -> dict:
        if not self.is_available:
            return {"error": "Supabase not configured"}
        try:
            res = self.client.auth.sign_in_with_password({"email": email, "password": password})
            return {
                "access_token": res.session.access_token,
                "refresh_token": res.session.refresh_token,
                "user_id": res.user.id,
                "email": res.user.email,
            }
        except Exception as e:
            return {"error": str(e)}

    # --- Prediction History ---
    def save_prediction(self, user_id: str, prediction: dict,
                        image_url: Optional[str] = None,
                        gemini_validation: Optional[dict] = None) -> dict:
        """Save a prediction to history."""
        if not self.is_available:
            return {"status": "skipped", "reason": "supabase_not_configured"}
        try:
            data = {
                "user_id": user_id,
                "prediction_class": prediction.get("class", ""),
                "probability": prediction.get("probability", 0),
                "uncertainty": prediction.get("uncertainty", 0),
                "top_k": json.dumps(prediction.get("top_k", [])),
                "image_url": image_url,
                "created_at": datetime.utcnow().isoformat(),
            }
            if gemini_validation is not None:
                data["gemini_validation"] = json.dumps(gemini_validation)
            result = self._admin_client.table("predictions").insert(data).execute()
            return {"status": "saved", "id": result.data[0]["id"] if result.data else None}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_history(self, user_id: str, limit: int = 20) -> List[dict]:
        """Get prediction history for a user."""
        if not self.is_available:
            logger.warning("get_history called but Supabase not available")
            return []
        try:
            result = (
                self._admin_client.table("predictions")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

    # --- Image Storage ---
    def upload_image(self, file_bytes: bytes, filename: str, user_id: str) -> Optional[str]:
        """Upload image to Supabase Storage and return URL."""
        if not self.is_available:
            return None
        try:
            import uuid
            ext = filename.rsplit('.', 1)[-1] if '.' in filename else 'jpg'
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            path = f"{user_id}/{unique_name}"
            self._admin_client.storage.from_("plant-images").upload(
                path, file_bytes,
                file_options={"content-type": f"image/{ext}", "upsert": "true"}
            )
            url = self._admin_client.storage.from_("plant-images").get_public_url(path)
            return url
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            return None

    def get_stats(self, user_id: str) -> dict:
        """Get user prediction statistics."""
        if not self.is_available:
            return {"total": 0, "diseases_detected": 0}
        try:
            result = (
                self._admin_client.table("predictions")
                .select("prediction_class")
                .eq("user_id", user_id)
                .execute()
            )
            data = result.data or []
            unique_diseases = len(set(d["prediction_class"] for d in data))
            return {"total": len(data), "diseases_detected": unique_diseases}
        except Exception:
            return {"total": 0, "diseases_detected": 0}

    # --- Training Feedback ---
    def save_feedback(self, user_id: Optional[str], prediction_id: Optional[str],
                      image_url: str, predicted_class: str,
                      actual_class: str, is_correct: bool) -> dict:
        """Save user feedback on a prediction for model training."""
        if not self.is_available:
            return {"status": "skipped", "reason": "supabase_not_configured"}
        try:
            data = {
                "predicted_class": predicted_class,
                "actual_class": actual_class,
                "is_correct": is_correct,
                "created_at": datetime.utcnow().isoformat(),
            }
            if user_id and user_id != "anonymous":
                data["user_id"] = user_id
            if prediction_id:
                data["prediction_id"] = prediction_id
            if image_url:
                data["image_url"] = image_url
            result = self._admin_client.table("training_feedback").insert(data).execute()
            return {"status": "saved", "id": result.data[0]["id"] if result.data else None}
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return {"status": "error", "error": str(e)}

    def get_feedback(self, limit: int = 500) -> List[dict]:
        """Get all training feedback (for fine-tuning)."""
        if not self.is_available:
            return []
        try:
            result = (
                self._admin_client.table("training_feedback")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get feedback: {e}")
            return []

    def get_feedback_stats(self) -> dict:
        """Get feedback statistics for training dashboard."""
        if not self.is_available:
            return {"total": 0, "correct": 0, "incorrect": 0, "classes": {}}
        try:
            result = (
                self._admin_client.table("training_feedback")
                .select("*")
                .execute()
            )
            data = result.data or []
            correct = sum(1 for d in data if d.get("is_correct"))
            incorrect = len(data) - correct
            classes = {}
            for d in data:
                cls = d.get("actual_class", "unknown")
                classes[cls] = classes.get(cls, 0) + 1
            return {
                "total": len(data),
                "correct": correct,
                "incorrect": incorrect,
                "classes": classes,
            }
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {"total": 0, "correct": 0, "incorrect": 0, "classes": {}}


# Global instance
supabase_service = SupabaseService()
