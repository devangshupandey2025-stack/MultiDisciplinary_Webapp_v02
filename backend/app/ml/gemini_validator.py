"""
Gemini AI cross-validation service.
Validates MobileNetV3 predictions using Google Gemini with multimodal (image + text) input.
"""
import io
import json
import os
import asyncio
from typing import Optional

from google import genai
from google.genai import types


SYSTEM_PROMPT = """You are an expert plant pathologist AI assistant. Your role is to cross-validate 
plant disease predictions made by a MobileNetV3 computer vision model.

You will receive:
1. The actual leaf image for visual inspection
2. The model's predicted disease class and confidence
3. Top alternative predictions with probabilities
4. Image metadata (brightness, contrast, color distribution, dimensions)

Your job is to:
- Visually inspect the leaf image for disease symptoms (spots, lesions, discoloration, wilting, etc.)
- Assess whether the prediction is reasonable given what you see in the image
- Provide a confidence assessment (High, Moderate, or Low)
- Explain your reasoning based on visible disease symptoms
- Suggest alternative diseases if the prediction seems off
- Provide practical treatment advice for the predicted disease

IMPORTANT: Respond ONLY in valid JSON with exactly these fields:
{
  "agrees": true/false,
  "confidence_assessment": "High/Moderate/Low — brief explanation",
  "reasoning": "2-3 sentences explaining why you agree or disagree based on visual symptoms",
  "alternative_suggestions": ["list", "of", "alternatives"] or [],
  "treatment_advice": "Practical treatment steps",
  "summary": "One-line summary of your validation"
}

Do NOT include any text outside the JSON object. No explanations, no markdown, just the JSON."""


def _build_prompt(prediction: dict, image_metadata: dict) -> str:
    """Build the text prompt for Gemini validation."""
    top_k_str = "\n".join(
        f"  {i+1}. {p['class']} — {p['probability']*100:.1f}%"
        for i, p in enumerate(prediction.get("top_k", []))
    )

    return f"""Validate this plant disease prediction by examining the leaf image I've provided:

**MobileNetV3 Prediction:**
- Disease: {prediction['class']}
- Confidence: {prediction['probability']*100:.1f}%
- Uncertainty: {prediction['uncertainty']*100:.1f}%

**Top-K Alternatives:**
{top_k_str}

**Image Metadata:**
- Dimensions: {image_metadata.get('width', '?')}x{image_metadata.get('height', '?')} pixels
- Brightness: {image_metadata.get('brightness', '?')} (0=dark, 1=bright)
- Contrast: {image_metadata.get('contrast', '?')} (0=flat, 1=high)
- Dominant color: {image_metadata.get('dominant_color', '?')}
- Green ratio: {image_metadata.get('green_ratio', '?')} (higher = more green/leaf content)
- RGB means: R={image_metadata.get('r_mean', '?')}, G={image_metadata.get('g_mean', '?')}, B={image_metadata.get('b_mean', '?')}

Look at the leaf image carefully and respond with your validation JSON:"""


def _parse_gemini_response(response_text: str) -> Optional[dict]:
    """Parse Gemini JSON response, handling markdown code blocks and extra text."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    required = ["agrees", "confidence_assessment", "reasoning", "summary"]
    if not all(k in data for k in required):
        return None

    return {
        "agrees": bool(data["agrees"]),
        "confidence_assessment": str(data["confidence_assessment"]),
        "reasoning": str(data["reasoning"]),
        "alternative_suggestions": list(data.get("alternative_suggestions", [])),
        "treatment_advice": str(data.get("treatment_advice", "")),
        "summary": str(data["summary"]),
    }


class GeminiValidator:
    """Validates MobileNetV3 predictions using Google Gemini with multimodal input."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._available = False
        try:
            self._client = genai.Client(api_key=api_key)
            self._available = True
            print(f"✓ Gemini client initialized: {self.model_name}")
        except Exception as e:
            print(f"⚠️  Gemini client initialization failed: {e}")
            self._client = None

    def check_availability(self) -> bool:
        """Verify Gemini API is reachable with a lightweight call."""
        if not self._client:
            self._available = False
            return False
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents="Reply with only the word: OK",
                config=types.GenerateContentConfig(
                    max_output_tokens=5,
                ),
            )
            self._available = response.text is not None
            if self._available:
                print(f"✓ Gemini API reachable: {self.model_name}")
            return self._available
        except Exception as e:
            print(f"⚠️  Gemini API not reachable: {e}")
            self._available = False
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def validate(
        self, prediction: dict, image_metadata: dict, image_bytes: bytes = None
    ) -> Optional[dict]:
        """
        Validate a MobileNet prediction with Gemini (multimodal: image + text).
        Returns parsed validation dict or None if unavailable/failed.
        """
        if not self._available or not self._client:
            return None

        prompt_text = _build_prompt(prediction, image_metadata)

        contents = []
        # Include the actual leaf image if available
        if image_bytes:
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            )
        contents.append(prompt_text)

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )
            content = response.text
            if not content:
                print("Gemini returned empty response")
                return None

            parsed = _parse_gemini_response(content)
            if parsed is None:
                print(f"Gemini response could not be parsed. Raw (first 500 chars): {content[:500]}")
            return parsed
        except Exception as e:
            print(f"Gemini validation failed: {e}")
            return None

    async def validate_async(
        self, prediction: dict, image_metadata: dict, image_bytes: bytes = None
    ) -> Optional[dict]:
        """Async wrapper around validate() for use in FastAPI endpoints."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.validate, prediction, image_metadata, image_bytes
        )


# Global validator instance
_validator: Optional[GeminiValidator] = None


def get_validator() -> GeminiValidator:
    """Get or create the global Gemini validator instance."""
    global _validator
    if _validator is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        _validator = GeminiValidator(api_key=api_key, model_name=model)
    return _validator
