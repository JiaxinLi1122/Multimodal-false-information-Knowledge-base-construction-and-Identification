# backend/main.py
#
# FastAPI server for the Multi-False Detector extension.
#
# Run with:
#   uvicorn main:app --reload
#
# Routing logic:
#   image_data present and decodable → predict(text, image_tensor)  [real model]
#   image_data missing or decode fails → predict_text_only(text)     [fallback]

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importing model_service triggers model loading (BERT + VGG19).
# This happens once when uvicorn starts, not on every request.
from model_service import load_image_from_base64, predict, predict_text_only, generate_explanations

app = FastAPI(title="Multi-False Detector API", version="0.3.0")

# Allow the Chrome extension (and local dev tools) to reach this API.
# In production, replace "*" with your actual extension origin or domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #

class AnalyseRequest(BaseModel):
    text: str
    image_data: Optional[str] = None  # raw base64 screenshot (no data-URL prefix)


class AnalyseResponse(BaseModel):
    risk: str               # "LOW" | "MEDIUM" | "HIGH"
    reason: str
    confidence: float       # model's fake-news probability (0–1)
    explanations: list[str] # short bullet-point explanations
    used_image: bool        # True when the real model ran with an image; False for fallback


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.post("/analyze", response_model=AnalyseResponse)
def analyze(request: AnalyseRequest) -> AnalyseResponse:
    """
    Analyse page text (and optionally an image) for misinformation risk.

    Decision tree:
      1. image_data provided → decode base64 → run full multimodal model
         - If decode or preprocessing fails, fall back to text-only
      2. No image_data → text-only fallback (model requires paired image)
    """
    print("\n--- /analyze request ---")
    print(f"  [1] text length    : {len(request.text)} chars")
    print(f"  [2] image_data     : {'yes (' + str(len(request.image_data)) + ' b64 chars)' if request.image_data else 'not provided'}")

    if request.image_data:
        try:
            image_tensor = load_image_from_base64(request.image_data)
            print(f"  [3] image decode   : success")
            print(f"  [4] tensor shape   : {list(image_tensor.shape)}")

            result = predict(request.text, image_tensor)
            print(f"  [5] model prob     : {result['prob']:.4f}")
            print(f"  [6] risk level     : {result['risk']}")

            return AnalyseResponse(**result, used_image=True)

        except Exception as exc:
            print(f"  [3] image decode   : FAILED ({type(exc).__name__}: {exc})")
            print(f"  [5] model prob     : N/A (fallback)")
            print(f"  [6] risk level     : MEDIUM (fallback)")

            fallback_prob = 0.5
            fallback_explanations = generate_explanations(request.text, fallback_prob, used_image=False)
            fallback_explanations.insert(0, f"Screenshot could not be decoded ({type(exc).__name__})")
            return AnalyseResponse(
                risk="MEDIUM",
                reason=f"Screenshot decode failed ({type(exc).__name__}); text-only fallback used.",
                confidence=fallback_prob,
                explanations=fallback_explanations,
                used_image=False,
            )

    # No screenshot provided
    print(f"  [3] image decode   : skipped (no data)")
    print(f"  [4] tensor shape   : N/A")

    result = predict_text_only(request.text)
    print(f"  [5] model prob     : N/A (text-only fallback)")
    print(f"  [6] risk level     : {result['risk']}")

    return AnalyseResponse(
        risk=result["risk"],
        reason=result["reason"],
        confidence=result["confidence"],
        explanations=result["explanations"],
        used_image=False,
    )
