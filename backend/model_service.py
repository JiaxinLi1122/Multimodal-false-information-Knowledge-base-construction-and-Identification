# backend/model_service.py
#
# Loads the trained Text_Concat_Vision model once at server startup and exposes
# two inference functions:
#
#   predict(text, image_tensor)  – full multimodal inference (future use)
#   predict_text_only(text)      – text-only fallback (current extension behaviour)
#
# The model is a multimodal classifier: sigmoid output close to 1.0 means fake,
# close to 0.0 means real.  See mult_models.py for the architecture.

import base64
import io
import re
import sys
import json
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

# --------------------------------------------------------------------------- #
# Path setup – allow importing mult_models and dataset from the project root,
# regardless of where uvicorn is launched from.
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from mult_models import Text_Concat_Vision   # noqa: E402 (import after sys.path)
from dataset    import text_preprocessing    # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CHECKPOINT_PATH = PROJECT_ROOT / "saved_models" / "best_model.pt"
CONFIG_PATH     = PROJECT_ROOT / "config" / "config.json"
BERT_MODEL_NAME = "bert-base-uncased"

# Must match the value used during training (config.json: "max_len": 500)
_MAX_LEN = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eval image transform – must be identical to eval_transform in main.py
# (Resize → ToTensor → ImageNet normalisation).  No augmentation at inference.
_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# --------------------------------------------------------------------------- #
# Model loading
#
# _load_model() is called once at module import time (i.e. when uvicorn starts).
# Do NOT move this call inside the request handler – loading BERT + VGG takes
# several seconds and must not happen per-request.
# --------------------------------------------------------------------------- #

def _load_model() -> Text_Concat_Vision:
    """Instantiate Text_Concat_Vision and load the best saved checkpoint."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    model = Text_Concat_Vision(cfg)

    # map_location lets the checkpoint run on CPU-only machines even if it was
    # saved on a GPU.
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # disable dropout for inference
    return model


# Module-level singletons – loaded exactly once.
_model     = _load_model()
_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, do_lower_case=True)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the same BERT pre-processing used during training (see dataset.py).
    Returns (input_ids, attention_mask), each shaped [1, MAX_LEN].
    """
    encoded = _tokenizer.encode_plus(
        text=text_preprocessing(text),   # cleans @mentions, &amp; etc.
        add_special_tokens=True,
        max_length=_MAX_LEN,
        padding="max_length",
        return_attention_mask=True,
        truncation=True,
    )
    input_ids      = torch.tensor(encoded["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded["attention_mask"]).unsqueeze(0).to(device)
    return input_ids, attention_mask


_SENSATIONAL_WORDS = [
    "breaking", "shocking", "explosive", "bombshell", "exposed",
    "conspiracy", "cover-up", "hoax", "scandal", "leaked", "secret",
    "banned", "censored", "suppressed", "deep state", "plandemic",
    "you won't believe", "must see", "share immediately",
]

_EMOTIONAL_WORDS = [
    "outrage", "disgusting", "unbelievable", "incredible", "insane",
    "terrifying", "devastating", "alarming", "urgent", "critical",
]


def _prob_to_risk(prob: float) -> tuple[str, str]:
    """
    Map the model's sigmoid output to a risk label.

    The model was trained as a binary classifier:
        label 1 = fake news   → sigmoid output near 1.0
        label 0 = real news   → sigmoid output near 0.0

    Thresholds are deliberately asymmetric: we require higher confidence
    before declaring HIGH to reduce false positives.
    """
    if prob >= 0.65:
        return (
            "HIGH",
            f"Model predicts likely misinformation ({prob:.0%} fake confidence).",
        )
    if prob >= 0.40:
        return (
            "MEDIUM",
            f"Model is uncertain ({prob:.0%} fake confidence). Manual review recommended.",
        )
    return (
        "LOW",
        f"Model predicts likely real content ({1 - prob:.0%} real confidence).",
    )


def generate_explanations(text: str, prob: float, used_image: bool) -> list[str]:
    """Generate human-readable bullet points explaining the risk assessment."""
    points = []
    text_lower = text.lower()

    # Model confidence summary
    if prob >= 0.65:
        points.append(f"AI model flagged content with {prob:.0%} fake-news probability")
    elif prob >= 0.40:
        points.append(f"AI model is uncertain — {prob:.0%} fake-news probability detected")
    else:
        points.append(f"AI model found low risk ({(1 - prob):.0%} confidence in authenticity)")

    # Sensational language
    hits = [w for w in _SENSATIONAL_WORDS if w in text_lower]
    if hits:
        sample = ", ".join(f'"{h}"' for h in hits[:3])
        points.append(f"Sensational language detected: {sample}")

    # Emotional language
    emo = [w for w in _EMOTIONAL_WORDS if w in text_lower]
    if emo:
        n = len(emo)
        points.append(f"Emotionally charged phrasing present ({n} indicator{'s' if n > 1 else ''})")

    # ALL-CAPS words
    caps_count = len(re.findall(r'\b[A-Z]{4,}\b', text))
    if caps_count > 3:
        points.append(f"Heavy ALL-CAPS usage detected ({caps_count} instances)")

    # Excessive exclamation marks
    excl = text.count('!')
    if excl >= 3:
        points.append(f"Excessive exclamation marks found ({excl})")

    # Image note
    if used_image:
        points.append("Both text and image were analyzed by the multimodal model")
    else:
        points.append("No image available — analysis based on text only")

    return points


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #

def load_image_from_base64(data: str) -> torch.Tensor:
    """
    Decode a raw base64 string (no data-URL prefix) into a [1, 3, 224, 224] tensor.

    Raises:
        binascii.Error            – data is not valid base64
        PIL.UnidentifiedImageError – decoded bytes are not a valid image
    """
    image_bytes = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _IMAGE_TRANSFORM(image).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Public inference API
# --------------------------------------------------------------------------- #

def predict(text: str, image_tensor: torch.Tensor) -> dict:
    """
    Full multimodal inference – use this once the extension also sends an image.

    Args:
        text:         Raw visible page text (extracted by content.js).
        image_tensor: Pre-processed image tensor, shape [1, 3, 224, 224],
                      normalised with ImageNet mean/std (see main.py transforms).

    Returns:
        {"risk": "LOW|MEDIUM|HIGH", "reason": "<string>"}
    """
    input_ids, attention_mask = _tokenize(text)
    image = image_tensor.to(device)

    with torch.no_grad():
        # model() returns a scalar sigmoid probability per sample
        prob = _model(text=[input_ids, attention_mask], image=image).item()

    risk, reason = _prob_to_risk(prob)
    return {
        "risk": risk,
        "reason": reason,
        "confidence": prob,
        "explanations": generate_explanations(text, prob, used_image=True),
        "prob": prob,
    }


def predict_text_only(text: str) -> dict:
    """
    Text-only fallback used when no image is available (current extension behaviour).

    The Text_Concat_Vision architecture concatenates a VGG-19 image embedding
    with the BERT text embedding before classification.  Passing a dummy image
    would introduce a signal the model was never trained to handle in this context,
    so we choose to surface the limitation explicitly instead.

    TODO: update the extension to capture a screenshot or a representative image
          from the page, then call predict(text, image_tensor) instead.
    """
    prob = 0.5
    return {
        "risk": "MEDIUM",
        "reason": "Real model expects multimodal input; text-only fallback used.",
        "confidence": prob,
        "explanations": generate_explanations(text, prob, used_image=False),
    }
