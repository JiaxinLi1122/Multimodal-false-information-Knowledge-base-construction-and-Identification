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

import sys
import json
import torch
from pathlib import Path
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
    return {"risk": risk, "reason": reason}


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
    return {
        "risk":   "MEDIUM",
        "reason": "Real model expects multimodal input; text-only fallback used.",
    }
