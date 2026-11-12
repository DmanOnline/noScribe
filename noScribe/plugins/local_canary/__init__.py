"""Bundled NVIDIA Canary 1B v2 backend plugin (CPU, ONNX via onnx-asr)."""

from .models import CANARY_LANGUAGES, discover_canary_models, is_canary_model_dir
from .plugin import LocalCanaryPlugin

__all__ = [
    "CANARY_LANGUAGES",
    "LocalCanaryPlugin",
    "discover_canary_models",
    "is_canary_model_dir",
]
