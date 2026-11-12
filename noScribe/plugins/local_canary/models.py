"""Discovery of local Canary 1B v2 ONNX model folders.

A Canary folder is recognised by its ``config.json`` marker (written by the
istupakov/canary-1b-v2-onnx export) with ``model_type: nemo-conformer-aed``,
plus the files onnx-asr needs to load it offline and the Silero VAD model in
``silero-vad/``. Whisper's ``model.bin`` check never matches such a folder, so
the two discoveries do not overlap.

This module must stay import-light: it runs in the GUI process.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

CANARY_MODEL_TYPE = "nemo-conformer-aed"
VAD_SUBDIR = "silero-vad"
VAD_FILE = "silero_vad.onnx"

# Source languages supported by nvidia/canary-1b-v2 (model card).
CANARY_LANGUAGES = frozenset({
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu",
    "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv",
    "uk",
})


def model_quantization(path: Path) -> str | None:
    """Return ``"int8"`` or ``None`` (float32) depending on which files exist.

    int8 is preferred: it is about four times smaller and faster on CPU.
    Raises ``FileNotFoundError`` when neither variant is complete.
    """
    path = Path(path)
    if (path / "encoder-model.int8.onnx").is_file() and (path / "decoder-model.int8.onnx").is_file():
        return "int8"
    if (path / "encoder-model.onnx").is_file() and (path / "decoder-model.onnx").is_file():
        return None
    raise FileNotFoundError(f"No complete Canary encoder/decoder pair in {path}")


def is_canary_model_dir(path: Path) -> bool:
    path = Path(path)
    try:
        with (path / "config.json").open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict) or config.get("model_type") != CANARY_MODEL_TYPE:
        return False
    if not (path / "vocab.txt").is_file():
        logger.warning("Canary model dir %s lacks vocab.txt. Ignoring.", path)
        return False
    if not (path / VAD_SUBDIR / VAD_FILE).is_file():
        logger.warning("Canary model dir %s lacks %s/%s. Ignoring.", path, VAD_SUBDIR, VAD_FILE)
        return False
    try:
        model_quantization(path)
    except FileNotFoundError:
        logger.warning("Canary model dir %s lacks encoder/decoder ONNX files. Ignoring.", path)
        return False
    return True


def discover_canary_models(*directories: Path | str | None) -> dict[str, str]:
    """Map model IDs (folder names) to absolute paths of complete Canary folders.

    Earlier directories win on duplicate names, matching WhisperModelManager
    (package ``models/`` first, then the user models directory).
    """
    found: dict[str, str] = {}
    for directory in directories:
        if directory is None:
            continue
        if isinstance(directory, str):
            directory = Path(directory)
        # Also accepts importlib.resources Traversables such as the
        # MultiplexedPath returned by ``files("models")``.
        if not directory.is_dir():
            continue
        entries = []
        for entry in directory.iterdir():
            try:
                entries.append(Path(os.fspath(entry)))
            except TypeError:
                logger.warning("Cannot resolve model dir %s to a file system path. Ignoring.", entry)
        for entry in sorted(entries):
            if not entry.is_dir() or entry.name in found or ":" in entry.name:
                continue
            if is_canary_model_dir(entry):
                found[entry.name] = str(entry.absolute())
    return found
