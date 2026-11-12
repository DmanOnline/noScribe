"""Download the models that are bundled with the Windows build.

    python tools/download_models.py            # Canary 1B v2 + Whisper "fast"
    python tools/download_models.py --only canary

Revisions are pinned so a build always contains the tested files. Run from the
repository root; files land in models/<name>/ next to NOSCRIBE_README.txt.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

CANARY_REPO = "istupakov/canary-1b-v2-onnx"
CANARY_REV = "5ebc1520cef7b6b318b3526ad17adbfe00bc1bfc"
CANARY_FILES = ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder-model.int8.onnx"]
SILERO_REPO = "istupakov/silero-vad-onnx"
SILERO_REV = "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6"

# Whisper large-v3-turbo, int8 (the "fast" model of noScribe)
FAST_REPO = "mukowaty/faster-whisper-int8"
FAST_REV = "db1cdb5990df067cf53670d86241330c7ea454cf"
FAST_SUBDIR = "faster-whisper-large-v3-turbo-int8"
FAST_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]


def download_canary() -> None:
    target = MODELS / "canary-1b-v2"
    for name in CANARY_FILES:
        hf_hub_download(CANARY_REPO, name, revision=CANARY_REV, local_dir=target)
    hf_hub_download(SILERO_REPO, "silero_vad.onnx", revision=SILERO_REV,
                    local_dir=target / "silero-vad")
    print(f"Canary 1B v2 -> {target}")


def download_fast() -> None:
    target = MODELS / "fast"
    for name in FAST_FILES:
        path = Path(hf_hub_download(FAST_REPO, f"{FAST_SUBDIR}/{name}", revision=FAST_REV))
        shutil.copyfile(path, target / name)
    print(f"Whisper fast -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["canary", "fast"])
    args = parser.parse_args()
    if args.only in (None, "canary"):
        download_canary()
    if args.only in (None, "fast"):
        download_fast()
    # hf_hub_download with local_dir leaves a cache folder that must not be bundled
    for cache in MODELS.rglob(".cache"):
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
