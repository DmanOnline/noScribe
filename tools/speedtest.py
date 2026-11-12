"""Speed and accuracy test of the built app on long Dutch audio.

    python tools/speedtest.py make  --fleurs <test parquet> --n 60 --out speedtest
    python tools/speedtest.py score --ref speedtest/reference.txt --hyp transcript.txt

``make`` concatenates FLEURS nl_nl test utterances (different speakers, read
speech with known text) into one WAV of about ten minutes, with short pauses,
and writes the reference text. ``score`` prints the word error rate of a
noScribe .txt transcript against it.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

SR = 16000


def make(args) -> None:
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.fleurs).to_pylist()[: args.n]
    pause = np.zeros(int(SR * 0.7), dtype="float32")
    parts, refs = [], []
    for row in rows:
        data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        assert sr == SR
        parts += [data, pause]
        refs.append(row["raw_transcription"].strip())
    audio = np.concatenate(parts)
    sf.write(out / "speedtest_nl.wav", audio, SR, subtype="PCM_16")
    (out / "reference.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")
    print(f"{len(rows)} utterances, {len(audio) / SR / 60:.1f} min -> {out}")


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def transcript_text(path: str) -> str:
    """Spoken text of a noScribe .txt transcript, without header and markers."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    spoken = []
    for line in lines:
        m = re.match(r"^(?:S\d\d|[^:\[\(]{1,40}):\s+(.*)$", line)
        if not m:
            continue
        text = m.group(1)
        text = re.sub(r"\[\d\d:\d\d:\d\d\]", " ", text)        # timestamps
        text = re.sub(r"\((?:\.+|\d+[^)]*)\)", " ", text)       # pause markers
        text = re.sub(r"//S\d\d:|//", " ", text)                # overlapping speech
        spoken.append(text)
    return " ".join(spoken)


def score(args) -> None:
    import jiwer

    ref = _normalize(Path(args.ref).read_text(encoding="utf-8"))
    hyp = _normalize(transcript_text(args.hyp))
    print(f"{jiwer.wer(ref, hyp) * 100:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--fleurs", required=True)
    m.add_argument("--n", type=int, default=60)
    m.add_argument("--out", default="speedtest")
    s = sub.add_parser("score")
    s.add_argument("--ref", required=True)
    s.add_argument("--hyp", required=True)
    args = ap.parse_args()
    {"make": make, "score": score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
