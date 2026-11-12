"""Compare transcription models on Dutch speech: word error rate and CPU speed.

    python tools/benchmark_models.py --fleurs <test parquet> --n 60 \
        --model canary=models/canary-1b-v2 --model fast=models/fast \
        [--extra-audio my_recording.wav] [--threads 4] [--out results.json]

Uses the FLEURS nl_nl test set (read speech with reference transcripts).
Every model runs on the CPU with the same number of threads, like on a laptop
without an NVIDIA GPU. A folder with config.json + *.onnx is treated as Canary,
a folder with model.bin as a faster-whisper (CTranslate2) model.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer
    return jiwer.wer([normalize(r) for r in refs], [normalize(h) for h in hyps])


def load_fleurs(path: str, n: int):
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pylist()[:n]
    items = []
    for row in table:
        audio = row["audio"]
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        assert sr == SR, sr
        items.append((data, row["raw_transcription"]))
    return items


def load_wav(path: str):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        idx = np.linspace(0, len(data) - 1, int(len(data) * SR / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype("float32")
    return data


class WhisperRunner:
    def __init__(self, path, threads):
        from faster_whisper import WhisperModel
        # compute_type "auto" is what noScribe uses
        self.model = WhisperModel(path, device="cpu", compute_type="auto", cpu_threads=threads)

    def short(self, audio):
        segs, _ = self.model.transcribe(audio, language="nl", beam_size=5, vad_filter=False)
        return " ".join(s.text.strip() for s in segs)

    def long(self, audio):
        segs, _ = self.model.transcribe(audio, language="nl", beam_size=5, vad_filter=True)
        return " ".join(s.text.strip() for s in segs)


class CanaryRunner:
    def __init__(self, path, threads):
        import onnx_asr
        import onnxruntime as rt

        so = rt.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        self.asr = onnx_asr.load_model("nemo-canary-1b-v2", path, quantization="int8",
                                       sess_options=so, providers=prov)
        vad = onnx_asr.load_vad("silero", str(Path(path) / "silero-vad"), sess_options=so, providers=prov)
        self.vad_model = self.asr.with_vad(vad, batch_size=1, max_speech_duration_s=20,
                                           min_silence_duration_ms=300, speech_pad_ms=100)

    def short(self, audio):
        return self.asr.recognize(audio, sample_rate=SR, language="nl", pnc=True)

    def long(self, audio):
        return " ".join(s.text for s in self.vad_model.recognize(audio, sample_rate=SR, language="nl", pnc=True))


def make_runner(path, threads):
    p = Path(path)
    if (p / "model.bin").exists():
        return WhisperRunner(str(p), threads)
    return CanaryRunner(str(p), threads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", action="append", required=True, help="name=path")
    ap.add_argument("--extra-audio", action="append", default=[],
                    help="wav file, or wav=reference.txt to also score it")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()

    items = load_fleurs(args.fleurs, args.n)
    total_audio = sum(len(a) for a, _ in items) / SR
    refs = [r for _, r in items]
    extras, extra_refs = {}, {}
    for spec in args.extra_audio:
        wav, _, ref = spec.partition("=")
        extras[Path(wav).name] = load_wav(wav)
        if ref:
            extra_refs[Path(wav).name] = Path(ref).read_text(encoding="utf-8")
    print(f"FLEURS nl: {len(items)} utterances, {total_audio:.0f} s audio, {args.threads} CPU threads")

    results = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        t0 = time.monotonic()
        runner = make_runner(path, args.threads)
        load_s = time.monotonic() - t0
        runner.short(items[0][0])  # warm-up
        hyps = []
        t0 = time.monotonic()
        for audio, _ in items:
            hyps.append(runner.short(audio))
        infer_s = time.monotonic() - t0
        res = {
            "wer": round(wer(refs, hyps) * 100, 2),
            "rtf": round(infer_s / total_audio, 3),
            "load_s": round(load_s, 1),
            "size_mb": round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6),
            "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:5]],
            "extra": {},
        }
        for ename, audio in extras.items():
            t0 = time.monotonic()
            text = runner.long(audio)
            res["extra"][ename] = {"text": text, "rtf": round((time.monotonic() - t0) / (len(audio) / SR), 3)}
            if ename in extra_refs:
                res["extra"][ename]["wer"] = round(wer([extra_refs[ename]], [text]) * 100, 1)
        results[name] = res
        extra_info = "  ".join(f"{k}: WER {v['wer']}%" for k, v in res["extra"].items() if "wer" in v)
        print(f"{name:12s} WER {res['wer']:5.1f}%  RTF {res['rtf']:.3f}  load {res['load_s']}s  {res['size_mb']} MB  {extra_info}", flush=True)
        del runner

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
