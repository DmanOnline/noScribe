"""Child-process worker for NVIDIA Canary 1B v2 (ONNX, CPU) via onnx-asr.

Keep module-level imports light: on Windows and macOS the worker is started
with ``spawn`` and this module is re-imported in the child. numpy,
onnxruntime and onnx_asr are imported inside the entry point only.
"""

from __future__ import annotations

import os
import time
import wave

from .worker_errors import WorkerState

TARGET_SAMPLE_RATE = 16000


def _load_audio(path: str):
    """Return ``(float32 mono waveform, sample_rate)``.

    noScribe's local working file is 16 kHz mono PCM16 WAV, which is read with
    the standard library. Other containers (e.g. FLAC on the server) are
    decoded with PyAV, which noScribe already ships.
    """
    import numpy as np

    try:
        with wave.open(path, "rb") as stream:
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            raw = stream.readframes(stream.getnframes())
        if width != 2:
            raise wave.Error(f"unsupported sample width {width}")
        waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        del raw
        if channels > 1:
            waveform = waveform.reshape(-1, channels).mean(axis=1, dtype=np.float32)
        return waveform, sample_rate
    except (wave.Error, EOFError):
        pass

    import av

    chunks = []
    resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SAMPLE_RATE)
    with av.open(path, mode="r") as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    waveform = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    return waveform, TARGET_SAMPLE_RATE


def _resample_to_16k(waveform, sample_rate):
    """Linear-interpolation fallback for rates onnx-asr cannot resample."""
    import numpy as np

    if sample_rate == TARGET_SAMPLE_RATE or len(waveform) == 0:
        return waveform
    duration = len(waveform) / sample_rate
    target_len = int(round(duration * TARGET_SAMPLE_RATE))
    positions = np.linspace(0, len(waveform) - 1, target_len, dtype=np.float64)
    return np.interp(positions, np.arange(len(waveform)), waveform).astype(np.float32)



def diarization_turns(segments, *, max_gap_s: float = 0.8, min_len_s: float = 0.25):
    """Turn diarization segments (ms, label) into speaker turns in seconds.

    Consecutive segments of the same speaker with a short gap are merged, so a
    turn is one stretch of speech by one person. Overlapping segments are
    clipped so no audio is transcribed twice. Returns [(start_s, end_s, label)].
    """
    items = sorted(
        (float(seg["start"]) / 1000.0, float(seg["end"]) / 1000.0, str(seg.get("label", "")))
        for seg in segments or []
        if float(seg["end"]) > float(seg["start"])
    )
    turns: list[list] = []
    for start, end, label in items:
        if turns:
            prev = turns[-1]
            if label == prev[2] and start - prev[1] <= max_gap_s:
                prev[1] = max(prev[1], end)
                continue
            if end <= prev[1]:
                continue  # fully inside the previous turn (overlapping speech)
            start = max(start, prev[1])
        turns.append([start, end, label])
    return [(s, e, l) for s, e, l in turns if e - s >= min_len_s]


class _Segment:
    __slots__ = ("start", "end", "text")

    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text

def canary_proc_entrypoint(args: dict, q):
    """Run Canary in a child process and stream events to the parent via ``q``.

    Events use the same shapes as ``whisper_mp_worker``:
      {"type": "state", "device": "cpu", "stage": str}
      {"type": "log", "level": str, "msg": str}
      {"type": "status", "level": str, "message_id": str, "params": {...}}
      {"type": "progress", "pct": float, "detail": str}
      {"type": "segment", "segment": {"start": float, "end": float, "text": str}}
      {"type": "result", "ok": True, "info": {...}}
    Failed results carry the structured fields of ``worker_errors.error_result``.
    Canary's ONNX decoder does not produce token timestamps, so segments carry
    VAD chunk boundaries and no ``words``.
    """
    state = WorkerState(q, "cpu")
    try:
        state.update("imports")
        # Never reach out to the Hugging Face Hub: models are local only.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import numpy as np
        import onnxruntime as rt
        import onnx_asr

        def log(level, msg):
            q.put({"type": "log", "level": level, "msg": msg})

        model_dir = str(args["model_path"])
        quantization = args.get("quantization")
        cpu_threads = int(args.get("cpu_threads", 4))
        language = args["language"]

        state.update("model_loading", device="cpu")
        started = time.monotonic()
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = cpu_threads
        sess_options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        asr = onnx_asr.load_model(
            "nemo-canary-1b-v2",
            model_dir,
            quantization=quantization,
            sess_options=sess_options,
            providers=providers,
        )
        vad = onnx_asr.load_vad(
            "silero",
            os.path.join(model_dir, "silero-vad"),
            sess_options=sess_options,
            providers=providers,
        )
        model = asr.with_vad(
            vad,
            batch_size=1,  # stream every chunk as soon as it is decoded
            threshold=float(args.get("vad_threshold", 0.5)),
            max_speech_duration_s=float(args.get("max_segment_seconds", 20.0)),
            min_silence_duration_ms=float(args.get("min_silence_ms", 300)),
            speech_pad_ms=100,
        )
        log("info", f"Canary: loaded {quantization or 'fp32'} model on CPU "
                    f"({cpu_threads} threads) in {time.monotonic() - started:.1f} s")

        state.update("audio_loading")
        audio_path = args.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio path does not exist: {audio_path}")
        waveform, sample_rate = _load_audio(audio_path)
        if sample_rate not in (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000):
            waveform, sample_rate = _resample_to_16k(waveform, sample_rate), TARGET_SAMPLE_RATE
        duration = len(waveform) / sample_rate if sample_rate else 0.0

        q.put({"type": "status", "level": "info", "message_id": "start_transcription", "params": {}})
        state.update("transcription")
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        max_segment_s = float(args.get("max_segment_seconds", 20.0))
        diarization = args.get("diarization") or {}
        turns = diarization_turns(diarization.get("segments"))

        def by_speaker_turns():
            """Transcribe each speaker turn on its own, so a segment never
            mixes two speakers (Canary has no word timestamps to split on)."""
            pad = 0.15
            for turn_start, turn_end, _label in turns:
                a = max(0, int((turn_start - pad) * sample_rate))
                b = min(len(waveform), int((turn_end + pad) * sample_rate))
                if b <= a:
                    continue
                piece = waveform[a:b]
                if turn_end - turn_start <= max_segment_s:
                    text = asr.recognize(piece, sample_rate=sample_rate,
                                         language=language, pnc=True)
                    yield _Segment(turn_start, turn_end, text)
                else:
                    # long monologue: cut it at pauses, as without diarization
                    offset = a / sample_rate
                    for seg in model.recognize(piece, sample_rate=sample_rate,
                                               language=language, pnc=True):
                        yield _Segment(max(turn_start, offset + float(seg.start)),
                                       min(turn_end, offset + float(seg.end)), seg.text)

        if not len(waveform):
            segments = ()
        elif turns:
            log("info", f"Canary: transcribing {len(turns)} speaker turns")
            segments = by_speaker_turns()
        else:
            segments = model.recognize(waveform, sample_rate=sample_rate,
                                       language=language, pnc=True)
        for segment in segments:
            text = (segment.text or "").strip()
            start = float(segment.start)
            end = float(segment.end)
            if text:
                # Leading space mirrors Whisper segments, which the desktop
                # concatenates directly into paragraphs.
                q.put({"type": "segment",
                       "segment": {"start": start, "end": end, "text": " " + text}})
            if duration > 0:
                q.put({"type": "progress", "pct": min(100.0, end / duration * 100.0),
                       "detail": f"{end:.0f}/{duration:.0f} s"})

        q.put({"type": "result", "ok": True, "info": {
            "duration": duration,
            "language": language,
            "language_probability": None,
            "sample_rate": TARGET_SAMPLE_RATE,
        }})
    except Exception as error:
        try:
            q.put(state.failure(error))
        except Exception:
            pass
