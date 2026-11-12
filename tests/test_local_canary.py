"""Tests for the bundled Canary 1B v2 plugin; the ONNX model is always mocked."""

import json
import os
import subprocess
import sys
import types
import wave
from collections import deque
from pathlib import Path
from queue import Empty

import numpy as np
import pytest

from noScribe.inference import InferenceCancelled, TranscriptionRequest
from noScribe.models import ModelRef
from noScribe.plugins.factory import create_builtin_registry
from noScribe.plugins.local_canary import (
    LocalCanaryPlugin,
    discover_canary_models,
    is_canary_model_dir,
)
from noScribe.plugins.local_canary import plugin as canary_plugin
from noScribe.plugins.local_canary.models import model_quantization
from noScribe.plugins.local_canary.settings import CanaryWorkerSettings
from noScribe.plugins.manifest import ExecutionType
from noScribe.plugins.settings import PluginSettingsStore
from noScribe.worker_errors import WorkerError

REPO = str(Path(__file__).resolve().parent.parent)
CANARY = ModelRef("local-canary", "canary-1b-v2")


# --- fakes -----------------------------------------------------------------

class _FakeQueue:
    def __init__(self, messages, on_get=None):
        self.messages = deque(messages)
        self.on_get = on_get

    def get(self, timeout):
        if self.on_get is not None:
            self.on_get()
        if not self.messages:
            raise Empty
        return self.messages.popleft()

    def close(self):
        pass

    def join_thread(self):
        pass


class _FakeProcess:
    def __init__(self, target, args, alive_after_empty=True):
        self.target = target
        self.args = args
        self.exitcode = 0
        self.alive = False
        self.terminated = False
        self.alive_after_empty = alive_after_empty

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive and self.alive_after_empty

    def join(self, timeout):
        self.alive = False

    def terminate(self):
        self.terminated = True
        self.alive = False

    def close(self):
        pass


class _FakeContext:
    def __init__(self, messages, *, alive=True, on_get=None):
        self.messages = messages
        self.alive = alive
        self.on_get = on_get
        self.process = None

    def Queue(self):
        return _FakeQueue(self.messages, self.on_get)

    def Process(self, target, args):
        self.process = _FakeProcess(target, args, self.alive)
        return self.process


def _make_canary_dir(root: Path, name="canary-1b-v2", *, int8=True, fp32=False, vad=True):
    path = root / name
    (path / "silero-vad").mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "nemo-conformer-aed"}))
    (path / "vocab.txt").write_text("")
    if int8:
        (path / "encoder-model.int8.onnx").write_bytes(b"")
        (path / "decoder-model.int8.onnx").write_bytes(b"")
    if fp32:
        (path / "encoder-model.onnx").write_bytes(b"")
        (path / "decoder-model.onnx").write_bytes(b"")
    if vad:
        (path / "silero-vad" / "silero_vad.onnx").write_bytes(b"")
    return path


@pytest.fixture
def models_dir(tmp_path):
    root = tmp_path / "models"
    _make_canary_dir(root)
    return root


def _registry(models_dir, settings_dir=None):
    return create_builtin_registry(
        {}, settings_dir=settings_dir, canary_models=discover_canary_models(models_dir),
    )


# --- manifest and catalogue -------------------------------------------------

def test_manifest_is_builtin_transcription_only():
    plugin = LocalCanaryPlugin({})
    assert plugin.manifest.id == "local-canary"
    assert plugin.manifest.execution.type is ExecutionType.BUILTIN
    assert plugin.manifest.engine == "canary"
    # No token timestamps from the ONNX decoder, so no word_timestamps claim.
    assert plugin.manifest.capabilities == frozenset({"transcription"})


def test_catalogue_is_empty_without_model_files(tmp_path):
    (tmp_path / "models" / "precise").mkdir(parents=True)
    (tmp_path / "models" / "precise" / "model.bin").write_bytes(b"")
    registry = _registry(tmp_path / "models")
    assert "local-canary" in {m.id for m in registry.list_plugins()}
    assert [m for m in registry.list_models() if m.ref.backend_id == "local-canary"] == []


def test_registry_without_canary_argument_lists_no_canary_models():
    registry = create_builtin_registry({"precise": "models/precise"})
    assert {str(m.ref) for m in registry.list_models("transcription")} == {"local-whisper:precise"}


def test_catalogue_lists_model_when_files_present(models_dir):
    registry = _registry(models_dir)
    models = [m for m in registry.list_models("transcription") if m.ref.backend_id == "local-canary"]
    assert [str(m.ref) for m in models] == ["local-canary:canary-1b-v2"]
    assert models[0].capabilities == frozenset({"transcription"})
    assert "canary-1b-v2" in registry.model_options("transcription")


@pytest.mark.parametrize("kwargs", [{"vad": False}, {"int8": False}])
def test_incomplete_model_dirs_are_ignored(tmp_path, kwargs):
    path = _make_canary_dir(tmp_path, **kwargs)
    assert not is_canary_model_dir(path)
    assert discover_canary_models(tmp_path, None, tmp_path / "missing") == {}


def test_wrong_marker_is_ignored(tmp_path):
    path = _make_canary_dir(tmp_path)
    (path / "config.json").write_text(json.dumps({"model_type": "nemo-conformer-tdt"}))
    assert not is_canary_model_dir(path)


def test_quantization_prefers_int8_and_falls_back_to_fp32(tmp_path):
    assert model_quantization(_make_canary_dir(tmp_path, "both", fp32=True)) == "int8"
    assert model_quantization(_make_canary_dir(tmp_path, "fp32", int8=False, fp32=True)) is None


def test_first_directory_wins_on_duplicate_names(tmp_path):
    first = _make_canary_dir(tmp_path / "a")
    _make_canary_dir(tmp_path / "b")
    assert discover_canary_models(tmp_path / "a", tmp_path / "b") == {
        "canary-1b-v2": str(first.absolute())
    }


# --- settings ----------------------------------------------------------------

def test_settings_validation():
    assert CanaryWorkerSettings().fallback_language == "nl"
    with pytest.raises(ValueError):
        CanaryWorkerSettings(cpu_threads=0)
    with pytest.raises(ValueError):
        CanaryWorkerSettings(device="cuda")
    with pytest.raises(ValueError):
        CanaryWorkerSettings(fallback_language="ja")
    with pytest.raises(ValueError, match="Unknown plugin settings"):
        CanaryWorkerSettings.from_mapping({"beam_size": 4})


# --- routing, events, language -------------------------------------------------

def test_registry_routes_to_canary_worker_and_streams_segments(monkeypatch, models_dir, tmp_path):
    store = PluginSettingsStore(tmp_path / "settings")
    store.save("local-canary", {"cpu_threads": 2, "max_segment_seconds": 15})
    registry = _registry(models_dir, tmp_path / "settings")
    context = _FakeContext([
        {"type": "state", "device": "cpu", "stage": "transcription"},
        {"type": "log", "level": "info", "msg": "loaded"},
        {"type": "segment", "segment": {"start": 0.5, "end": 3.0, "text": " Goedemorgen."}},
        {"type": "progress", "pct": 50.0, "detail": "3/6 s"},
        {"type": "segment", "segment": {"start": 3.5, "end": 6.0, "text": " Tot ziens."}},
        {"type": "result", "ok": True, "info": {"duration": 6.0, "language": "nl"}},
    ])
    monkeypatch.setattr(canary_plugin.mp, "get_context", lambda method: context)
    segments, progress, logs = [], [], []

    info = registry.transcribe(
        TranscriptionRequest("audio.wav", CANARY, language="nl"),
        on_segment=segments.append,
        on_progress=lambda pct, detail: progress.append(pct),
        on_log=lambda level, msg: logs.append(msg),
    )

    args = context.process.args[0]
    assert context.process.target.__name__ == "canary_proc_entrypoint"
    assert args["language"] == "nl"
    assert args["cpu_threads"] == 2
    assert args["max_segment_seconds"] == 15
    assert args["quantization"] == "int8"
    assert args["model_path"].endswith("canary-1b-v2")
    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.5, 3.0, " Goedemorgen."), (3.5, 6.0, " Tot ziens."),
    ]
    assert all(s.words == () for s in segments)
    assert progress == [50.0]
    assert logs == ["loaded"]
    assert info.duration == 6.0 and info.language == "nl"
    assert not registry.busy


@pytest.mark.parametrize("language, multilingual, reason", [
    (None, False, "auto"), ("auto", False, "auto"), (None, True, "multilingual"),
])
def test_auto_language_falls_back_with_status(monkeypatch, models_dir, language, multilingual, reason):
    plugin = LocalCanaryPlugin(discover_canary_models(models_dir),
                               CanaryWorkerSettings(fallback_language="de"))
    context = _FakeContext([{"type": "result", "ok": True, "info": {}}])
    monkeypatch.setattr(canary_plugin.mp, "get_context", lambda method: context)
    statuses, logs = [], []
    plugin.transcribe(
        TranscriptionRequest("audio.wav", CANARY, language=language, multilingual=multilingual),
        on_segment=lambda s: None,
        on_status=lambda mid, params, level: statuses.append((mid, params, level)),
        on_log=lambda level, msg: logs.append(level),
    )
    assert context.process.args[0]["language"] == "de"
    assert statuses == [("canary_language_fallback", {"lang": "de", "reason": reason}, "warning")]
    assert logs == ["warning"]


def test_unsupported_language_is_rejected_before_starting(monkeypatch, models_dir):
    plugin = LocalCanaryPlugin(discover_canary_models(models_dir))
    context = _FakeContext([])
    monkeypatch.setattr(canary_plugin.mp, "get_context", lambda method: context)
    with pytest.raises(ValueError, match="does not support language 'ja'"):
        plugin.transcribe(TranscriptionRequest("audio.wav", CANARY, language="ja"),
                          on_segment=lambda s: None)
    assert context.process is None


def test_unknown_model_and_gpu_override_are_rejected(models_dir):
    plugin = LocalCanaryPlugin(discover_canary_models(models_dir))
    with pytest.raises(ValueError, match="Unknown local Canary model"):
        plugin.transcribe(TranscriptionRequest("a.wav", ModelRef("local-canary", "x"), language="nl"),
                          on_segment=lambda s: None)
    with pytest.raises(ValueError, match="CPU only"):
        plugin.transcribe(TranscriptionRequest("a.wav", CANARY, language="nl"),
                          on_segment=lambda s: None, device_override="cuda")


# --- cancellation and errors -----------------------------------------------------

def test_cancel_callback_terminates_worker(monkeypatch, models_dir):
    plugin = LocalCanaryPlugin(discover_canary_models(models_dir))
    context = _FakeContext([{"type": "segment", "segment": {"start": 0, "end": 1, "text": " a"}}])
    monkeypatch.setattr(canary_plugin.mp, "get_context", lambda method: context)
    seen = []
    with pytest.raises(InferenceCancelled):
        plugin.transcribe(TranscriptionRequest("a.wav", CANARY, language="nl"),
                          on_segment=seen.append, is_cancelled=lambda: bool(seen))
    assert context.process.terminated
    assert len(seen) == 1


def test_registry_cancel_reaches_active_canary_worker(monkeypatch, models_dir):
    registry = _registry(models_dir)
    plugin = registry.get("local-canary")
    calls = []

    def cancel_during_run():
        calls.append(1)
        if len(calls) == 1:
            registry.cancel()

    # The worker still delivers a message after cancel(); the next loop
    # iteration must then raise InferenceCancelled rather than wait for more.
    context = _FakeContext([{"type": "state", "device": "cpu", "stage": "transcription"}],
                           on_get=cancel_during_run)
    monkeypatch.setattr(canary_plugin.mp, "get_context", lambda method: context)
    with pytest.raises(InferenceCancelled):
        registry.transcribe(TranscriptionRequest("a.wav", CANARY, language="nl"),
                            on_segment=lambda s: None)
    assert context.process.terminated
    assert not registry.busy
    # The cancel flag is cleared for the next job.
    context2 = _FakeContext([{"type": "result", "ok": True, "info": {}}])
    monkeypatch.setattr(canary_plugin.mp, "get_context", lambda method: context2)
    plugin.transcribe(TranscriptionRequest("a.wav", CANARY, language="nl"), on_segment=lambda s: None)


def test_worker_error_result_raises_worker_error(monkeypatch, models_dir):
    registry = _registry(models_dir)
    context = _FakeContext([{
        "type": "result", "ok": False, "error": "model broken",
        "exception_type": "onnx_asr.utils.ModelFileNotFoundError",
        "classification": "other", "device": "cpu", "stage": "model_loading", "trace": "tb",
    }])
    monkeypatch.setattr(canary_plugin.mp, "get_context", lambda method: context)
    with pytest.raises(WorkerError) as caught:
        registry.transcribe(TranscriptionRequest("a.wav", CANARY, language="nl"),
                            on_segment=lambda s: None)
    assert caught.value.stage == "model_loading"
    assert caught.value.trace == "tb"
    assert not caught.value.allows_cpu_retry
    assert not registry.busy


def test_unexpected_worker_exit_is_reported(monkeypatch, models_dir):
    plugin = LocalCanaryPlugin(discover_canary_models(models_dir))
    context = _FakeContext([{"type": "state", "device": "cpu", "stage": "model_loading"}], alive=False)
    monkeypatch.setattr(canary_plugin.mp, "get_context", lambda method: context)
    with pytest.raises(WorkerError) as caught:
        plugin.transcribe(TranscriptionRequest("a.wav", CANARY, language="nl"),
                          on_segment=lambda s: None)
    assert caught.value.classification == "worker_crash"
    assert caught.value.stage == "model_loading"


# --- worker entry point with a mocked onnx_asr ---------------------------------------

class _ListQueue(list):
    def put(self, item):
        self.append(item)


def _install_fake_onnx_asr(monkeypatch, segments, recorded):
    class _VadModel:
        def recognize(self, waveform, **kwargs):
            recorded["waveform_len"] = len(waveform)
            recorded["recognize"] = kwargs
            return iter(segments)

    class _Asr:
        def with_vad(self, vad, **kwargs):
            recorded["vad"] = kwargs
            return _VadModel()

    fake = types.ModuleType("onnx_asr")

    def load_model(name, path, **kwargs):
        recorded["load_model"] = (name, path, kwargs)
        return _Asr()

    def load_vad(name, path, **kwargs):
        recorded["load_vad"] = (name, path)
        return object()

    fake.load_model = load_model
    fake.load_vad = load_vad
    monkeypatch.setitem(sys.modules, "onnx_asr", fake)


def _write_wav(path, seconds=4.0, rate=16000):
    samples = (np.sin(np.arange(int(seconds * rate)) / 10) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())


def test_worker_entrypoint_streams_events(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    from noScribe.canary_mp_worker import canary_proc_entrypoint

    seg = types.SimpleNamespace
    recorded = {}
    _install_fake_onnx_asr(monkeypatch, [
        seg(start=0.1, end=2.0, text="Hallo daar."),
        seg(start=2.2, end=2.5, text="  "),
        seg(start=2.6, end=4.0, text="Tot ziens."),
    ], recorded)
    audio = tmp_path / "a.wav"
    _write_wav(audio)
    q = _ListQueue()
    canary_proc_entrypoint({
        "model_path": str(tmp_path), "quantization": "int8", "cpu_threads": 3,
        "audio_path": str(audio), "language": "nl", "max_segment_seconds": 12,
    }, q)

    assert q[-1]["type"] == "result" and q[-1]["ok"], q[-1]
    assert q[-1]["info"]["language"] == "nl"
    assert q[-1]["info"]["duration"] == pytest.approx(4.0)
    segments = [m["segment"] for m in q if m["type"] == "segment"]
    assert segments == [
        {"start": 0.1, "end": 2.0, "text": " Hallo daar."},
        {"start": 2.6, "end": 4.0, "text": " Tot ziens."},
    ]
    assert [m["pct"] for m in q if m["type"] == "progress"] == pytest.approx([50.0, 62.5, 100.0])
    assert recorded["recognize"]["language"] == "nl"
    assert recorded["recognize"]["pnc"] is True
    assert recorded["recognize"]["sample_rate"] == 16000
    assert recorded["load_model"][0] == "nemo-canary-1b-v2"
    assert recorded["load_model"][2]["quantization"] == "int8"
    assert recorded["load_model"][2]["providers"] == ["CPUExecutionProvider"]
    assert recorded["load_model"][2]["sess_options"].intra_op_num_threads == 3
    assert recorded["load_vad"] == ("silero", os.path.join(str(tmp_path), "silero-vad"))
    assert recorded["vad"]["max_speech_duration_s"] == 12
    assert {m["stage"] for m in q if m["type"] == "state"} >= {"model_loading", "transcription"}


def test_worker_entrypoint_reports_missing_audio(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    from noScribe.canary_mp_worker import canary_proc_entrypoint

    _install_fake_onnx_asr(monkeypatch, [], {})
    q = _ListQueue()
    canary_proc_entrypoint({"model_path": str(tmp_path), "audio_path": str(tmp_path / "x.wav"),
                            "language": "nl"}, q)
    assert q[-1]["ok"] is False
    assert q[-1]["stage"] == "audio_loading"
    assert q[-1]["classification"] == "other"
    assert "FileNotFoundError" in q[-1]["exception_type"]


def test_worker_decodes_non_wav_audio_with_pyav(tmp_path):
    av = pytest.importorskip("av")
    from noScribe.canary_mp_worker import _load_audio

    path = tmp_path / "a.flac"
    rate = 48000
    samples = (np.sin(np.arange(rate * 2) / 20) * 8000).astype(np.int16)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("flac", rate=rate, layout="mono")
        frame = av.AudioFrame.from_ndarray(samples[None, :], format="s16", layout="mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    waveform, sample_rate = _load_audio(str(path))
    assert sample_rate == 16000
    assert waveform.dtype == np.float32
    assert abs(len(waveform) - 32000) < 400


def test_worker_and_plugin_imports_stay_light():
    code = (
        "import sys; import noScribe.canary_mp_worker, noScribe.plugins.local_canary; "
        "heavy = {'numpy', 'onnxruntime', 'onnx_asr', 'torch', 'av'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    env = {**os.environ, "PYTHONPATH": REPO}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_discovery_accepts_importlib_traversable(tmp_path):
    from importlib.resources.readers import MultiplexedPath

    first = _make_canary_dir(tmp_path / "a")
    (tmp_path / "b").mkdir()
    found = discover_canary_models(MultiplexedPath(tmp_path / "a", tmp_path / "b"))
    assert found == {"canary-1b-v2": str(first.absolute())}
