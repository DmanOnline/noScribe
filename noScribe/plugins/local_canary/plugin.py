"""Plugin adapter for the bundled Canary 1B v2 worker (ONNX on CPU).

Inference runs in a spawned child process (``canary_mp_worker``). The
process plumbing (event loop, cancellation, cleanup) is reused from a private
``LocalInferenceBackend`` instance so it behaves exactly like the Whisper and
Pyannote workers: ``cancel()`` terminates the child, an unexpected exit
becomes a ``WorkerError`` with classification ``worker_crash``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from importlib import resources
from typing import Callable, Mapping

from ...inference import (
    LocalInferenceBackend,
    TranscriptionInfo,
    TranscriptionSegment,
    _ignore_log,
    _ignore_status,
    _ignore_transcription_progress,
    _never_cancel,
)
from ...models import ModelDescriptor, ModelRef
from ...worker_errors import WorkerError
from ..manifest import PluginManifest
from .models import CANARY_LANGUAGES, model_quantization
from .settings import CanaryWorkerSettings

LOCAL_CANARY_BACKEND = "local-canary"


def _load_manifest() -> PluginManifest:
    resource = resources.files(__package__) / "backend.json"
    with resource.open("r", encoding="utf-8") as stream:
        return PluginManifest.from_mapping(json.load(stream))


# Short, readable names for the model menu (it is narrow).
_DISPLAY_NAMES = {"canary-1b-v2": "Canary"}


class LocalCanaryPlugin:
    def __init__(
        self,
        model_paths: Mapping[str, str],
        settings: CanaryWorkerSettings | None = None,
        *,
        settings_store=None,
    ):
        self.manifest = _load_manifest()
        self._model_paths = dict(model_paths)
        self.settings = settings or CanaryWorkerSettings()
        self._settings_store = settings_store
        self.settings_path = settings_store.path_for(self.manifest.id) if settings_store else None
        self._runner = LocalInferenceBackend()

    def list_models(self) -> list[ModelDescriptor]:
        return [
            ModelDescriptor(
                ref=ModelRef(self.manifest.id, model_id),
                display_name=_DISPLAY_NAMES.get(model_id, model_id),
                engine=self.manifest.engine or "canary",
                capabilities=self.manifest.capabilities,
            )
            for model_id in self._model_paths
        ]

    def resolve_language(self, request) -> tuple[str, str | None]:
        """Return ``(language, reason)``; reason is set when falling back.

        Canary has no language identification and cannot follow code
        switching, so Auto and Multilingual use the configured fallback.
        Unsupported explicit languages are rejected instead of guessed.
        """
        if request.multilingual:
            return self.settings.fallback_language, "multilingual"
        if not request.language or request.language == "auto":
            return self.settings.fallback_language, "auto"
        language = str(request.language).lower()
        if language not in CANARY_LANGUAGES:
            raise ValueError(
                f"Canary does not support language {request.language!r}. "
                f"Supported: {', '.join(sorted(CANARY_LANGUAGES))}."
            )
        return language, None

    def transcribe(
        self,
        request,
        on_segment: Callable[[TranscriptionSegment], None],
        on_log=_ignore_log,
        on_status=_ignore_status,
        on_progress=_ignore_transcription_progress,
        is_cancelled=_never_cancel,
        *,
        device_override: str | None = None,
    ) -> TranscriptionInfo:
        from ...canary_mp_worker import canary_proc_entrypoint

        if request.model.backend_id != self.manifest.id:
            raise ValueError(f"Backend {request.model.backend_id!r} cannot be handled by Canary.")
        if device_override not in (None, "cpu"):
            raise ValueError("Canary runs on CPU only.")
        try:
            model_path = self._model_paths[request.model.model_id]
        except KeyError as error:
            raise ValueError(f"Unknown local Canary model: {request.model}") from error

        language, fallback_reason = self.resolve_language(request)
        if fallback_reason is not None:
            on_log("warning",
                   f"Canary cannot detect languages ({fallback_reason}); "
                   f"transcribing as '{language}'. Select the spoken language "
                   f"explicitly or change fallback_language in local-canary.yml.")
            on_status("canary_language_fallback",
                      {"lang": language, "reason": fallback_reason}, "warning")

        worker_args = {
            "model_path": model_path,
            "quantization": model_quantization(model_path),
            "cpu_threads": self.settings.cpu_threads,
            "audio_path": request.audio_path,
            "language": language,
            "vad_threshold": self.settings.vad_threshold,
            "max_segment_seconds": self.settings.max_segment_seconds,
            "min_silence_ms": self.settings.min_silence_ms,
            "extensions": request.extensions,
        }

        runner = self._runner
        runner._cancel_event.clear()
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(target=canary_proc_entrypoint, args=(worker_args, result_queue))
        process.start()
        runner._set_active(process, result_queue)
        info = {}
        worker_state = {"device": "cpu", "stage": "starting"}
        try:
            while True:
                message = runner._next_message(process, result_queue, is_cancelled, worker_state)
                message_type = message.get("type")
                if message_type == "state":
                    worker_state = {"device": message.get("device"), "stage": message.get("stage")}
                elif message_type == "log":
                    on_log(message.get("level", "info"), str(message.get("msg", "")))
                elif message_type == "status":
                    on_status(str(message.get("message_id", "")),
                              message.get("params") or {},
                              str(message.get("level", "info")))
                elif message_type == "progress":
                    if message.get("pct") is not None:
                        on_progress(float(message["pct"]), message.get("detail"))
                elif message_type == "segment":
                    on_segment(TranscriptionSegment.from_mapping(message.get("segment") or {}))
                elif message_type == "result":
                    if not message.get("ok"):
                        raise WorkerError(message)
                    info = message.get("info") or {}
                    break
        finally:
            runner._cleanup(process, result_queue)

        return TranscriptionInfo.from_mapping(info)

    def cancel(self) -> None:
        self._runner.cancel()

    def close(self) -> None:
        self._runner.close()
