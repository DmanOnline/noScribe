"""Settings schema owned by the bundled Canary plugin (``local-canary.yml``)."""

from dataclasses import asdict, dataclass

from ..settings import settings_values, validate_device
from .models import CANARY_LANGUAGES


@dataclass
class CanaryWorkerSettings:
    # Canary runs on CPU only in noScribe; the field exists so consumers that
    # read ``plugin.settings.device`` (CPU-retry logic) see an explicit value.
    device: str = "cpu"
    cpu_threads: int = 4
    # Canary has no language identification. This language is used when the
    # job language is "Auto" or "Multilingual"; the worker reports it.
    fallback_language: str = "nl"
    vad_threshold: float = 0.5
    # Upper bound for one VAD chunk. Canary handles ~40 s, but shorter chunks
    # give finer segment timestamps for mapping Pyannote speakers.
    max_segment_seconds: float = 20.0
    min_silence_ms: int = 300

    def __post_init__(self):
        validate_device(self.device, {"cpu"})
        if type(self.cpu_threads) is not int or self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be a positive integer.")
        if self.fallback_language not in CANARY_LANGUAGES:
            raise ValueError(
                "fallback_language must be one of: " + ", ".join(sorted(CANARY_LANGUAGES)) + "."
            )
        if (type(self.vad_threshold) not in (float, int)
                or not 0 < self.vad_threshold < 1):
            raise ValueError("vad_threshold must be between 0 and 1.")
        if (type(self.max_segment_seconds) not in (float, int)
                or not 2 <= self.max_segment_seconds <= 40):
            raise ValueError("max_segment_seconds must be between 2 and 40.")
        if type(self.min_silence_ms) is not int or not 0 <= self.min_silence_ms <= 5000:
            raise ValueError("min_silence_ms must be an integer between 0 and 5000.")

    @classmethod
    def from_mapping(cls, value: dict, defaults=None):
        values = asdict(defaults) if defaults is not None else {}
        values.update(settings_values(value, {
            "device", "cpu_threads", "fallback_language", "vad_threshold",
            "max_segment_seconds", "min_silence_ms",
        }))
        return cls(**values)
