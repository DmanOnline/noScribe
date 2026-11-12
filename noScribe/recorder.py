"""Record audio from a microphone straight to a WAV file.

Audio is written to disk while recording, so a long recording does not fill the
memory and what was recorded so far survives a crash. The GUI part lives in
``record_dialog.py``; this module has no Tk dependency so it can be tested on
its own.
"""

from __future__ import annotations

import datetime
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import keep_awake


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    default_samplerate: int
    is_default: bool = False


def list_input_devices() -> list[InputDevice]:
    """Return the microphones the system offers, the default device first."""
    import sounddevice as sd

    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    preferred_api = None
    # On Windows every device appears once per host API (MME, DirectSound,
    # WASAPI, ...). Show only the one of the default device's API to keep the
    # list readable.
    if default_in is not None and default_in >= 0:
        preferred_api = sd.query_devices(default_in)["hostapi"]
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        if preferred_api is not None and info["hostapi"] != preferred_api:
            continue
        devices.append(InputDevice(
            index=index,
            name=str(info["name"]),
            default_samplerate=int(info["default_samplerate"] or 44100),
            is_default=(index == default_in),
        ))
    devices.sort(key=lambda d: not d.is_default)
    return devices


def default_recording_path(folder: Path, prefix: str = "Opname") -> Path:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix} {stamp}.wav"
    n = 2
    while path.exists():
        path = folder / f"{prefix} {stamp} ({n}).wav"
        n += 1
    return path


class Recorder:
    """Records one mono WAV file from an input device.

    ``level`` holds the peak level (0..1) of the most recent audio block, for
    a level meter. ``seconds`` is the recorded duration so far.
    """

    def __init__(self, path: Path, device: int | None = None, samplerate: int | None = None):
        self.path = Path(path)
        self.device = device
        self.samplerate = samplerate
        self.level = 0.0
        self.frames = 0
        self.error: Exception | None = None
        self._queue: queue.Queue = queue.Queue()
        self._stream = None
        self._writer: threading.Thread | None = None
        self._paused = False
        self._holding_awake = False

    @property
    def seconds(self) -> float:
        return self.frames / self.samplerate if self.samplerate else 0.0

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        import sounddevice as sd

        if self.samplerate is None:
            info = sd.query_devices(self.device, "input")
            self.samplerate = int(info["default_samplerate"] or 44100)
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        # a laptop that falls asleep would cut the recording short
        keep_awake.hold()
        self._holding_awake = True

    def pause(self, paused: bool = True) -> None:
        self._paused = paused

    def stop(self) -> Path:
        """Stop recording, finish the file and return its path."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        if self._holding_awake:
            self._holding_awake = False
            keep_awake.release()
        self._queue.put(None)
        if self._writer is not None:
            self._writer.join(timeout=10)
        if self.error is not None:
            raise self.error
        return self.path

    def _callback(self, indata, frames, time_info, status) -> None:
        block = indata.copy()
        self.level = float(np.max(np.abs(block))) if block.size else 0.0
        if not self._paused:
            self._queue.put(block)

    def _write_loop(self) -> None:
        import soundfile as sf

        try:
            with sf.SoundFile(self.path, mode="w", samplerate=self.samplerate,
                              channels=1, subtype="PCM_16") as wav:
                while True:
                    block = self._queue.get()
                    if block is None:
                        break
                    wav.write(block)
                    self.frames += len(block)
        except Exception as error:  # reported by stop()
            self.error = error
