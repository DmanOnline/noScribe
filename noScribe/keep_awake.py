"""Keep the computer from going to sleep while recording or transcribing.

A laptop that falls asleep halfway through a one-hour meeting stops the
recording and the transcription. ``hold()`` / ``release()`` are counted, so
the recorder and the transcription queue can each hold the computer awake
independently. Closing the lid can still put a laptop to sleep, depending on
the Windows power settings; this only prevents idle sleep.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_count = 0
_mac_process: subprocess.Popen | None = None
_win_thread: threading.Thread | None = None

# Windows: SetThreadExecutionState(ES_SYSTEM_REQUIRED) resets the idle timer.
# The setting belongs to the calling thread, so one keeper thread repeats it
# while anything holds the computer awake, independent of who called hold().
_ES_SYSTEM_REQUIRED = 0x00000001
_WIN_INTERVAL_S = 30


def _win_keeper() -> None:
    import ctypes
    import time
    while True:
        with _lock:
            if _count == 0:
                return
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_SYSTEM_REQUIRED)
        time.sleep(_WIN_INTERVAL_S)


def _enable() -> None:
    global _mac_process, _win_thread
    system = platform.system()
    if system == "Windows":
        if _win_thread is None or not _win_thread.is_alive():
            _win_thread = threading.Thread(target=_win_keeper, name="keep-awake", daemon=True)
            _win_thread.start()
    elif system == "Darwin":
        _mac_process = subprocess.Popen(["caffeinate", "-i"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _disable() -> None:
    global _mac_process
    # Windows: the keeper thread ends by itself once the count is zero.
    if platform.system() == "Darwin" and _mac_process is not None:
        _mac_process.terminate()
        _mac_process = None


def hold() -> None:
    """Ask the system to stay awake until the matching ``release()``."""
    global _count
    with _lock:
        _count += 1
        first = _count == 1
    if first:
        try:
            _enable()
        except Exception as error:  # never break recording over this
            logger.warning("Could not prevent sleep: %s", error)


def release() -> None:
    global _count
    with _lock:
        if _count == 0:
            return
        _count -= 1
        last = _count == 0
    if last:
        try:
            _disable()
        except Exception as error:
            logger.warning("Could not restore sleep settings: %s", error)
