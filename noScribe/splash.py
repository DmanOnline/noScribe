"""Status line of the PyInstaller start screen (Windows build only).

The start screen is shown by the bootloader before Python runs. Outside a
frozen Windows build ``pyi_splash`` does not exist and these calls do nothing.
"""

try:
    import pyi_splash  # type: ignore[import-not-found]
except ImportError:
    pyi_splash = None


def splash_status(text: str) -> None:
    if pyi_splash is not None and pyi_splash.is_alive():
        try:
            pyi_splash.update_text(text)
        except Exception:
            pass


def splash_close() -> None:
    if pyi_splash is not None and pyi_splash.is_alive():
        try:
            pyi_splash.close()
        except Exception:
            pass
