"""Recording window: pick a microphone, record, and hand the WAV to noScribe."""

from __future__ import annotations

import os
import tkinter as tk
import tkinter.messagebox
from pathlib import Path
from typing import Callable

import customtkinter as ctk

from . import branding
from .recorder import Recorder, default_recording_path, list_input_devices


def _format_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class RecordDialog(ctk.CTkToplevel):
    """Modal window that records one file and reports it through ``on_done``."""

    def __init__(self, master, t: Callable[..., str], folder: Path,
                 on_done: Callable[[Path], None], last_device: str = ""):
        super().__init__(master)
        self._t = t
        self._folder = Path(folder)
        self._on_done = on_done
        self._recorder: Recorder | None = None
        self._tick_job = None

        self.title(t("record_title"))
        self.geometry("460x290")
        self.resizable(False, False)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._close)

        try:
            self._devices = list_input_devices()
        except Exception as error:
            self._devices = []
            self._device_error = str(error)
        else:
            self._device_error = ""

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=24, pady=20)

        ctk.CTkLabel(body, text=t("record_device"), anchor="w").pack(fill="x")
        names = [d.name for d in self._devices] or [t("record_no_device")]
        self.device_menu = ctk.CTkOptionMenu(body, values=names, width=410, dynamic_resizing=False)
        self.device_menu.pack(fill="x", pady=(2, 16))
        self.device_menu.set(last_device if last_device in names else names[0])

        self.time_label = ctk.CTkLabel(body, text="00:00:00",
                                       font=ctk.CTkFont(size=40, weight="bold"))
        self.time_label.pack()

        self.level_bar = ctk.CTkProgressBar(body, width=410, height=10,
                                            progress_color=branding.CYAN)
        self.level_bar.set(0)
        self.level_bar.pack(pady=(6, 4))

        self.status_label = ctk.CTkLabel(body, text=t("record_hint"),
                                         text_color=("gray30", "gray70"), wraplength=410)
        self.status_label.pack(pady=(0, 12))

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.pack(fill="x")
        self.record_button = ctk.CTkButton(
            buttons, text="● " + t("record_start"), height=40,
            fg_color=branding.MAGENTA, hover_color=branding.MAGENTA_DARK,
            font=ctk.CTkFont(size=15, weight="bold"), command=self._toggle)
        self.record_button.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.pause_button = ctk.CTkButton(
            buttons, text=t("record_pause"), height=40, width=110,
            state="disabled", command=self._pause)
        self.pause_button.pack(side="left", padx=(0, 6))
        self.cancel_button = ctk.CTkButton(
            buttons, text=t("record_cancel"), height=40, width=100,
            fg_color="transparent", border_width=1,
            text_color=("gray20", "gray85"), command=self._close)
        self.cancel_button.pack(side="left")

        if not self._devices:
            self.record_button.configure(state="disabled")
            self.status_label.configure(
                text=t("record_no_device") + (f"\n{self._device_error}" if self._device_error else ""))

        self.after(100, self._grab)

    # Tk can refuse a grab until the window is visible.
    def _grab(self):
        try:
            self.grab_set()
            self.focus_force()
        except tk.TclError:
            self.after(100, self._grab)

    @property
    def selected_device_name(self) -> str:
        return self.device_menu.get()

    def _selected_device(self):
        name = self.device_menu.get()
        for device in self._devices:
            if device.name == name:
                return device
        return None

    def _toggle(self):
        if self._recorder is None:
            self._start()
        else:
            self._stop()

    def _start(self):
        device = self._selected_device()
        if device is None:
            return
        path = default_recording_path(self._folder, self._t("record_file_prefix"))
        self._recorder = Recorder(path, device=device.index,
                                  samplerate=device.default_samplerate)
        try:
            self._recorder.start()
        except Exception as error:
            self._recorder = None
            self.status_label.configure(text=self._t("record_error", error=str(error)))
            return
        self.device_menu.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self.record_button.configure(text="■ " + self._t("record_stop"),
                                     fg_color=branding.BLUE, hover_color=branding.BLUE_DARK)
        self.status_label.configure(text=self._t("record_running", file=path.name))
        self._tick()

    def _pause(self):
        if self._recorder is None:
            return
        paused = not self._recorder.paused
        self._recorder.pause(paused)
        self.pause_button.configure(text=self._t("record_resume" if paused else "record_pause"))
        self.status_label.configure(
            text=self._t("record_paused") if paused
            else self._t("record_running", file=self._recorder.path.name))

    def _tick(self):
        if self._recorder is None:
            return
        self.time_label.configure(text=_format_time(self._recorder.seconds))
        level = 0.0 if self._recorder.paused else min(1.0, (self._recorder.level ** 0.5) * 1.4)
        self.level_bar.set(level)
        self._tick_job = self.after(100, self._tick)

    def _stop(self):
        recorder, self._recorder = self._recorder, None
        if self._tick_job is not None:
            self.after_cancel(self._tick_job)
        try:
            path = recorder.stop()
        except Exception as error:
            self.status_label.configure(text=self._t("record_error", error=str(error)))
            return
        if recorder.frames == 0:
            path.unlink(missing_ok=True)
            self.status_label.configure(text=self._t("record_empty"))
            self._reset()
            return
        self.grab_release()
        self.destroy()
        self._on_done(path)

    def _reset(self):
        self.device_menu.configure(state="normal")
        self.pause_button.configure(state="disabled", text=self._t("record_pause"))
        self.record_button.configure(text="● " + self._t("record_start"),
                                     fg_color=branding.MAGENTA, hover_color=branding.MAGENTA_DARK)
        self.level_bar.set(0)

    def _close(self):
        if self._recorder is not None:
            if not tk.messagebox.askyesno(title="noScribe", message=self._t("record_discard"),
                                          parent=self):
                return
            recorder, self._recorder = self._recorder, None
            try:
                path = recorder.stop()
                os.remove(path)
            except Exception:
                pass
        if self._tick_job is not None:
            self.after_cancel(self._tick_job)
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
