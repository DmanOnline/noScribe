import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from noScribe import pdf_export
from noScribe.recorder import Recorder, default_recording_path

DATA = Path(__file__).parent / "data"


def test_pdf_from_sample_transcript():
    html = (DATA / "interview.html").read_text(encoding="utf-8")
    pdf = pdf_export.html_to_pdf_bytes(html, footer_label="noScribe")
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 5000


def test_pdf_handles_empty_and_non_latin_text():
    assert pdf_export.html_to_pdf_bytes("<html><body></body></html>").startswith(b"%PDF")
    html = "<html><body><p>Titel</p><p>S00: “Hallo” € 5 — 你好</p></body></html>"
    assert pdf_export.html_to_pdf_bytes(html).startswith(b"%PDF")


def test_speaker_detection_in_paragraphs():
    m = pdf_export._SPEAKER_RE.match("S01: [00:00:12] Goedemiddag.")
    assert m and m.group("label") == "S01"
    ts = pdf_export._TIMESTAMP_RE.match(m.group("rest"))
    assert ts and ts.group("ts") == "[00:00:12]"
    # an ordinary sentence with a colon after a full stop is no speaker turn
    assert pdf_export._SPEAKER_RE.match("Dat was het. Daarna: koffie.") is None


def test_default_recording_path_is_unique(tmp_path):
    first = default_recording_path(tmp_path, "Opname")
    first.write_bytes(b"")
    second = default_recording_path(tmp_path, "Opname")
    assert first != second and second.suffix == ".wav"


def test_recorder_writes_blocks_to_wav(tmp_path):
    rec = Recorder(tmp_path / "test.wav", samplerate=16000)
    rec._writer = threading.Thread(target=rec._write_loop)
    rec._writer.start()
    block = (np.sin(np.linspace(0, 100, 1600)) * 0.5).astype("float32").reshape(-1, 1)
    for _ in range(10):
        rec._callback(block, len(block), None, None)
    rec.pause(True)
    rec._callback(block, len(block), None, None)  # ignored while paused
    path = rec.stop()
    data, sr = sf.read(path)
    assert sr == 16000
    assert len(data) == 16000
    assert rec.seconds == 1.0
