from noScribe import keep_awake


def test_hold_release_is_counted(monkeypatch):
    calls = []
    monkeypatch.setattr(keep_awake, "_enable", lambda: calls.append("on"))
    monkeypatch.setattr(keep_awake, "_disable", lambda: calls.append("off"))
    keep_awake.hold()      # recorder
    keep_awake.hold()      # transcription
    keep_awake.release()   # recording stopped, transcription still running
    assert calls == ["on"]
    keep_awake.release()
    assert calls == ["on", "off"]
    keep_awake.release()   # extra release is ignored
    assert calls == ["on", "off"]
    assert keep_awake._count == 0


def test_failure_to_enable_does_not_raise(monkeypatch):
    def boom():
        raise OSError("no power API")
    monkeypatch.setattr(keep_awake, "_enable", boom)
    keep_awake.hold()
    keep_awake.release()
    assert keep_awake._count == 0
