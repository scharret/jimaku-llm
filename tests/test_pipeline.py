import os, shutil, time
from pathlib import Path

os.environ.update(BACKEND="dummy", MODELS="a,b", MIN_INTERVAL_S="0", MAX_RETRIES="2", BATCH_SIZE="3")

import subtrans.main as m  # noqa: E402

SRT = """1
00:00:01,000 --> 00:00:02,000
Hello.

2
00:00:03,000 --> 00:00:04,000
- Where is Doraemon?
- I don't know.

3
00:00:05,000 --> 00:00:06,000
<i>Wind the bobbin up</i>

4
00:00:07,000 --> 00:00:08,000
Bye.
"""


def setup(tmp_path):
    m.CFG["cache"] = str(tmp_path / "c.db")
    m.CFG["libs"] = [str(tmp_path / "lib")]
    m.CFG["settle"] = 0
    d = tmp_path / "lib" / "Film"
    d.mkdir(parents=True)
    p = d / "Film.en.srt"
    p.write_text(SRT, encoding="utf-8")
    (d / "Film.mkv").write_bytes(b"")
    return p


def test_scan_writes_outputs_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    p = setup(tmp_path)
    cache = m.Cache(m.CFG["cache"])
    assert m.handle(p, cache) is True
    fr = m.parse_srt(p.with_name("Film.fr.srt"))
    assert [c[0] for c in fr] == [1, 2, 3, 4]
    assert fr[1][1] == "00:00:03,000 --> 00:00:04,000"
    assert fr[1][2].count("\n") == 1
    assert p.with_name("Film.ja.ass").read_text(encoding="utf-8-sig").startswith("[Script Info]")
    assert m.handle(p, cache) is False
    assert m.scan(cache) == 0


def test_retry_then_fallback_model(tmp_path, monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    p = setup(tmp_path)
    real, calls = m.call_llm, []

    def flaky(model, s, u):
        calls.append(model)
        if model == "a":
            raise m.RetryableError("HTTP 503")
        return real(model, s, u)

    monkeypatch.setattr(m, "call_llm", flaky)
    cues = m.parse_srt(p)
    out = m.translate_batch(cues[:2], [], "fr", "", m.Cache(m.CFG["cache"]))
    assert out["1"].startswith("[b]")
    assert calls == ["a", "a", "b"]


def test_furigana_okurigana():
    from subtrans.furigana import ruby_segments
    segs = ruby_segments("家に帰った")
    assert ("帰", "かえ") in segs
    assert "".join(s for s, _ in segs) == "家に帰った"
    assert ("何", "なに") in ruby_segments("何してるの")
    assert ("何", "なん") in ruby_segments("何度も")


def test_pick_english_track():
    from subtrans.embedded import pick_english_track
    streams = [
        {"index": 2, "codec_name": "ass", "tags": {"language": "eng", "title": "Signs & Songs"}},
        {"index": 3, "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
        {"index": 4, "codec_name": "ass", "tags": {"language": "eng", "title": "Full"}},
        {"index": 5, "codec_name": "subrip", "tags": {"language": "eng", "title": "SDH"},
         "disposition": {"hearing_impaired": 1}},
        {"index": 6, "codec_name": "subrip", "tags": {"language": "jpn"}},
        {"index": 7, "codec_name": "subrip", "tags": {"language": "eng"}, "disposition": {"forced": 1}},
    ]
    assert pick_english_track(streams)["index"] == 5          # subrip beats ass, even SDH
    assert pick_english_track(streams[:3])["index"] is not None
    assert pick_english_track([streams[1]]) is None            # image-only


def test_embedded_extraction(tmp_path, monkeypatch):
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        import pytest; pytest.skip("ffmpeg not installed")
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    m.CFG["cache"] = str(tmp_path / "cache" / "c.db")
    d = tmp_path / "lib" / "Film"; d.mkdir(parents=True)
    srt = tmp_path / "in.srt"; srt.write_text(SRT, encoding="utf-8")
    video = d / "Film.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=9",
                    "-i", str(srt), "-map", "0", "-map", "1", "-c:v", "libx264", "-c:s", "srt",
                    "-metadata:s:s:0", "language=eng", str(video)], check=True)
    cache = m.Cache(m.CFG["cache"])
    assert m.handle(video, cache) is True
    fr = m.parse_srt(d / "Film.fr.srt")
    assert len(fr) == 4 and "Doraemon" in fr[1][2]
    assert not (d / "Film.en.srt").exists()   # extracted copy stays in the cache dir


def test_unavailable_model_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    monkeypatch.setattr(m, "_dead_models", set())
    p = setup(tmp_path)
    real, calls = m.call_llm, []

    def gone(model, s, u):
        calls.append(model)
        if model == "a":
            raise m.ModelUnavailable("HTTP 404: model retired")
        return real(model, s, u)

    monkeypatch.setattr(m, "call_llm", gone)
    cache = m.Cache(m.CFG["cache"])
    cues = m.parse_srt(p)
    assert m.translate_batch(cues[:2], [], "fr", "", cache)["1"].startswith("[b]")
    assert m.translate_batch(cues[2:], [], "fr", "", cache)["3"].startswith("[b]")
    assert calls == ["a", "b", "b"]   # 404 is not retried, and the dead model is not tried again


def test_all_models_unavailable_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    monkeypatch.setattr(m, "_dead_models", set())
    p = setup(tmp_path)

    def gone(model, s, u):
        raise m.ModelUnavailable("HTTP 404")

    monkeypatch.setattr(m, "call_llm", gone)
    import pytest
    with pytest.raises(RuntimeError, match="no usable model"):
        m.translate_batch(m.parse_srt(p), [], "fr", "", m.Cache(m.CFG["cache"]))
