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
    return p


def test_scan_writes_outputs_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    p = setup(tmp_path)
    cache = m.Cache(m.CFG["cache"])
    assert m.process(p, cache) is True
    fr = m.parse_srt(p.with_name("Film.fr.srt"))
    assert [c[0] for c in fr] == [1, 2, 3, 4]
    assert fr[1][1] == "00:00:03,000 --> 00:00:04,000"
    assert fr[1][2].count("\n") == 1
    assert p.with_name("Film.ja.ass").read_text(encoding="utf-8-sig").startswith("[Script Info]")
    assert m.process(p, cache) is False


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
