import os
import shutil
import subprocess

import pytest

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

JA_SRT = """1
00:00:01,000 --> 00:00:02,000
家に帰った。
"""

FR_SRT = """1
00:00:01,000 --> 00:00:02,000
Bonjour.
"""

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture(autouse=True)
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    monkeypatch.setattr(m, "_dead_models", set())
    monkeypatch.setitem(m.CFG, "cache", str(tmp_path / "cache" / "c.db"))
    monkeypatch.setitem(m.CFG, "libs", [str(tmp_path / "lib")])
    monkeypatch.setitem(m.CFG, "settle", 0)
    monkeypatch.setitem(m.CFG, "require_audio", ["ja"])
    (tmp_path / "lib" / "Film").mkdir(parents=True)
    return tmp_path


def film_dir(tmp_path):
    return tmp_path / "lib" / "Film"


def cache():
    return m.Cache(m.CFG["cache"])


def make_video(path, audio_lang, subs=()):
    """Tiny mkv with one audio track and the given (srt_text, lang) subtitle tracks."""
    tmp = path.parent / "_src"
    tmp.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=9",
           "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono"]
    for i, (text, _) in enumerate(subs):
        f = tmp / f"{i}.srt"
        f.write_text(text, encoding="utf-8")
        cmd += ["-i", str(f)]
    cmd += ["-map", "0:v", "-map", "1:a"] + sum((["-map", f"{i + 2}"] for i in range(len(subs))), [])
    cmd += ["-t", "9", "-c:v", "libx264", "-c:a", "aac", "-c:s", "srt",
            "-metadata:s:a:0", f"language={audio_lang}"]
    for i, (_, lang) in enumerate(subs):
        cmd += [f"-metadata:s:s:{i}", f"language={lang}"]
    subprocess.run(cmd + [str(path)], check=True)
    shutil.rmtree(tmp)
    return path


# ---------------------------------------------------------------- without ffmpeg

def test_external_srt_without_video(tmp_path):
    d = film_dir(tmp_path)
    src = d / "Film.en.srt"
    src.write_text(SRT, encoding="utf-8")
    assert m.handle(src, cache()) is True
    fr = m.parse_srt(d / "Film.fr.srt")
    assert [c[0] for c in fr] == [1, 2, 3, 4]
    assert fr[1][1] == "00:00:03,000 --> 00:00:04,000"
    assert fr[1][2].count("\n") == 1
    assert (d / "Film.ja.ass").read_text(encoding="utf-8-sig").startswith("[Script Info]")
    assert m.handle(src, cache()) is False  # idempotent


def test_external_variants_count_as_existing(tmp_path):
    d = film_dir(tmp_path)
    (d / "Film.en.srt").write_text(SRT, encoding="utf-8")
    (d / "Film.fre.hi.srt").write_text(FR_SRT, encoding="utf-8")       # Bazarr-style name
    (d / "Film.ja.forced.srt").write_text(JA_SRT, encoding="utf-8")    # forced: does not count
    ext = m.external_subs(d / "Film")
    assert [f.name for f in ext["fr"]] == ["Film.fre.hi.srt"]
    assert "ja" not in ext
    assert m.handle(d / "Film.en.srt", cache()) is True
    assert not (d / "Film.fr.srt").exists()
    assert (d / "Film.ja.srt").exists()


def test_resolve_subtitle_names(tmp_path):
    d = film_dir(tmp_path)
    assert m.resolve(d / "My.Film.2007.eng.hi.srt")[0] == d / "My.Film.2007"
    assert m.resolve(d / "My.Film.2007.mkv")[0] == d / "My.Film.2007"
    assert m.resolve(d / "notes.txt") == (None, None)


def test_retry_then_fallback_model(tmp_path, monkeypatch):
    real, calls = m.call_llm, []

    def flaky(model, s, u):
        calls.append(model)
        if model == "a":
            raise m.RetryableError("HTTP 503")
        return real(model, s, u)

    monkeypatch.setattr(m, "call_llm", flaky)
    src = film_dir(tmp_path) / "Film.en.srt"
    src.write_text(SRT, encoding="utf-8")
    out = m.translate_batch(m.parse_srt(src)[:2], [], "fr", "", cache())
    assert out["1"].startswith("[b]")
    assert calls == ["a", "a", "b"]


def test_unavailable_model_is_skipped(tmp_path, monkeypatch):
    real, calls = m.call_llm, []

    def gone(model, s, u):
        calls.append(model)
        if model == "a":
            raise m.ModelUnavailable("HTTP 404: model retired")
        return real(model, s, u)

    monkeypatch.setattr(m, "call_llm", gone)
    src = film_dir(tmp_path) / "Film.en.srt"
    src.write_text(SRT, encoding="utf-8")
    cues, c = m.parse_srt(src), cache()
    assert m.translate_batch(cues[:2], [], "fr", "", c)["1"].startswith("[b]")
    assert m.translate_batch(cues[2:], [], "fr", "", c)["3"].startswith("[b]")
    assert calls == ["a", "b", "b"]  # 404 is not retried, and the dead model is not tried again


def test_all_models_unavailable_fails_fast(tmp_path, monkeypatch):
    def gone(model, s, u):
        raise m.ModelUnavailable("HTTP 404")

    monkeypatch.setattr(m, "call_llm", gone)
    src = film_dir(tmp_path) / "Film.en.srt"
    src.write_text(SRT, encoding="utf-8")
    with pytest.raises(RuntimeError, match="no usable model"):
        m.translate_batch(m.parse_srt(src), [], "fr", "", cache())


def test_pick_text_track():
    from subtrans.embedded import pick_text_track
    streams = [
        {"index": 2, "codec_name": "ass", "tags": {"language": "eng", "title": "Signs & Songs"}},
        {"index": 3, "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
        {"index": 4, "codec_name": "ass", "tags": {"language": "eng", "title": "Full"}},
        {"index": 5, "codec_name": "subrip", "tags": {"language": "eng", "title": "SDH"},
         "disposition": {"hearing_impaired": 1}},
        {"index": 6, "codec_name": "subrip", "tags": {"language": "jpn"}},
        {"index": 7, "codec_name": "subrip", "tags": {"language": "eng"}, "disposition": {"forced": 1}},
    ]
    assert pick_text_track(streams, "en")["index"] == 5     # subrip beats ass, even SDH
    assert pick_text_track(streams, "ja")["index"] == 6
    assert pick_text_track([streams[1]], "en") is None      # image-only


def test_furigana_okurigana():
    from subtrans.furigana import ruby_segments
    segs = ruby_segments("家に帰った")
    assert ("帰", "かえ") in segs
    assert "".join(s for s, _ in segs) == "家に帰った"
    assert ("何", "なに") in ruby_segments("何してるの")
    assert ("何", "なん") in ruby_segments("何度も")


# ---------------------------------------------------------------- with ffmpeg

@needs_ffmpeg
def test_japanese_audio_embedded_english(tmp_path):
    d = film_dir(tmp_path)
    video = make_video(d / "Film.mkv", "jpn", [(SRT, "eng")])
    assert m.handle(video, cache()) is True
    fr = m.parse_srt(d / "Film.fr.srt")
    assert len(fr) == 4 and "Doraemon" in fr[1][2]
    assert (d / "Film.ja.srt").exists() and (d / "Film.ja.ass").exists()
    assert not (d / "Film.en.srt").exists()  # extracted copy stays in the cache dir
    assert m.scan(cache()) == 0


@needs_ffmpeg
def test_non_japanese_audio_is_skipped(tmp_path):
    d = film_dir(tmp_path)
    video = make_video(d / "Film.mkv", "eng", [(SRT, "eng")])
    assert m.handle(video, cache()) is False
    assert sorted(p.name for p in d.iterdir()) == ["Film.mkv"]


@needs_ffmpeg
def test_no_audio_filter_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setitem(m.CFG, "require_audio", [])
    d = film_dir(tmp_path)
    video = make_video(d / "Film.mkv", "eng", [(SRT, "eng")])
    assert m.handle(video, cache()) is True
    assert (d / "Film.fr.srt").exists()


@needs_ffmpeg
def test_embedded_target_languages_are_reused(tmp_path):
    d = film_dir(tmp_path)
    video = make_video(d / "Film.mkv", "jpn", [(SRT, "eng"), (FR_SRT, "fre"), (JA_SRT, "jpn")])
    assert m.handle(video, cache()) is True
    assert not (d / "Film.fr.srt").exists()   # French already embedded
    assert not (d / "Film.ja.srt").exists()   # Japanese already embedded: not translated...
    ass = (d / "Film.ja.ass").read_text(encoding="utf-8-sig")
    assert "家に帰った" in ass and "かえ" in ass  # ...but furigana built from the real Japanese track
    assert m.handle(video, cache()) is False
