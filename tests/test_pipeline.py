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


def ass_events(ass, style):
    """[(x, text)] of the Dialogue events of one style."""
    import re
    out = []
    for line in ass.splitlines():
        if line.startswith("Dialogue:") and line.split(",")[3] == style:
            text = line.split(",", 9)[9]
            out.append((int(re.search(r"\\pos\((\d+),", text)[1]), re.sub(r"\{[^}]*\}", "", text)))
    return out


def test_furigana_keeps_spaces():
    from subtrans.furigana import ruby_segments
    line = "〈風見〉 それも 事情聴取で伺います"
    assert "".join(s for s, _ in ruby_segments(line)) == line


def test_furigana_ruby_centred_on_its_kanji():
    # Regression: widths assumed 1 kanji = Fontsize, but libass renders Noto Sans CJK JP
    # at Fontsize / 1.448 per kanji, so the ruby drifted away from its kanji.
    from subtrans.furigana import EM_RATIO, FS, X0, _width, cues_to_ass
    line = "〈風見) それも 事情聴取で 伺います"
    ass = cues_to_ass([(1, "00:00:01,000 --> 00:00:02,000", line)])
    base, ruby = dict((t, x) for x, t in ass_events(ass, "Base")), ass_events(ass, "Ruby")
    assert [r for _, r in ruby] == ["かざみ", "じじょう", "ちょうしゅ", "うかが"]
    em = FS * EM_RATIO
    for (x, _), kanji in zip(ruby, ["風見", "事情", "聴取", "伺"]):
        assert abs(base[kanji] + len(kanji) * em / 2 - x) <= 1   # ruby centred on its own word
    assert "".join(t for _, t in ass_events(ass, "Base")).replace(" ", "") == line.replace(" ", "")
    width = sum(_width(c) for c in line) * em
    assert abs(min(base.values()) - (X0 - width / 2)) <= 1         # line centred on its real width


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
    text = "".join(t for _, t in ass_events(ass, "Base"))
    assert "家に帰った" in text and "かえ" in ass  # ...but furigana built from the real Japanese track
    assert m.handle(video, cache()) is False


def test_overload_stops_without_bisecting(tmp_path, monkeypatch):
    calls = []

    def busy(model, s, u):
        calls.append(model)
        raise m.Overloaded("HTTP 429", quota=True)

    monkeypatch.setattr(m, "call_llm", busy)
    src = film_dir(tmp_path) / "Film.en.srt"
    src.write_text(SRT * 1, encoding="utf-8")
    cues = m.parse_srt(src) * 5  # 20 cues: would be bisected if this were a format error
    cues = [(i + 1, tc, t) for i, (_, tc, t) in enumerate(cues)]
    with pytest.raises(m.ProviderBusy):
        m.translate_batch(cues, [], "fr", "", cache())
    assert len(calls) == 2 * m.CFG["max_retries"]  # models x retries, no extra requests from bisection


def test_bad_output_is_bisected(tmp_path, monkeypatch):
    real = m.call_llm

    def picky(model, s, u):
        import json as _j
        items = _j.loads(u.split("TRANSLATE:\n", 1)[1])
        if len(items) > 10:
            return "not json"
        return real(model, s, u)

    monkeypatch.setattr(m, "call_llm", picky)
    src = film_dir(tmp_path) / "Film.en.srt"
    src.write_text(SRT, encoding="utf-8")
    cues = [(i + 1, tc, t) for i, (_, tc, t) in enumerate(m.parse_srt(src) * 5)]
    out = m.translate_batch(cues, [], "fr", "", cache())
    assert len(out) == 20 and all(v.startswith("[") for v in out.values())


def test_file_mode_exits_75_when_provider_busy(tmp_path, monkeypatch):
    def busy(model, s, u):
        raise m.Overloaded("HTTP 503")

    monkeypatch.setattr(m, "call_llm", busy)
    src = film_dir(tmp_path) / "Film.en.srt"
    src.write_text(SRT, encoding="utf-8")
    assert m.main(["file", str(src)]) == 75


def test_french_source_when_no_english(tmp_path):
    d = film_dir(tmp_path)
    src = d / "Film.fr.srt"
    src.write_text(FR_SRT, encoding="utf-8")
    assert m.handle(src, cache()) is True
    assert not (d / "Film.en.srt").exists()
    ja = m.parse_srt(d / "Film.ja.srt")
    assert ja[0][2] == "[a] Bonjour."               # translated from the French file
    assert (d / "Film.ja.ass").exists()


def test_own_translation_is_never_a_source(tmp_path):
    d = film_dir(tmp_path)
    (d / "Film.en.srt").write_text(SRT, encoding="utf-8")
    c = cache()
    assert m.handle(d / "Film.en.srt", c) is True      # writes our Film.fr.srt / Film.ja.srt
    assert m.is_ours(d / "Film.fr.srt", c)
    for f in ("Film.en.srt", "Film.ja.srt", "Film.ja.ass"):
        (d / f).unlink()
    # Only our machine-translated French is left: it must not be re-translated into Japanese.
    assert m.handle(d / "Film.fr.srt", c) is False
    assert not (d / "Film.ja.srt").exists()


def test_replaced_file_is_no_longer_ours(tmp_path):
    d = film_dir(tmp_path)
    f = d / "Film.fr.srt"
    f.write_text(FR_SRT, encoding="utf-8")
    c = cache()
    m.mark_ours(f, c)
    f.write_text(FR_SRT + "\n2\n00:00:03,000 --> 00:00:04,000\nSalut.\n", encoding="utf-8")  # e.g. Bazarr download
    assert not m.is_ours(f, c)


@needs_ffmpeg
def test_embedded_french_only_gives_japanese(tmp_path):
    d = film_dir(tmp_path)
    video = make_video(d / "Film.mkv", "jpn", [(FR_SRT, "fre")])
    assert m.handle(video, cache()) is True
    assert not (d / "Film.fr.srt").exists()           # French already embedded
    assert m.parse_srt(d / "Film.ja.srt")[0][2] == "[a] Bonjour."
    assert (d / "Film.ja.ass").exists()


ASS = r"""[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: OP-Romaji,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,8,10,10,10,1
Style: Sign,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,8,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,Where is {\i1}Conan{\i0}?
Dialogue: 1,0:00:01.00,0:00:02.50,Default,,0,0,0,,Where is {\i1}Conan{\i0}?
Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,I don't know.\NLet's go.
Dialogue: 0,0:00:01.00,0:00:01.04,OP-Romaji,,0,0,0,,{\k10}ka{\k12}ze
Dialogue: 0,0:00:01.04,0:00:01.08,OP-Romaji,,0,0,0,,{\k10}ka{\k12}ze
Dialogue: 0,0:00:02.00,0:00:02.04,Default,,0,0,0,,{\pos(320,40)\fad(0,200)}KUDO DETECTIVE AGENCY
Dialogue: 0,0:00:02.00,0:00:02.04,Sign,,0,0,0,,Tokyo Tower
Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,Karaoke,la la la
Comment: 0,0:00:05.00,0:00:06.00,Default,,0,0,0,,note to self
"""


def test_ass_cleanup_keeps_dialogue_only():
    from subtrans.ass_clean import ass_to_cues
    cues, stats = ass_to_cues(ASS)
    assert stats == {"events": 8, "cues": 2}
    assert cues[0] == (1, "00:00:01,000 --> 00:00:02,500", "Where is Conan?")
    assert cues[1][2] == "I don't know.\nLet's go."


def test_max_cues_guard(tmp_path, monkeypatch):
    monkeypatch.setitem(m.CFG, "max_cues", 3)
    d = film_dir(tmp_path)
    (d / "Film.en.srt").write_text(SRT, encoding="utf-8")  # 4 cues
    assert m.handle(d / "Film.en.srt", cache()) is False
    assert not (d / "Film.fr.srt").exists()


@needs_ffmpeg
def test_embedded_ass_is_cleaned(tmp_path):
    d = film_dir(tmp_path)
    ass = tmp_path / "in.ass"
    ass.write_text(ASS, encoding="utf-8")
    video = d / "Film.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=9",
                    "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-i", str(ass),
                    "-map", "0:v", "-map", "1:a", "-map", "2", "-t", "9", "-c:v", "libx264", "-c:a", "aac",
                    "-c:s", "ass", "-metadata:s:a:0", "language=jpn", "-metadata:s:s:0", "language=eng",
                    str(video)], check=True)
    assert m.handle(video, cache()) is True
    fr = m.parse_srt(d / "Film.fr.srt")
    assert [c[2] for c in fr] == ["[a] Where is Conan?", "[a] I don't know.\nLet's go."]


def test_report_tracks_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(m.CFG, "max_cues", 3)
    d = film_dir(tmp_path)
    (d / "Film.en.srt").write_text(SRT, encoding="utf-8")
    c = cache()
    m.handle(d / "Film.en.srt", c)
    rows = c.statuses(reason="too_many_cues")
    assert len(rows) == 1 and "4 cues > MAX_CUES=3" in rows[0][3]
    assert m.main(["report"]) == 0
    out = capsys.readouterr().out
    assert "skipped/too_many_cues=1" in out and "Film" in out
    monkeypatch.setitem(m.CFG, "max_cues", 5000)
    m.handle(d / "Film.en.srt", c)                 # fixed: now translated, status updated
    assert c.statuses(state="skipped") == []
    assert c.statuses(state="done")[0][2] == "translated"


@needs_ffmpeg
def test_report_audio_language(tmp_path):
    d = film_dir(tmp_path)
    video = make_video(d / "Film.mkv", "eng", [(SRT, "eng")])
    c = cache()
    m.handle(video, c)
    assert c.statuses(reason="audio_language")[0][3] == "audio: en"


def test_metrics_exposition(tmp_path, monkeypatch):
    import socket
    import urllib.request
    from subtrans import metrics
    monkeypatch.setattr(metrics, "_counters", {})
    d = film_dir(tmp_path)
    (d / "Film.en.srt").write_text(SRT, encoding="utf-8")
    c = cache()
    m.handle(d / "Film.en.srt", c)
    m.handle(d / "Film.en.srt", c, force=True)        # second pass served from the cache
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    srv = metrics.serve(port, m.CFG["cache"])
    try:
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    finally:
        srv.shutdown()
    assert '# TYPE jimaku_llm_requests_total counter' in body
    assert 'jimaku_llm_requests_total{model="a",outcome="ok"}' in body
    assert 'jimaku_cues_translated_total{lang="fr"} 4' in body
    assert 'jimaku_files_written_total{kind="ja.ass"} 2' in body
    assert 'jimaku_cache_hits_total' in body
    assert 'jimaku_videos{reason="translated",state="done"} 1' in body


# ---------------------------------------------------------------- OCR (PGS)
from tests.pgs_encode import encode as encode_pgs  # noqa: E402

LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
needs_tesseract = pytest.mark.skipif(not shutil.which("tesseract") or not os.path.exists(LATIN_FONT),
                                     reason="tesseract or test font missing")
PGS_SUBS = [(1.0, 2.5, "Where is Conan?"), (3.0, 4.5, "I don't know.")]


@needs_tesseract
def test_pgs_parse_and_tesseract():
    from subtrans.ocr import ocr_tesseract, parse_sup, render
    sets = list(parse_sup(encode_pgs(PGS_SUBS, LATIN_FONT)))
    assert [(round(a, 2), round(b, 2)) for a, b, _ in sets] == [(1.0, 2.5), (3.0, 4.5)]
    texts = ocr_tesseract([render(objs) for _, _, objs in sets], "en")
    assert "Conan" in texts[0] and "know" in texts[1]


def mux_pgs(tmp_path, video, audio_lang, tracks):
    """tracks: [(sup_bytes, lang)]"""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:d=6",
           "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono"]
    for i, (sup, _) in enumerate(tracks):
        f = tmp_path / f"t{i}.sup"
        f.write_bytes(sup)
        cmd += ["-i", str(f)]
    cmd += ["-map", "0:v", "-map", "1:a"] + sum((["-map", f"{i + 2}"] for i in range(len(tracks))), [])
    cmd += ["-t", "6", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-c:s", "copy",
            "-metadata:s:a:0", f"language={audio_lang}"]
    for i, (_, lang) in enumerate(tracks):
        cmd += [f"-metadata:s:s:{i}", f"language={lang}"]
    subprocess.run(cmd + [str(video)], check=True)
    return video


@needs_ffmpeg
@needs_tesseract
def test_pgs_english_is_ocrd_then_translated(tmp_path):
    d = film_dir(tmp_path)
    video = mux_pgs(tmp_path, d / "Film.mkv", "jpn", [(encode_pgs(PGS_SUBS, LATIN_FONT), "eng")])
    c = cache()
    assert m.handle(video, c) is True
    fr = m.parse_srt(d / "Film.fr.srt")
    assert len(fr) == 2 and "Conan" in fr[0][2] and fr[0][2].startswith("[a] ")
    assert (d / "Film.ja.ass").exists()
    assert c.statuses(state="done")[0][2] == "translated"


@needs_ffmpeg
@needs_tesseract
def test_pgs_japanese_uses_vision_and_is_not_translated(tmp_path, monkeypatch):
    monkeypatch.setitem(m.CFG, "furigana_strip_sdh", False)  # the dummy OCR text is "[vision ...]"
    d = film_dir(tmp_path)
    video = mux_pgs(tmp_path, d / "Film.mkv", "jpn", [(encode_pgs(PGS_SUBS, LATIN_FONT), "eng"),
                                                     (encode_pgs(PGS_SUBS, LATIN_FONT), "jpn")])
    assert m.handle(video, cache()) is True
    assert not (d / "Film.ja.srt").exists()            # Japanese PGS exists: not translated
    ass = (d / "Film.ja.ass").read_text(encoding="utf-8-sig")
    assert "[vision a] 1" in ass                       # furigana built from the OCR'd Japanese (vision engine)
    assert (d / "Film.fr.srt").exists()                # French still translated (from OCR'd English)


@needs_tesseract
def test_pgs_repeated_display_sets_are_merged():
    from subtrans.ocr import ocr_tesseract, parse_sup, render
    sets = list(parse_sup(encode_pgs(PGS_SUBS, LATIN_FONT, repeats=5)))
    assert len(sets) == 2                                      # not 2 x 6 images
    assert round(sets[0][0], 2) == 1.0 and round(sets[0][1], 2) == 2.5
    texts = ocr_tesseract([render(objs) for _, _, objs in sets], "en")
    assert "Conan" in texts[0]                                 # opaque palette kept, not the faded one


@needs_ffmpeg
def test_ocr_image_cap(tmp_path, monkeypatch):
    if not os.path.exists(LATIN_FONT):
        pytest.skip("font missing")
    monkeypatch.setitem(m.CFG, "max_cues", 1)
    d = film_dir(tmp_path)
    video = mux_pgs(tmp_path, d / "Film.mkv", "jpn", [(encode_pgs(PGS_SUBS, LATIN_FONT), "jpn")])
    calls = []
    monkeypatch.setattr(m, "call_vision", lambda *a: calls.append(a) or "{}")
    with pytest.raises(Exception, match="MAX_CUES"):
        m.handle(video, cache())
    assert calls == []                                         # nothing sent to the API


# ---------------------------------------------------------------- budget / media resolution
def test_ocr_media_resolution(monkeypatch):
    monkeypatch.setitem(m.CFG, "ocr_media_resolution", "low")
    assert m.vision_config()["mediaResolution"] == "MEDIA_RESOLUTION_LOW"
    monkeypatch.setitem(m.CFG, "ocr_media_resolution", "")
    assert "mediaResolution" not in m.vision_config()


def test_cost_and_unknown_model_price():
    from subtrans import budget
    prices = budget.parse_prices("my-model=1/10")
    assert budget.cost("gemini-3.5-flash-lite", 1_000_000, 0, prices) == pytest.approx(0.30)
    assert budget.cost("my-model", 1_000_000, 1_000_000, prices) == pytest.approx(11.0)
    assert budget.cost("never-heard-of", 1_000_000, 0, prices) == pytest.approx(1.50)


def test_daily_budget_blocks_requests(tmp_path, monkeypatch):
    sent = []

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                    "usageMetadata": {"promptTokenCount": 500_000, "candidatesTokenCount": 100_000,
                                      "thoughtsTokenCount": 100_000}}

    monkeypatch.setattr(m.requests, "post", lambda *a, **k: sent.append(1) or Resp())
    monkeypatch.setitem(m.CFG, "backend", "gemini")
    monkeypatch.setitem(m.CFG, "daily_budget_usd", 0.5)
    cache()  # create the DB
    m.call_llm("gemini-3.5-flash-lite", "s", "u")   # 0.5M*0.30 + 0.2M*2.50 = 0.65 USD
    from subtrans import budget
    assert budget.spent_today(m.CFG["cache"]) == pytest.approx(0.65)
    with pytest.raises(m.BudgetExceeded):
        m.call_llm("gemini-3.5-flash-lite", "s", "u")
    assert len(sent) == 1                             # the second request never left
    assert m.main(["usage"]) == 0
def test_vision_prompt_ignores_furigana_for_japanese():
    from subtrans.ocr import vision_prompt
    assert "furigana" in vision_prompt("ja")
    assert "furigana" not in vision_prompt("fr")


def test_full_title_overrides_forced_flag():
    from subtrans.embedded import available_sub_langs, pick_text_track
    mislabelled = {"index": 3, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle",
                   "tags": {"language": "fre", "title": "Français Complets PGS"},
                   "disposition": {"forced": 1, "default": 1}}
    really_forced = {"index": 4, "codec_type": "subtitle", "codec_name": "subrip",
                     "tags": {"language": "jpn", "title": "Forcés"}, "disposition": {"forced": 1}}
    assert available_sub_langs([mislabelled, really_forced]) == {"fr"}
    full_text = dict(mislabelled, codec_name="subrip")
    assert pick_text_track([full_text], "fr")["index"] == 3


def test_furigana_strips_sdh(tmp_path):
    d = film_dir(tmp_path)
    (d / "Film.ja.hi.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n［銃声］\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n（男）家に帰った。\n\n", encoding="utf-8")
    (d / "Film.en.srt").write_text(SRT, encoding="utf-8")
    assert m.handle(d / "Film.en.srt", cache()) is True
    ass = (d / "Film.ja.ass").read_text(encoding="utf-8-sig")
    text = "".join(t for _, t in ass_events(ass, "Base"))
    assert "銃声" not in ass and "（男）" not in ass and text == "家に帰った。"
    starts = {line.split(",")[1] for line in ass.splitlines() if line.startswith("Dialogue: 0,")}
    assert starts == {"0:00:03.00"}   # the sound-only cue was dropped
