"""Inspect video containers (mkv/mp4) with ffprobe and extract embedded text subtitle tracks with ffmpeg."""
import json
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("subtrans")

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
IMAGE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}
SKIP_TITLE_WORDS = ("sign", "song", "forced", "forcé", "karaoke", "commentary")
# A title that explicitly says "full" wins over a (frequently wrong) forced flag, e.g. "Français Complets".
FULL_TITLE_WORDS = ("complet", "full", "intégra", "integra", "entier")

# ISO 639-1 / 639-2 (B and T) codes and common names, all lowercase.
LANG_ALIASES = {
    "en": {"en", "eng", "english"},
    "fr": {"fr", "fre", "fra", "french", "francais", "français"},
    "ja": {"ja", "jpn", "jp", "japanese"},
    "es": {"es", "spa", "spanish"},
    "de": {"de", "ger", "deu", "german"},
    "it": {"it", "ita", "italian"},
}


def aliases(lang):
    lang = lang.lower()
    for codes in LANG_ALIASES.values():
        if lang in codes:
            return codes
    return {lang}


def normalize(code):
    """'jpn' -> 'ja', 'fre' -> 'fr', unknown codes unchanged."""
    code = (code or "").lower()
    for short, codes in LANG_ALIASES.items():
        if code in codes:
            return short
    return code


def probe(video: Path):
    """All audio and subtitle streams of a video, as returned by ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries",
         "stream=index,codec_type,codec_name:stream_tags=language,title"
         ":stream_disposition=default,forced,hearing_impaired",
         "-of", "json", str(video)],
        capture_output=True, text=True, timeout=120, check=True)
    return [s for s in json.loads(out.stdout).get("streams", [])
            if s.get("codec_type") in ("audio", "subtitle")]


def _tags(s):
    return s.get("tags", {}) or {}


def _disp(s):
    return s.get("disposition", {}) or {}


def _is_forced(s):
    title = (_tags(s).get("title") or "").lower()
    if any(w in title for w in FULL_TITLE_WORDS):
        return False
    return bool(_disp(s).get("forced"))


def _skip_title(s):
    title = (_tags(s).get("title") or "").lower()
    if any(w in title for w in FULL_TITLE_WORDS):
        return False
    return any(w in title for w in SKIP_TITLE_WORDS)


def stream_lang(s):
    return normalize(_tags(s).get("language"))


def audio_langs(streams):
    return {stream_lang(s) for s in streams if s.get("codec_type") == "audio"}


def full_text_sub_langs(streams):
    """Languages that already have a complete (non-forced, text) subtitle track in the container."""
    return {stream_lang(s) for s in streams
            if s.get("codec_type") == "subtitle" and s.get("codec_name") in TEXT_CODECS
            and not _is_forced(s) and not _skip_title(s)}


def available_sub_langs(streams):
    """Languages with a complete subtitle track, text or image (PGS/VobSub): no need to translate them."""
    return full_text_sub_langs(streams) | {
        stream_lang(s) for s in streams
        if s.get("codec_type") == "subtitle" and s.get("codec_name") in IMAGE_CODECS
        and not _is_forced(s) and not _skip_title(s)}


def pick_text_track(streams, lang):
    """Best full text track in `lang`: not forced, not signs/songs, SubRip first, plain before SDH."""
    codes = aliases(lang)
    subs = [s for s in streams if s.get("codec_type", "subtitle") == "subtitle"
            and (_tags(s).get("language") or "").lower() in codes]
    text = [s for s in subs if s.get("codec_name") in TEXT_CODECS]
    if not text:
        if any(s.get("codec_name") in IMAGE_CODECS for s in subs):
            log.info("only image-based %s subs (PGS/VobSub): OCR not supported, skipping", lang)
        return None
    usable = [s for s in text
              if not _is_forced(s) and not _skip_title(s)]
    if not usable:
        return None
    usable.sort(key=lambda s: (
        s.get("codec_name") not in ("subrip", "srt"),   # SubRip first (ASS carries signs/karaoke noise)
        bool(_disp(s).get("hearing_impaired")),         # plain over SDH
        s["index"]))
    return usable[0]


def pick_english_track(streams, languages=None):
    """Kept for backward compatibility."""
    return pick_text_track(streams, "en")


def extract_srt(video: Path, dest: Path, lang, streams=None):
    """Write the best `lang` text track to dest as SRT. Returns True on success, False if none usable."""
    track = pick_text_track(streams if streams is not None else probe(video), lang)
    if track is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("extracting embedded %s subtitle stream #%s (%s) from %s",
             lang, track["index"], track.get("codec_name"), video.name)
    tmp = Path(str(dest) + ".tmp.srt")
    if track.get("codec_name") in ("ass", "ssa"):
        # Keep the ASS as-is and clean it ourselves: a plain ffmpeg conversion turns every
        # karaoke/sign effect line into a cue (100k+ cues on some fansubs).
        from .ass_clean import ass_to_cues, render_srt
        raw = Path(str(dest) + ".tmp.ass")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
             "-map", f"0:{track['index']}", "-c:s", "copy", str(raw)],
            check=True, timeout=1800)
        cues, stats = ass_to_cues(raw.read_text(encoding="utf-8-sig", errors="replace"))
        raw.unlink()
        log.info("ASS cleanup: %d events -> %d dialogue cues", stats["events"], stats["cues"])
        if not cues:
            return False
        tmp.write_text(render_srt(cues), encoding="utf-8")
    else:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
             "-map", f"0:{track['index']}", "-c:s", "srt", str(tmp)],
            check=True, timeout=1800)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        return False
    os.replace(tmp, dest)
    return True


def extract_english_srt(video: Path, dest: Path):
    """Kept for backward compatibility."""
    return extract_srt(video, dest, "en")
