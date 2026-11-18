"""Extract an embedded English text subtitle track from a video container (mkv/mp4) with ffmpeg."""
import json
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("subtrans")

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
IMAGE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}
SKIP_TITLE_WORDS = ("sign", "song", "forced", "karaoke", "commentary")


def probe_subs(video: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index,codec_name:stream_tags=language,title:stream_disposition=forced,hearing_impaired",
         "-of", "json", str(video)],
        capture_output=True, text=True, timeout=120, check=True)
    return json.loads(out.stdout).get("streams", [])


def pick_english_track(streams, languages=("eng", "en")):
    """Best English text track: not forced, not signs/songs, SubRip preferred, non-SDH preferred."""
    def tags(s):
        return s.get("tags", {}) or {}

    eng = [s for s in streams if (tags(s).get("language") or "").lower() in languages]
    text = [s for s in eng if s.get("codec_name") in TEXT_CODECS]
    if not text:
        if any(s.get("codec_name") in IMAGE_CODECS for s in eng):
            log.info("only image-based English subs (PGS/VobSub): OCR not supported, skipping")
        return None
    usable = [s for s in text
              if not (s.get("disposition", {}) or {}).get("forced")
              and not any(w in (tags(s).get("title") or "").lower() for w in SKIP_TITLE_WORDS)]
    if not usable:
        return None
    usable.sort(key=lambda s: (
        s.get("codec_name") not in ("subrip", "srt"),            # SubRip first (ASS carries signs/karaoke noise)
        bool((s.get("disposition", {}) or {}).get("hearing_impaired")),  # plain over SDH
        s["index"]))
    return usable[0]


def extract_english_srt(video: Path, dest: Path):
    """Write the chosen track to dest as SRT. Returns True on success, False if no usable track."""
    track = pick_english_track(probe_subs(video))
    if track is None:
        return False
    tmp = Path(str(dest) + ".tmp.srt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("extracting embedded subtitle stream #%s (%s) from %s",
             track["index"], track.get("codec_name"), video.name)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
         "-map", f"0:{track['index']}", "-c:s", "srt", str(tmp)],
        check=True, timeout=1800)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        return False
    os.replace(tmp, dest)
    return True
