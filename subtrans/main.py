"""subtrans — translate English subtitles into other languages with an LLM.

For every video under LIBRARY_PATHS:
  1. skip it unless it has an audio track in REQUIRE_AUDIO_LANGS (default: Japanese);
  2. for each TARGET_LANGS language, skip it if subtitles already exist, either next to
     the video (X.fr.srt, X.fre.srt, X.fra.hi.srt...) or embedded as a text track;
  3. otherwise translate the English source: an external X.en.srt, else the embedded
     English text track (extracted with ffmpeg; image-based PGS/VobSub are skipped);
  4. build X.ja.ass with furigana, preferring real Japanese subtitles over our translation.

Modes:
  subtrans file  <video|subtitle>       # single item (e.g. Bazarr/Radarr post-processing)
  subtrans scan  [--watch SECONDS]      # walk LIBRARY_PATHS

Idempotent: never overwrites an existing output (unless --force, which only rewrites our own files).
Resumable: every translated batch is cached (SQLite); a crash only loses the current batch.
"""
import argparse
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger("subtrans")

PROMPT_VERSION = "v1"
LANG_NAMES = {"fr": "French", "ja": "Japanese", "es": "Spanish", "de": "German", "it": "Italian"}
RETRYABLE = {408, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------- config
def env(name, default=None, cast=str):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


CFG = dict(
    backend=env("BACKEND", "gemini"),                 # gemini | openai | anthropic | dummy
    models=[m.strip() for m in env("MODELS", "gemini-3.8-flash").split(",") if m.strip()],
    api_key=env("API_KEY", ""),
    openai_base=env("OPENAI_BASE_URL", "http://ollama:11434/v1"),
    targets=[t.strip() for t in env("TARGET_LANGS", "fr,ja").split(",") if t.strip()],
    furigana=env("FURIGANA", "1") == "1",
    batch=env("BATCH_SIZE", 60, int),
    context=env("CONTEXT_CUES", 8, int),
    max_retries=env("MAX_RETRIES", 6, int),
    min_interval=env("MIN_INTERVAL_S", 1.0, float),   # simple rate limit between requests
    cache=env("CACHE_DB", "/cache/subtrans.db"),
    libs=[p for p in env("LIBRARY_PATHS", "/media").split(":") if p],
    settle=env("SETTLE_S", 120, int),                 # skip files modified too recently
    glossary=env("GLOSSARY_FILE", ""),
    timeout=env("HTTP_TIMEOUT_S", 180, int),
    extract_embedded=env("EXTRACT_EMBEDDED", "1") == "1",
    keep_extracted=env("KEEP_EXTRACTED", "0") == "1",   # 1 = also write <video>.en.srt next to the video
    # Only process videos having an audio track in one of these languages ("" = no filter).
    require_audio=[x.strip() for x in env("REQUIRE_AUDIO_LANGS", "ja").split(",") if x.strip()],
    video_exts=[e.strip().lower() for e in env("VIDEO_EXTS", ".mkv,.mp4,.m4v,.avi").split(",")],
)


# ---------------------------------------------------------------- SRT
def parse_srt(path):
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").strip()
    cues = []
    for block in re.split(r"\n\s*\n", raw):
        lines = block.split("\n")
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        cues.append((int(lines[0].strip()), lines[1].strip(), "\n".join(lines[2:]).strip()))
    if not cues:
        raise ValueError(f"no readable cue in {path}")
    return cues


def write_atomic(path, content, encoding="utf-8"):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def render_srt(cues):
    return "".join(f"{i}\n{tc}\n{t}\n\n" for i, tc, t in cues)


# ---------------------------------------------------------------- cache
class Cache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k):
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v, ensure_ascii=False)))
        self.db.commit()


# ---------------------------------------------------------------- LLM backends
class RetryableError(Exception):
    pass


class ModelUnavailable(Exception):
    """The model does not exist or is not offered to this account (HTTP 404): skip it for good."""


_dead_models = set()


_last_call = [0.0]


def _throttle():
    wait = CFG["min_interval"] - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def _post(url, headers, body):
    _throttle()
    try:
        r = requests.post(url, headers=headers, json=body, timeout=CFG["timeout"])
    except requests.RequestException as e:
        raise RetryableError(f"network: {type(e).__name__}") from e
    if r.status_code in RETRYABLE:
        ra = r.headers.get("Retry-After")
        raise RetryableError(f"HTTP {r.status_code}" + (f" retry-after={ra}" if ra else ""))
    if r.status_code == 404:
        raise ModelUnavailable(f"HTTP 404: {r.text[:300]}")
    if r.status_code >= 400:
        # never dump URL/headers: the API key must not end up in logs
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def call_llm(model, system, user):
    b = CFG["backend"]
    if b == "gemini":
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": CFG["api_key"], "Content-Type": "application/json"},
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": [{"role": "user", "parts": [{"text": user}]}],
             "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}})
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RetryableError(f"empty/blocked response: {str(data)[:200]}")
    if b == "openai":  # OpenAI, Ollama, vLLM, LiteLLM, OpenRouter...
        data = _post(f"{CFG['openai_base'].rstrip('/')}/chat/completions",
                     {"Authorization": f"Bearer {CFG['api_key'] or 'none'}"},
                     {"model": model, "temperature": 0.3, "response_format": {"type": "json_object"},
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}]})
        return data["choices"][0]["message"]["content"]
    if b == "anthropic":
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": CFG["api_key"], "anthropic-version": "2023-06-01"},
                     {"model": model, "max_tokens": 8192, "temperature": 0.3, "system": system,
                      "messages": [{"role": "user", "content": user}]})
        return "".join(p.get("text", "") for p in data["content"])
    if b == "dummy":  # offline tests
        items = json.loads(user.split("TRANSLATE:\n", 1)[1])
        return json.dumps({k: f"[{model}] {v}" for k, v in items.items()}, ensure_ascii=False)
    raise ValueError(f"unknown BACKEND: {b}")


def parse_json_obj(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


# ---------------------------------------------------------------- translation
def system_prompt(lang, glossary):
    name = LANG_NAMES.get(lang, lang)
    p = (f"You are a professional subtitle translator. Translate English subtitles into natural, "
         f"spoken {name} suitable for on-screen subtitles (concise, readable).\n"
         "Rules:\n"
         "- Input is a JSON object {id: text}. Output ONLY a JSON object with exactly the same ids.\n"
         "- Keep line breaks (\\n) where the source has them; keep leading '- ' dialogue dashes.\n"
         "- Keep tags like <i>...</i> unchanged.\n"
         "- Use the CONTEXT only for continuity; do not translate or output it.\n"
         "- Keep character names and invented terms consistent across the whole film.\n"
         "- Fix obvious errors in the English source rather than translating them literally.\n")
    if lang == "ja":
        p += "- Use natural casual Japanese for children, polite forms for staff/servants/adults addressing guests. Use full-width punctuation.\n"
    if lang == "fr":
        p += "- Children tutoient each other; staff and strangers vouvoient.\n"
    if glossary:
        p += f"\nGlossary (English => preferred term):\n{glossary}\n"
    return p


def translate_batch(batch, ctx, lang, glossary, cache):
    src = {str(i): t for i, _, t in batch}
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, lang, glossary, ctx, src],
                                    ensure_ascii=False).encode()).hexdigest()
    hit = cache.get(key)
    if hit:
        return hit
    sys_p = system_prompt(lang, glossary)
    user = ("CONTEXT (previous lines, source => translation):\n"
            + "\n".join(f"{s} => {t}" for s, t in ctx)
            + "\n\nTRANSLATE:\n" + json.dumps(src, ensure_ascii=False))
    last_err = None
    for model in CFG["models"]:
        if model in _dead_models:
            continue
        for attempt in range(CFG["max_retries"]):
            try:
                out = parse_json_obj(call_llm(model, sys_p, user))
                if set(out) != set(src):
                    raise RetryableError(f"id mismatch ({len(out)}/{len(src)})")
                out = {k: str(v).replace("\\n", "\n").strip() or src[k] for k, v in out.items()}
                cache.put(key, out)
                return out
            except ModelUnavailable as e:
                _dead_models.add(model)
                last_err = e
                log.error("model %s unavailable, disabling it for this run: %s", model, e)
                break
            except (RetryableError, json.JSONDecodeError) as e:
                last_err = e
                delay = min(120, 2 ** attempt * 2) + random.uniform(0, 1.5)
                log.warning("%s batch %s-%s: %s (attempt %d, waiting %.0fs)", model, batch[0][0],
                            batch[-1][0], e, attempt + 1, delay)
                time.sleep(delay)
        if model not in _dead_models:
            log.warning("falling back to next model after failures on %s", model)
    if all(m in _dead_models for m in CFG["models"]):
        raise RuntimeError(f"no usable model in MODELS={CFG['models']}: {last_err}")
    # last resort: split the batch in half (large batches break the JSON more often)
    if len(batch) > 8:
        mid = len(batch) // 2
        a = translate_batch(batch[:mid], ctx, lang, glossary, cache)
        tail_ctx = (ctx + [(batch[j][2], a[str(batch[j][0])]) for j in range(mid)])[-CFG["context"]:]
        return {**a, **translate_batch(batch[mid:], tail_ctx, lang, glossary, cache)}
    raise RuntimeError(f"batch {batch[0][0]}-{batch[-1][0]} failed to translate: {last_err}")


def load_glossary(srt_path):
    parts = []
    for g in (CFG["glossary"], str(Path(srt_path).parent / "glossary.txt")):
        if g and Path(g).is_file():
            parts.append(Path(g).read_text(encoding="utf-8").strip())
    return "\n".join(parts)


def translate_file(src_path, lang, cache):
    cues = parse_srt(src_path)
    glossary = load_glossary(src_path)
    out, ctx = {}, []
    for i in range(0, len(cues), CFG["batch"]):
        batch = cues[i:i + CFG["batch"]]
        res = translate_batch(batch, ctx[-CFG["context"]:], lang, glossary, cache)
        out.update(res)
        ctx += [(t, res[str(n)]) for n, _, t in batch]
        log.info("  %s: %d/%d", lang, min(i + CFG["batch"], len(cues)), len(cues))
    return [(n, tc, out[str(n)]) for n, tc, _ in cues]


# ---------------------------------------------------------------- orchestration
SUB_EXTS = {".srt", ".ass", ".ssa", ".vtt"}


def pending_outputs(base: Path, force=False):
    """Our own output files that do not exist yet (cheap check, no ffprobe)."""
    out = [Path(f"{base}.{lang}.srt") for lang in CFG["targets"]]
    if "ja" in CFG["targets"] and CFG["furigana"]:
        out.append(Path(f"{base}.ja.ass"))
    return out if force else [o for o in out if not o.exists()]


def external_subs(base: Path):
    """{lang: [paths]} of full (non-forced) subtitle files next to the video.

    Understands Bazarr/Radarr naming: X.fr.srt, X.fre.srt, X.fra.hi.srt, X.ja.ass, ...
    """
    from .embedded import LANG_ALIASES, normalize
    found = {}
    prefix = base.name + "."
    try:
        entries = list(base.parent.iterdir())
    except FileNotFoundError:
        return found
    for f in entries:
        if not f.name.startswith(prefix) or f.suffix.lower() not in SUB_EXTS:
            continue
        tokens = f.name[len(prefix):-len(f.suffix)].lower().split(".")
        if "forced" in tokens:
            continue  # forced tracks only cover foreign-language parts
        lang = next((normalize(t) for t in tokens if normalize(t) in LANG_ALIASES), None)
        if lang:
            found.setdefault(lang, []).append(f)
    for paths in found.values():  # exact "X.<lang>.srt" first, SDH/HI variants last
        paths.sort(key=lambda f: (f.suffix.lower() != ".srt", len(f.name)))
    return found


def probe_cached(video: Path, cache: Cache):
    """ffprobe streams, cached by path/size/mtime so rescans don't touch the NAS."""
    from .embedded import probe
    st = video.stat()
    key = f"probe:{video}:{st.st_size}:{int(st.st_mtime)}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        streams = probe(video)
    except Exception as e:  # corrupt/partial file: skip it, retry when it changes
        log.warning("ffprobe failed on %s: %s", video.name, e)
        streams = None
    cache.put(key, streams)
    return streams


def extracted_path(video: Path, base: Path, lang):
    if CFG["keep_extracted"] and lang == "en":
        return Path(f"{base}.en.srt")
    h = hashlib.sha1(str(video).encode()).hexdigest()[:16]
    return Path(CFG["cache"]).parent / "extracted" / f"{h}.{lang}.srt"


def find_srt(lang, base: Path, video: Path | None, streams, ext, cache: Cache):
    """An SRT in `lang`: external file first, else the embedded text track (extracted with ffmpeg)."""
    for f in ext.get(lang, []):
        if f.suffix.lower() == ".srt":
            return f
    if not (video and streams and CFG["extract_embedded"]):
        return None
    st = video.stat()
    neg_key = f"noembed:{lang}:{video}:{st.st_size}:{int(st.st_mtime)}"
    if cache.get(neg_key):
        return None  # already probed, nothing usable: don't re-read a multi-GB file every scan
    dest = extracted_path(video, base, lang)
    if dest.exists() and dest.stat().st_mtime >= st.st_mtime:
        return dest
    from .embedded import extract_srt
    if extract_srt(video, dest, lang, streams):
        return dest
    cache.put(neg_key, True)
    return None


def resolve(path: Path):
    """(base, video) for a video file or a subtitle file such as X.en.srt / X.eng.hi.srt."""
    from .embedded import LANG_ALIASES, normalize
    if path.suffix.lower() in CFG["video_exts"]:
        return path.with_suffix(""), path
    if path.suffix.lower() not in SUB_EXTS:
        return None, None
    parts = path.name.split(".")[:-1]
    idx = next((i for i in range(len(parts) - 1, 0, -1) if normalize(parts[i]) in LANG_ALIASES), None)
    if idx is None:
        return None, None
    base = path.with_name(".".join(parts[:idx]))
    video = next((Path(f"{base}{e}") for e in CFG["video_exts"] if Path(f"{base}{e}").exists()), None)
    return base, video


def handle(path: Path, cache, force=False):
    """Decide what a video needs and do it. Returns True if something was written."""
    from .embedded import audio_langs, full_text_sub_langs, normalize
    base, video = resolve(path)
    if base is None:
        log.info("skipped (not a video nor a subtitle file): %s", path.name)
        return False
    if not pending_outputs(base, force):
        return False  # everything we would produce already exists

    streams = probe_cached(video, cache) if video else []
    if video and streams is None:
        return False
    wanted_audio = {normalize(x) for x in CFG["require_audio"]}
    if video and wanted_audio:
        found = audio_langs(streams)
        if not found & wanted_audio:
            log.debug("skipped %s: no %s audio (found: %s)", video.name,
                      "/".join(sorted(wanted_audio)), ", ".join(sorted(found)) or "untagged")
            return False

    ext = external_subs(base)
    embedded = full_text_sub_langs(streams)
    did = False

    todo = []
    for lang in CFG["targets"]:
        own = Path(f"{base}.{lang}.srt")
        others = [f for f in ext.get(lang, []) if f != own]
        if lang in embedded or others:
            log.debug("%s: %s subtitles already available, not translating", base.name, lang)
            continue
        if force or not own.exists():
            todo.append(lang)

    if todo:
        src = find_srt("en", base, video, streams, ext, cache)
        if src is None:
            log.info("no usable English subtitle for %s", base.name)
        else:
            for lang in todo:
                dst = Path(f"{base}.{lang}.srt")
                log.info("translating %s -> %s", base.name, lang)
                t0 = time.time()
                cues = translate_file(src, lang, cache)
                write_atomic(dst, render_srt(cues))
                log.info("wrote %s (%.0fs)", dst.name, time.time() - t0)
                did = True

    if "ja" in CFG["targets"] and CFG["furigana"]:
        ass = Path(f"{base}.ja.ass")
        if force or not ass.exists():
            # Prefer real Japanese subtitles (embedded or external) over our own translation.
            ext = external_subs(base)
            ja_src = find_srt("ja", base, video, streams, ext, cache)
            if ja_src:
                from .furigana import cues_to_ass
                write_atomic(ass, cues_to_ass(parse_srt(ja_src), title=base.name), encoding="utf-8-sig")
                log.info("wrote %s (from %s)", ass.name,
                         "embedded track" if ja_src.parent != base.parent else ja_src.name)
                did = True
    return did


def scan(cache, force=False):
    now = time.time()
    n_ok = n_err = 0
    for lib in CFG["libs"]:
        for p in sorted(Path(lib).rglob("*")):
            if p.suffix.lower() not in CFG["video_exts"] or not p.is_file():
                continue
            if now - p.stat().st_mtime < CFG["settle"]:
                continue  # still being copied/remuxed
            try:
                if handle(p, cache, force):
                    n_ok += 1
            except Exception as e:  # one failing file must not block the rest
                n_err += 1
                log.error("failed %s: %s", p, e)
    log.info("scan done: %d processed, %d failed", n_ok, n_err)
    return n_err


def main(argv=None):
    ap = argparse.ArgumentParser(prog="subtrans")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("file", help="process one video file or one subtitle file")
    f.add_argument("path")
    f.add_argument("--force", action="store_true")
    s = sub.add_parser("scan", help="walk LIBRARY_PATHS for video files")
    s.add_argument("--watch", type=int, default=0, help="rescan every N seconds")
    s.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if CFG["backend"] not in ("dummy", "openai") and not CFG["api_key"]:
        sys.exit("API_KEY is missing")
    cache = Cache(CFG["cache"])
    if a.cmd == "file":
        handle(Path(a.path), cache, a.force)
        return 0
    while True:
        errs = scan(cache, a.force)
        if not a.watch:
            return 1 if errs else 0
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
