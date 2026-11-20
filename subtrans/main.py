"""subtrans — translate subtitles (English, else French) into other languages with an LLM.

For every video under LIBRARY_PATHS:
  1. skip it unless it has an audio track in REQUIRE_AUDIO_LANGS (default: Japanese);
  2. for each TARGET_LANGS language, skip it if subtitles already exist, either next to
     the video (X.fr.srt, X.fre.srt, X.fra.hi.srt...) or embedded as a text track;
  3. otherwise translate from the first available SOURCE_LANGS subtitle (default: en, then fr):
     an external file, else the embedded text track (extracted with ffmpeg; image-based
     PGS/VobSub are skipped). Our own machine translations are never used as a source;
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

from . import budget, metrics

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
    max_backoff=env("MAX_BACKOFF_S", 300, int),
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
    # Source subtitle languages to translate from, in order of preference.
    source_langs=[x.strip() for x in env("SOURCE_LANGS", "en,fr").split(",") if x.strip()],
    # Safety net against runaway sources (effect-heavy ASS, broken files): a film has ~1-3k cues.
    max_cues=env("MAX_CUES", 5000, int),
    metrics_port=env("METRICS_PORT", 9108, int),
    # OCR of image-based PGS (Blu-ray) subtitles: Tesseract locally, a vision LLM for OCR_VISION_LANGS.
    ocr=env("OCR", "1") == "1",
    # Gemini media resolution for OCR images: low (~280 tokens/image), medium, high, or "" (API default,
    # ~1120 tokens/image on Gemini 3). Subtitle strips are plain text: low is usually enough.
    ocr_media_resolution=env("OCR_MEDIA_RESOLUTION", "low").lower(),
    # Estimated USD per UTC day; once reached, no new LLM request is sent until tomorrow (0 = no limit).
    daily_budget_usd=env("DAILY_BUDGET_USD", 1.0, float),
    prices=budget.parse_prices(env("PRICES", "")),
    ocr_vision_langs=[x.strip() for x in env("OCR_VISION_LANGS", "ja").split(",") if x.strip()],   # 0 = disabled; only served in scan --watch mode
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
        self.db.execute("CREATE TABLE IF NOT EXISTS status (path TEXT PRIMARY KEY, state TEXT, "
                        "reason TEXT, detail TEXT, updated REAL)")

    def get(self, k):
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, json.dumps(v, ensure_ascii=False)))
        self.db.commit()

    def set_status(self, path, state, reason, detail=""):
        """Last outcome per video (done / skipped / failed), shown by `subtrans report`."""
        row = self.db.execute("SELECT state, reason, detail FROM status WHERE path=?", (str(path),)).fetchone()
        if row == (state, reason, detail):
            return  # unchanged: avoid a write per video on every scan
        self.db.execute("INSERT OR REPLACE INTO status VALUES (?,?,?,?,?)",
                        (str(path), state, reason, detail, time.time()))
        self.db.commit()

    def statuses(self, state=None, reason=None):
        q, args = "SELECT path, state, reason, detail, updated FROM status WHERE 1=1", []
        if state:
            q, args = q + " AND state=?", args + [state]
        if reason:
            q, args = q + " AND reason=?", args + [reason]
        return self.db.execute(q + " ORDER BY state, reason, path", args).fetchall()


# ---------------------------------------------------------------- LLM backends
class RetryableError(Exception):
    pass


class Overloaded(RetryableError):
    """Capacity/quota problem on the provider side (429, 5xx, timeouts): retrying smaller won't help."""

    def __init__(self, msg, retry_after=None, quota=False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.quota = quota


class BadOutput(RetryableError):
    """The model answered but the JSON is broken or ids don't match: a smaller batch may help."""


class ProviderBusy(Exception):
    """Every model is overloaded or out of quota: stop now, resume from the cache later."""


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
    _check_budget()
    _throttle()
    try:
        r = requests.post(url, headers=headers, json=body, timeout=CFG["timeout"])
    except requests.RequestException as e:
        raise Overloaded(f"network: {type(e).__name__}") from e
    if r.status_code in RETRYABLE:
        ra = r.headers.get("Retry-After")
        try:
            reason = (r.json().get("error") or {}).get("message", "")[:160]
        except ValueError:
            reason = ""
        raise Overloaded(f"HTTP {r.status_code}" + (f" ({reason})" if reason else ""),
                         retry_after=float(ra) if ra and ra.replace(".", "", 1).isdigit() else None,
                         quota=r.status_code == 429)
    if r.status_code == 404:
        raise ModelUnavailable(f"HTTP 404: {r.text[:300]}")
    if r.status_code >= 400:
        # never dump URL/headers: the API key must not end up in logs
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


class BudgetExceeded(ProviderBusy):
    """Daily budget reached: stop like a busy provider, resume tomorrow."""


def _check_budget():
    limit = CFG["daily_budget_usd"]
    if limit and budget.spent_today(CFG["cache"]) >= limit:
        raise BudgetExceeded(f"daily budget reached ({budget.spent_today(CFG['cache']):.2f} / {limit:.2f} USD, "
                             "resets at 00:00 UTC)")


def _tokens(model, inp, out, thoughts=0):
    inp, out = int(inp or 0), int(out or 0) + int(thoughts or 0)  # thinking is billed as output
    if inp:
        metrics.inc("jimaku_llm_tokens_total", inp, model=model, kind="input")
    if out:
        metrics.inc("jimaku_llm_tokens_total", out, model=model, kind="output")
    if inp or out:
        usd = budget.record(CFG["cache"], model, inp, out, CFG["prices"])
        metrics.inc("jimaku_llm_cost_usd_total", usd, model=model)


def call_llm(model, system, user):
    b = CFG["backend"]
    if b == "gemini":
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": CFG["api_key"], "Content-Type": "application/json"},
            {"systemInstruction": {"parts": [{"text": system}]},
             "contents": [{"role": "user", "parts": [{"text": user}]}],
             "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}})
        u = data.get("usageMetadata") or {}
        _tokens(model, u.get("promptTokenCount"), u.get("candidatesTokenCount"), u.get("thoughtsTokenCount"))
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
        u = data.get("usage") or {}
        _tokens(model, u.get("prompt_tokens"), u.get("completion_tokens"))
        return data["choices"][0]["message"]["content"]
    if b == "anthropic":
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": CFG["api_key"], "anthropic-version": "2023-06-01"},
                     {"model": model, "max_tokens": 8192, "temperature": 0.3, "system": system,
                      "messages": [{"role": "user", "content": user}]})
        u = data.get("usage") or {}
        _tokens(model, u.get("input_tokens"), u.get("output_tokens"))
        return "".join(p.get("text", "") for p in data["content"])
    if b == "dummy":  # offline tests
        items = json.loads(user.split("TRANSLATE:\n", 1)[1])
        return json.dumps({k: f"[{model}] {v}" for k, v in items.items()}, ensure_ascii=False)
    raise ValueError(f"unknown BACKEND: {b}")


def vision_config():
    cfg = {"temperature": 0.0, "responseMimeType": "application/json"}
    res = CFG["ocr_media_resolution"]
    if res in ("low", "medium", "high"):
        cfg["mediaResolution"] = f"MEDIA_RESOLUTION_{res.upper()}"
    return cfg


def call_vision(model, prompt, items):
    """items: [(id, png_bytes)]. Returns the raw model text (JSON expected)."""
    from .ocr import b64
    b = CFG["backend"]
    if b == "gemini":
        parts = [{"text": prompt}]
        for k, png in items:
            parts += [{"text": f"id {k}:"}, {"inline_data": {"mime_type": "image/png", "data": b64(png)}}]
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": CFG["api_key"], "Content-Type": "application/json"},
            {"contents": [{"role": "user", "parts": parts}],
             "generationConfig": vision_config()})
        u = data.get("usageMetadata") or {}
        _tokens(model, u.get("promptTokenCount"), u.get("candidatesTokenCount"), u.get("thoughtsTokenCount"))
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise RetryableError(f"empty/blocked response: {str(data)[:200]}")
    if b == "anthropic":
        content = [{"type": "text", "text": prompt}]
        for k, png in items:
            content += [{"type": "text", "text": f"id {k}:"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png)}}]
        data = _post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": CFG["api_key"], "anthropic-version": "2023-06-01"},
                     {"model": model, "max_tokens": 8192, "temperature": 0.0,
                      "messages": [{"role": "user", "content": content}]})
        u = data.get("usage") or {}
        _tokens(model, u.get("input_tokens"), u.get("output_tokens"))
        return "".join(p.get("text", "") for p in data["content"])
    if b == "openai":
        content = [{"type": "text", "text": prompt}]
        for k, png in items:
            content += [{"type": "text", "text": f"id {k}:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(png)}"}}]
        data = _post(f"{CFG['openai_base'].rstrip('/')}/chat/completions",
                     {"Authorization": f"Bearer {CFG['api_key'] or 'none'}"},
                     {"model": model, "temperature": 0.0, "response_format": {"type": "json_object"},
                      "messages": [{"role": "user", "content": content}]})
        u = data.get("usage") or {}
        _tokens(model, u.get("prompt_tokens"), u.get("completion_tokens"))
        return data["choices"][0]["message"]["content"]
    if b == "dummy":
        return json.dumps({k: f"[vision {model}] {k}" for k, _ in items})
    raise ValueError(f"unknown BACKEND: {b}")


def vision_batch(lang, items, cache):
    """OCR a batch of subtitle images with a vision model; same retry/fallback rules as translation."""
    from .ocr import images_key, vision_prompt
    key = "ocr:" + images_key(items) + ":" + lang
    hit = cache.get(key)
    if hit:
        metrics.inc("jimaku_cache_hits_total")
        return hit
    ids = {k for k, _ in items}
    last_err = None
    for model in CFG["models"]:
        if model in _dead_models:
            continue
        for attempt in range(CFG["max_retries"]):
            try:
                try:
                    out = parse_json_obj(call_vision(model, vision_prompt(lang), items))
                except json.JSONDecodeError as e:
                    raise BadOutput(f"invalid JSON: {e}") from e
                if not isinstance(out, dict) or not set(out) <= ids:
                    raise BadOutput("unexpected ids")
                out = {k: str(out.get(k, "")).replace("\\n", "\n").strip() for k in ids}
                cache.put(key, out)
                metrics.inc("jimaku_llm_requests_total", model=model, outcome="ok")
                return out
            except ModelUnavailable as e:
                _dead_models.add(model)
                last_err = e
                break
            except RetryableError as e:
                last_err = e
                outcome = ("quota" if getattr(e, "quota", False) else "overloaded") \
                    if isinstance(e, Overloaded) else "bad_output"
                metrics.inc("jimaku_llm_requests_total", model=model, outcome=outcome)
                delay = min(CFG["max_backoff"], (e.retry_after if isinstance(e, Overloaded) and e.retry_after
                                                 else 15 * 2 ** attempt if getattr(e, "quota", False)
                                                 else 2 ** attempt * 2)) + random.uniform(0, 1.5)
                log.warning("%s OCR batch: %s (attempt %d/%d, waiting %.0fs)", model, e, attempt + 1,
                            CFG["max_retries"], delay)
                time.sleep(delay)
    if isinstance(last_err, Overloaded):
        raise ProviderBusy(f"all models overloaded or out of quota ({last_err})")
    raise RuntimeError(f"OCR batch failed: {last_err}")


def parse_json_obj(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


# ---------------------------------------------------------------- translation
def system_prompt(lang, glossary, src_lang="en"):
    name = LANG_NAMES.get(lang, lang)
    src_name = LANG_NAMES.get(src_lang, "English" if src_lang == "en" else src_lang)
    p = (f"You are a professional subtitle translator. Translate {src_name} subtitles into natural, "
         f"spoken {name} suitable for on-screen subtitles (concise, readable).\n"
         "Rules:\n"
         "- Input is a JSON object {id: text}. Output ONLY a JSON object with exactly the same ids.\n"
         "- Keep line breaks (\\n) where the source has them; keep leading '- ' dialogue dashes.\n"
         "- Keep tags like <i>...</i> unchanged.\n"
         "- Use the CONTEXT only for continuity; do not translate or output it.\n"
         "- Keep character names and invented terms consistent across the whole film.\n"
         "- Fix obvious errors in the source rather than translating them literally.\n")
    if lang == "ja":
        p += "- Use natural casual Japanese for children, polite forms for staff/servants/adults addressing guests. Use full-width punctuation.\n"
    if lang == "fr":
        p += "- Children tutoient each other; staff and strangers vouvoient.\n"
    if glossary:
        p += f"\nGlossary (source term => preferred term):\n{glossary}\n"
    return p


def translate_batch(batch, ctx, lang, glossary, cache, src_lang="en"):
    src = {str(i): t for i, _, t in batch}
    key_parts = [PROMPT_VERSION, lang, glossary, ctx, src] + ([src_lang] if src_lang != "en" else [])
    key = hashlib.sha256(json.dumps(key_parts, ensure_ascii=False).encode()).hexdigest()
    hit = cache.get(key)
    if hit:
        metrics.inc("jimaku_cache_hits_total")
        return hit
    sys_p = system_prompt(lang, glossary, src_lang)
    user = ("CONTEXT (previous lines, source => translation):\n"
            + "\n".join(f"{s} => {t}" for s, t in ctx)
            + "\n\nTRANSLATE:\n" + json.dumps(src, ensure_ascii=False))
    last_err = None
    for model in CFG["models"]:
        if model in _dead_models:
            continue
        for attempt in range(CFG["max_retries"]):
            try:
                t0 = time.time()
                try:
                    raw = call_llm(model, sys_p, user)
                finally:
                    metrics.inc("jimaku_llm_request_seconds_total", time.time() - t0, model=model)
                try:
                    out = parse_json_obj(raw)
                except json.JSONDecodeError as e:
                    raise BadOutput(f"invalid JSON: {e}") from e
                if not isinstance(out, dict) or set(out) != set(src):
                    raise BadOutput(f"id mismatch ({len(out) if isinstance(out, dict) else '?'}/{len(src)})")
                out = {k: str(v).replace("\\n", "\n").strip() or src[k] for k, v in out.items()}
                cache.put(key, out)
                metrics.inc("jimaku_llm_requests_total", model=model, outcome="ok")
                metrics.inc("jimaku_cues_translated_total", len(out), lang=lang)
                return out
            except ModelUnavailable as e:
                metrics.inc("jimaku_llm_requests_total", model=model, outcome="unavailable")
                _dead_models.add(model)
                last_err = e
                log.error("model %s unavailable, disabling it for this run: %s", model, e)
                break
            except RetryableError as e:
                last_err = e
                outcome = ("quota" if getattr(e, "quota", False) else "overloaded") \
                    if isinstance(e, Overloaded) else "bad_output"
                metrics.inc("jimaku_llm_requests_total", model=model, outcome=outcome)
                if isinstance(e, Overloaded) and e.retry_after:
                    delay = min(CFG["max_backoff"], e.retry_after)
                elif isinstance(e, Overloaded) and e.quota:
                    delay = min(CFG["max_backoff"], 15 * 2 ** attempt)  # quota windows are ~1 min
                else:
                    delay = min(CFG["max_backoff"], 2 ** attempt * 2)
                delay += random.uniform(0, 1.5)
                log.warning("%s batch %s-%s: %s (attempt %d/%d, waiting %.0fs)", model, batch[0][0],
                            batch[-1][0], e, attempt + 1, CFG["max_retries"], delay)
                time.sleep(delay)
        if model not in _dead_models:
            log.warning("giving up on %s for batch %s-%s", model, batch[0][0], batch[-1][0])
    if all(m in _dead_models for m in CFG["models"]):
        raise RuntimeError(f"no usable model in MODELS={CFG['models']}: {last_err}")
    if isinstance(last_err, Overloaded):
        # Capacity/quota problem: splitting the batch would only multiply requests. Stop and resume later.
        raise ProviderBusy(f"all models overloaded or out of quota ({last_err})")
    # Broken output: split the batch in half (large batches break the JSON more often)
    if len(batch) > 8:
        mid = len(batch) // 2
        a = translate_batch(batch[:mid], ctx, lang, glossary, cache, src_lang)
        tail_ctx = (ctx + [(batch[j][2], a[str(batch[j][0])]) for j in range(mid)])[-CFG["context"]:]
        return {**a, **translate_batch(batch[mid:], tail_ctx, lang, glossary, cache, src_lang)}
    raise RuntimeError(f"batch {batch[0][0]}-{batch[-1][0]} failed to translate: {last_err}")


def load_glossary(srt_path):
    parts = []
    for g in (CFG["glossary"], str(Path(srt_path).parent / "glossary.txt")):
        if g and Path(g).is_file():
            parts.append(Path(g).read_text(encoding="utf-8").strip())
    return "\n".join(parts)


def translate_file(src_path, lang, cache, src_lang="en"):
    cues = parse_srt(src_path)
    glossary = load_glossary(src_path)
    out, ctx = {}, []
    for i in range(0, len(cues), CFG["batch"]):
        batch = cues[i:i + CFG["batch"]]
        res = translate_batch(batch, ctx[-CFG["context"]:], lang, glossary, cache, src_lang)
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
    # "v2": extraction with ASS cleanup; older (uncleaned) extractions are ignored.
    return Path(CFG["cache"]).parent / "extracted" / f"{h}.v2.{lang}.srt"


def _sig(f: Path):
    st = f.stat()
    return [st.st_size, int(st.st_mtime)]


def mark_ours(f: Path, cache: Cache):
    """Remember that we produced this file (machine translation)."""
    cache.put(f"ours:{f}", _sig(f))


def is_ours(f: Path, cache: Cache):
    """True if the file is still the one we wrote (Bazarr replacing it changes size/mtime)."""
    return cache.get(f"ours:{f}") == _sig(f)


def find_srt(lang, base: Path, video: Path | None, streams, ext, cache: Cache, exclude_ours=False):
    """An SRT in `lang`: external file first, else the embedded text track (extracted with ffmpeg).

    exclude_ours: skip files we generated, so a translation is never used as a translation source.
    """
    for f in ext.get(lang, []):
        if f.suffix.lower() == ".srt" and not (exclude_ours and is_ours(f, cache)):
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
    if CFG["ocr"]:
        ocr = ocr_source(lang, base, video, streams, cache)
        if ocr:
            return ocr
    cache.put(neg_key, True)
    return None


def ocr_source(lang, base: Path, video: Path, streams, cache: Cache):
    """OCR the best PGS track in `lang` into an SRT in the cache dir."""
    from .embedded import aliases
    from .ocr import ocr_track
    codes = aliases(lang)
    pgs = [s for s in streams if s.get("codec_type") == "subtitle" and s.get("codec_name") == "hdmv_pgs_subtitle"
           and ((s.get("tags") or {}).get("language") or "").lower() in codes
           and not (s.get("disposition") or {}).get("forced")
           and "sign" not in ((s.get("tags") or {}).get("title") or "").lower()]
    if not pgs:
        return None
    pgs.sort(key=lambda s: (bool((s.get("disposition") or {}).get("hearing_impaired")), s["index"]))
    vision = lang in CFG["ocr_vision_langs"] and CFG["backend"] in ("gemini", "anthropic", "openai", "dummy")
    engine = "vision" if vision else "tesseract"
    dest = Path(str(extracted_path(video, base, lang))[:-4] + ".ocr.srt")
    if dest.exists() and dest.stat().st_mtime >= video.stat().st_mtime:
        return dest
    log.info("OCR of embedded %s PGS stream #%s from %s (%s)", lang, pgs[0]["index"], video.name, engine)
    from .ocr import TooManyImages
    st = video.stat()
    big_key = f"ocr_too_many:{lang}:{video}:{st.st_size}:{int(st.st_mtime)}"
    if cache.get(big_key):
        raise TooManyImages(cache.get(big_key))  # known: don't re-read the file every scan
    try:
        srt = ocr_track(video, pgs[0]["index"], lang, engine,
                        call_vision=lambda lg, items: vision_batch(lg, items, cache),
                        max_images=CFG["max_cues"])
    except TooManyImages as e:
        cache.put(big_key, str(e))
        raise
    n = srt.count(" --> ")
    metrics.inc("jimaku_ocr_cues_total", n, lang=lang, engine=engine)
    if not n:
        log.info("OCR found no text in %s", video.name)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(dest, srt)
    log.info("OCR %s: %d cues", lang, n)
    return dest


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
    from .embedded import audio_langs, available_sub_langs, normalize
    base, video = resolve(path)
    if base is None:
        log.info("skipped (not a video nor a subtitle file): %s", path.name)
        return False
    if not pending_outputs(base, force):
        return False  # everything we would produce already exists

    key = video or base
    streams = probe_cached(video, cache) if video else []
    if video and streams is None:
        cache.set_status(key, "skipped", "probe_failed", "ffprobe could not read the file")
        return False
    wanted_audio = {normalize(x) for x in CFG["require_audio"]}
    if video and wanted_audio:
        found = audio_langs(streams)
        if not found & wanted_audio:
            found_s = ", ".join(sorted(found)) or "untagged"
            log.debug("skipped %s: no %s audio (found: %s)", video.name, "/".join(sorted(wanted_audio)), found_s)
            cache.set_status(key, "skipped", "audio_language", f"audio: {found_s}")
            return False

    ext = external_subs(base)
    embedded = available_sub_langs(streams)  # text or image tracks: already watchable
    did = False
    written, problem = [], None

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
        src_lang, src = None, None
        for cand in CFG["source_langs"]:
            if cand in todo:
                continue  # can't be both missing and the source
            src = find_srt(cand, base, video, streams, ext, cache, exclude_ours=True)
            if src:
                src_lang = cand
                break
        n_cues = len(parse_srt(src)) if src is not None else 0
        if src is None:
            from .embedded import IMAGE_CODECS, aliases
            src_codes = set().union(*(aliases(lang) for lang in CFG["source_langs"]))
            image_only = any(s.get("codec_type") == "subtitle" and s.get("codec_name") in IMAGE_CODECS
                             and ((s.get("tags") or {}).get("language") or "").lower() in src_codes
                             for s in streams or [])
            log.info("no usable %s subtitle for %s", "/".join(CFG["source_langs"]), base.name)
            problem = ("image_subs_only", "image-based subtitles only (VobSub, or OCR disabled/empty)") if image_only else \
                ("no_source", f"no {'/'.join(CFG['source_langs'])} text subtitle")
        elif n_cues > CFG["max_cues"]:
            log.error("skipping %s: source %s has %d cues (> MAX_CUES=%d), probably effects/karaoke. "
                      "Check the file or raise MAX_CUES.", base.name, src.name, n_cues, CFG["max_cues"])
            problem = ("too_many_cues", f"{src.name}: {n_cues} cues > MAX_CUES={CFG['max_cues']}")
        else:
            for lang in todo:
                if lang == src_lang:
                    continue
                dst = Path(f"{base}.{lang}.srt")
                log.info("translating %s (%s) -> %s", base.name, src_lang, lang)
                t0 = time.time()
                cues = translate_file(src, lang, cache, src_lang)
                write_atomic(dst, render_srt(cues))
                mark_ours(dst, cache)
                log.info("wrote %s (%.0fs)", dst.name, time.time() - t0)
                written.append(f"{lang} (from {src_lang})")
                metrics.inc("jimaku_files_written_total", kind=f"{lang}.srt")
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
                written.append("ja.ass")
                metrics.inc("jimaku_files_written_total", kind="ja.ass")
                did = True

    if problem:
        cache.set_status(key, "skipped", *problem)
    elif written:
        cache.set_status(key, "done", "translated", ", ".join(written))
    else:
        cache.set_status(key, "done", "already_available", "subtitles already present")
    return did


def scan(cache, force=False):
    now = time.time()
    t_start = now
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
            except ProviderBusy as e:  # hitting the next files would fail the same way
                cache.set_status(p, "failed", "provider_busy", str(e)[:300])
                log.error("stopping scan: %s; progress is cached, next scan resumes", e)
                return n_err + 1
            except Exception as e:  # one failing file must not block the rest
                n_err += 1
                cache.set_status(p, "failed", "error", f"{type(e).__name__}: {e}"[:300])
                log.error("failed %s: %s", p, e)
    summary = cache.db.execute("SELECT state || '/' || reason, COUNT(*) FROM status "
                               "WHERE state != 'done' GROUP BY 1 ORDER BY 2 DESC").fetchall()
    metrics.set_gauge("jimaku_scan_duration_seconds", time.time() - t_start)
    metrics.set_gauge("jimaku_scan_last_success_timestamp_seconds", time.time())
    log.info("scan done: %d processed, %d failed%s", n_ok, n_err,
             ("; not translated: " + ", ".join(f"{k}={v}" for k, v in summary)
              + " (details: subtrans report)") if summary else "")
    return n_err


def report(cache, a):
    rows = cache.statuses(a.state, a.reason)
    if not (a.all or a.state or a.reason):
        rows = [r for r in rows if r[1] != "done" and r[2] != "audio_language"]
    if a.json:
        print(json.dumps([dict(zip(("path", "state", "reason", "detail", "updated"), r)) for r in rows],
                         ensure_ascii=False, indent=2))
        return 0
    counts = {}
    for _, state, reason, _, _ in cache.statuses():
        counts[(state, reason)] = counts.get((state, reason), 0) + 1
    print("summary: " + ", ".join(f"{s}/{r}={n}" for (s, r), n in sorted(counts.items())))
    for path, state, reason, detail, updated in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(updated))
        print(f"{when}  {state:<7} {reason:<17} {Path(path).name}\n{'':42}{detail}")
    return 0


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
    r = sub.add_parser("report", help="show the last outcome of every video (skipped/failed/done)")
    r.add_argument("--state", choices=["done", "skipped", "failed"])
    r.add_argument("--reason", help="e.g. too_many_cues, no_source, image_subs_only, audio_language")
    r.add_argument("--all", action="store_true", help="include done and audio_language rows")
    r.add_argument("--json", action="store_true")
    sub.add_parser("usage", help="estimated LLM spend per day and model (last 7 days)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "report":
        return report(Cache(CFG["cache"]), a)
    if a.cmd == "usage":
        Cache(CFG["cache"])  # make sure the DB exists
        print(f"today ({budget.today()} UTC): {budget.spent_today(CFG['cache']):.3f} USD"
              f" / budget {CFG['daily_budget_usd'] or 'unlimited'}")
        for day, model, inp, out, usd in budget.summary(CFG["cache"]):
            print(f"{day}  {model:<28} in={inp:>10,} out={out:>9,}  {usd:7.3f} USD")
        return 0
    if CFG["backend"] not in ("dummy", "openai") and not CFG["api_key"]:
        sys.exit("API_KEY is missing")
    cache = Cache(CFG["cache"])
    if a.cmd == "file":
        try:
            handle(Path(a.path), cache, a.force)
        except ProviderBusy as e:
            log.error("%s. Progress is cached: rerun the same command later to resume.", e)
            return 75  # EX_TEMPFAIL
        return 0
    if a.watch and CFG["metrics_port"]:
        metrics.serve(CFG["metrics_port"], CFG["cache"])
        log.info("metrics on :%d/metrics", CFG["metrics_port"])
    while True:
        errs = scan(cache, a.force)
        if not a.watch:
            return 1 if errs else 0
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
