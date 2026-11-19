# jimaku-llm (字幕)

Automatically translates the English subtitles of a Jellyfin/Plex media library into other languages
(`*.fr.srt`, `*.ja.srt`, …) with an LLM, and generates a Japanese `*.ja.ass` with **furigana** above every kanji
— handy for watching anime/films while learning Japanese.

Works with Gemini, Claude, or any OpenAI-compatible API (OpenAI, Ollama, vLLM, LiteLLM, OpenRouter).

![pipeline](https://img.shields.io/badge/pipeline-.en.srt%20→%20LLM%20→%20.fr.srt%20%2F%20.ja.srt%20%2B%20furigana%20.ass-blue)

## How it works

```
library scan ──► video with Japanese audio, missing <lang> subtitles
                    │
                    ▼
         English source: external *.en.srt, else embedded track (ffmpeg)
                    │
                    ▼
         batches of ~60 cues + rolling context (previous lines)
                    │  JSON in / JSON out, ids validated
                    ▼
         LLM (Gemini / Claude / OpenAI-compatible)  ◄── SQLite cache (resume)
                    │
                    ▼
         *.fr.srt, *.ja.srt  ──► MeCab/UniDic (fugashi) ──► *.ja.ass with furigana
```

- **Translation** is done by the LLM, with a per-film or global **glossary** to keep names consistent.
- **Furigana** are *not* LLM-generated: readings come from MeCab + UniDic (deterministic, free, fast),
  with a small override table for context-dependent words (何 なに/なん, 一日, …). Each reading is placed
  with `\pos` above its kanji, okurigana excluded.

## What gets translated

For every video under `LIBRARY_PATHS`:

1. **Audio filter** — skipped unless it has an audio track in `REQUIRE_AUDIO_LANGS` (default `ja`:
   Japanese films, anime, tokusatsu…). Untagged (`und`) audio does not match: fix the tag with
   `mkvpropedit X.mkv --edit track:a1 --set language=jpn`, or set `REQUIRE_AUDIO_LANGS=` to disable the filter.
2. **Existing subtitles win** — a target language is skipped when subtitles already exist, either
   - next to the video, with Bazarr/Radarr naming: `X.fr.srt`, `X.fre.srt`, `X.fra.hi.srt`, `X.ja.ass`… (`*.forced.*` does not count), or
   - embedded in the container as a full text track (not forced, not "Signs & Songs").
3. **English source** — an external `X.en.srt` / `X.eng.srt`, otherwise the **embedded** English text track,
   extracted with `ffprobe`/`ffmpeg` (SubRip preferred over ASS, plain over SDH). Image-based tracks
   (PGS, VobSub) would need OCR and are skipped. The extracted copy stays in the cache dir
   (`KEEP_EXTRACTED=1` writes `X.en.srt` next to the video instead).
4. **Furigana** — `X.ja.ass` is built from **real Japanese subtitles** when the video has some (embedded or external),
   and only falls back to our own Japanese translation otherwise.

`ffprobe` results are cached by path/size/mtime, so periodic rescans of a large library do not re-read the files.

## Reliability

- **Idempotent**: an existing output is never overwritten (`--force` to redo). A human-made subtitle
  downloaded by Bazarr always wins.
- **Resumable**: every translated batch is cached in SQLite — a 503 halfway through a film only costs the current batch.
- **Resilient**:
  - capacity errors (429, 5xx, timeouts): backoff honouring `Retry-After`, longer waits on 429 (quota windows),
    then the next model in `MODELS`. If every model is overloaded, the run **stops cleanly** instead of hammering
    the API: `file` exits with code `75` (EX_TEMPFAIL) and `scan` stops until the next cycle. Progress is cached,
    rerunning resumes where it stopped;
  - broken output (invalid JSON, missing ids): retry, then split the batch in half;
  - retired model (404): disabled for the rest of the run.
- **Atomic writes** (`.tmp` + rename): the media server never sees a half-written file.
- Ignores sources modified less than `SETTLE_S` ago (Bazarr still writing).
- The API key is sent as a header, never in the URL, so it does not leak into proxy/app logs.

## Quick start (Docker Compose)

```bash
cp .env.example .env && $EDITOR .env      # API_KEY=...
$EDITOR docker-compose.yml                 # set your media volume
docker compose pull
# try a single file first
docker compose run --rm subtrans file "/media/Movies/X (1993)/X.mkv" -v
# then run as a service (rescans every 15 min)
docker compose up -d
```

## Kubernetes

`k8s/cronjob.yaml` runs a scan every 30 minutes (`concurrencyPolicy: Forbid`), media over NFS, API key from a Secret:

```bash
kubectl -n media create secret generic subtrans-api --from-literal=API_KEY=...
kubectl apply -f k8s/cronjob.yaml
```

## Glossary

`glossary.txt` in the film folder (or a global one via `GLOSSARY_FILE`):

```
Burikin Hotel => ブリキンホテル / Hôtel Burikin
Bamboo copter => タケコプター / hélicoptère de bambou
```

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `BACKEND` | `gemini` | `gemini` \| `anthropic` \| `openai` \| `dummy` (offline tests) |
| `API_KEY` | — | not needed for a local OpenAI-compatible server |
| `MODELS` | `gemini-3.8-flash` | comma-separated, tried in order |
| `OPENAI_BASE_URL` | `http://ollama:11434/v1` | `openai` backend only |
| `TARGET_LANGS` | `fr,ja` | |
| `REQUIRE_AUDIO_LANGS` | `ja` | only process videos with one of these audio languages; empty = all |
| `EXTRACT_EMBEDDED` | `1` | fall back to the embedded English track |
| `KEEP_EXTRACTED` | `0` | `1` = write the extracted `<video>.en.srt` next to the video |
| `VIDEO_EXTS` | `.mkv,.mp4,.m4v,.avi` | |
| `FURIGANA` | `1` | generate `.ja.ass` |
| `BATCH_SIZE` / `CONTEXT_CUES` | `60` / `8` | |
| `MAX_RETRIES` / `MIN_INTERVAL_S` | `6` / `1.0` | per model / seconds between requests |
| `MAX_BACKOFF_S` | `300` | longest wait between retries |
| `HTTP_TIMEOUT_S` | `180` | raise it for slow "thinking" models |
| `LIBRARY_PATHS` | `/media` | `:`-separated |
| `SETTLE_S` | `120` | |
| `CACHE_DB` | `/cache/subtrans.db` | |

## Playback notes

- Jellyfin web/desktop renders ASS client-side: enable **fallback fonts** (Dashboard → Playback) and drop a
  Japanese font there, e.g. `NotoSansCJKjp-Regular.otf` (not the `.ttc`).
- TV clients often ignore ASS positioning: set subtitles to **burn-in** for this track, or use mpv / Jellyfin Media Player.

## Image

`ghcr.io/scharret/jimaku-llm`, built by CI: `latest` and `sha-*` (amd64) on every push to `main`; semver tags (amd64 + arm64) on `v*` tags.

## Development

```bash
# needs ffmpeg on PATH for the embedded-subtitle test
pip install requests "fugashi[unidic-lite]" pytest && pytest -q
```

## Contributing

- Work on a branch and open a PR to `main`; `main` is protected.
- Commits and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `chore:`, `ci:`, `refactor:`, `test:`, `perf:`, `build:`), enforced in CI.
- PRs are squash-merged: the PR title becomes the commit message on `main`.

## Disclaimer

Machine translation: good enough to follow a film, not a substitute for official subtitles.
Translate only subtitles you have the right to use.
