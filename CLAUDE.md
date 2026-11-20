# CLAUDE.md

Guidance for Claude Code, both in local sessions and as `@claude` in GitHub Actions.
The README is the user-facing reference; read it before changing behaviour.

## Project

`jimaku-llm` scans a Jellyfin/Plex library, translates subtitles with an LLM (`*.fr.srt`, `*.ja.srt`)
and builds a Japanese `*.ja.ass` with furigana. Python ≥ 3.11, three dependencies
(`requests`, `fugashi[unidic-lite]`, `pillow`), no framework.

| file | role |
|---|---|
| `subtrans/main.py` | config (`CFG` from env vars), LLM backends and retries, SQLite cache, scan/file/report CLI |
| `subtrans/embedded.py` | ffprobe/ffmpeg: track selection, forced/SDH/"full" title rules, extraction |
| `subtrans/ocr.py` | PGS (`.sup`) parsing, display-set merging, Tesseract / vision-LLM OCR |
| `subtrans/ass_clean.py` | embedded ASS → dialogue cues (drops signs, songs, effects) |
| `subtrans/furigana.py` | MeCab/UniDic readings → positioned ASS ruby (deterministic, no LLM) |
| `subtrans/metrics.py` | Prometheus `/metrics` |
| `tests/test_pipeline.py` | the whole test suite; `tests/pgs_encode.py` builds PGS fixtures |

## Commands

```bash
pip install requests "fugashi[unidic-lite]" pillow pytest
pytest -q
```

Tests force `BACKEND=dummy`: they must **never** call a real API. Tests needing `ffmpeg` or
`tesseract` are skipped when the binary is missing; CI installs both and runs everything.

## Invariants (do not break)

- **Money.** Every LLM call costs real money (incident #14: 5 € drained by 5,900 vision calls on one film).
  Any change that can multiply API calls (batching, OCR, retries, cache keys) needs a cap and a test
  proving the cap holds *before* any call.
- **Cache keys.** If a prompt changes, its cache key must change too (`PROMPT_VERSION` for translation,
  prompt hash for OCR), otherwise stale results are reused.
- **Never overwrite** a file we did not write; human subtitles (Bazarr) always win. `--force` only rewrites our own files.
- **Never translate a translation**: files we generated are never a source.
- **Atomic writes** only (`write_atomic`).
- **API keys** go in headers, never in URLs or logs.
- A new env var goes in `CFG`, in the README configuration table, and in `.env.example` if users must set it.
- A new skip/fail reason goes in the README "What was skipped" table.

## Style

- Small, readable functions; comments explain *why*. Match the existing code, no new dependency without a reason.
- Code, comments, commits, PR titles and descriptions in **English**.

## Git and PRs

- Never push to `main` (protected). One branch per change, based on `main`: `feat/…`, `fix/…`, `ci/…`, `docs/…`.
- [Conventional Commits](https://www.conventionalcommits.org/) for commits and PR titles
  (`feat`, `fix`, `docs`, `chore`, `ci`, `refactor`, `test`, `perf`, `build`); CI enforces it.
  PRs are squash-merged, so the PR title becomes the commit on `main`.
- PR description: what was observed (symptom, example), root cause, the fix, the tests added,
  and anything the owner must do (delete a cached file, restart, merge order).
- Every bug fix comes with a test that fails without the fix.

## As `@claude` in GitHub Actions

- Only the repo owner can trigger you.
- When asked to implement something from an issue: create a branch, commit, push, then **open the PR yourself**
  with `gh pr create --base main` (do not just post a "Create PR" link), and link the issue (`Closes #N`).
- On an existing PR: push fixes to that PR's branch, do not open a new PR.
- Run `pytest -q` before pushing and say in your comment which tests were skipped.
- **Never merge**, never approve, never change branch protection or secrets: the owner reviews and merges.
- Reply to the owner in English; keep code and PR text in English.
