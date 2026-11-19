"""Turn an ASS/SSA track into clean dialogue cues.

Fansub ASS tracks carry typesetting (signs), karaoke and frame-by-frame effects. A naive
ASS -> SRT conversion turns every effect line into a cue: a film can end up with 100k+
cues, which is both useless and very expensive to translate. Keep the dialogue only.
"""
import re

NON_DIALOGUE_STYLE = re.compile(
    r"sign|song|kara|kfx|lyric|romaji|kanji|title|typeset|insert|note|credit|staff|logo|"
    r"(^|[^a-z])(op|ed)([^a-z]|$)", re.I)
# Override tags typical of typesetting / karaoke / animation, never of plain dialogue.
EFFECT_TAGS = re.compile(r"\\(pos|move|k|K|kf|ko|t\(|clip|iclip|org|fr[xyz]?|fa[xy]|p[1-9])", re.I)
TAG = re.compile(r"\{[^}]*\}")


def _t(ts):
    h, m, s = ts.strip().split(":")
    sec, cs = (s.split(".") + ["0"])[:2]
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(cs.ljust(2, "0")[:2]) / 100


def _srt_time(x):
    ms = int(round(x * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_to_cues(text):
    """Returns (cues, stats). cues: [(index, 'hh:mm:ss,mmm --> hh:mm:ss,mmm', text)]."""
    fmt, events, total = None, [], 0
    in_events = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            fmt = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
            continue
        if not line.lower().startswith("dialogue:") or not fmt:
            continue
        total += 1
        values = line.split(":", 1)[1].split(",", len(fmt) - 1)
        if len(values) != len(fmt):
            continue
        ev = dict(zip(fmt, (v.strip() if k != "text" else v for k, v in zip(fmt, values))))
        raw = ev.get("text", "")
        if (NON_DIALOGUE_STYLE.search(ev.get("style", "")) or NON_DIALOGUE_STYLE.search(ev.get("name", ""))
                or ev.get("effect") or EFFECT_TAGS.search(raw)):
            continue
        clean = TAG.sub("", raw).replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
        clean = "\n".join(" ".join(part.split()) for part in clean.split("\n") if part.strip())
        if not clean:
            continue
        try:
            events.append([_t(ev["start"]), _t(ev["end"]), clean])
        except (KeyError, ValueError):
            continue
    events.sort(key=lambda e: (e[0], e[1]))
    merged = []
    for start, end, txt in events:
        prev = merged[-1] if merged else None
        if prev and prev[2] == txt and start <= prev[1] + 0.1:   # layered duplicates / split lines
            prev[1] = max(prev[1], end)
            continue
        merged.append([start, end, txt])
    cues = [(i + 1, f"{_srt_time(s)} --> {_srt_time(e)}", t) for i, (s, e, t) in enumerate(merged)]
    return cues, {"events": total, "cues": len(cues)}


def render_srt(cues):
    return "".join(f"{i}\n{tc}\n{t}\n\n" for i, tc, t in cues)
