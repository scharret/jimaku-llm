"""Japanese SRT -> ASS with furigana positioned (\\pos) above each kanji."""
import re
import unicodedata

import fugashi

_TAGGER = None

# Reading fixes for words unidic often gets wrong out of context.
OVERRIDES = {
    "泳": "およ", "一日": "いちにち", "何時": "なんじ", "日本": "にほん",
    "私": "わたし", "野比": "のび", "何者": "なにもの", "冥界": "めいかい",
}

FS, RFS, X0 = 64, 30, 960
LINE_H = FS + RFS + 14

# libass (Jellyfin web, mpv, ffmpeg burn-in) scales a font so that its line height,
# usWinAscent + usWinDescent, equals the style Fontsize. For Noto Sans CJK JP that is
# 1160 + 288 = 1448 units, so one em (the advance of a kanji) is Fontsize / 1.448.
EM_RATIO = 1000 / 1448
# Advance widths of printable ASCII (U+0020..U+007E) in Noto Sans CJK JP, per 1000 em.
_ASCII_ADV = (
    224, 323, 474, 555, 555, 921, 680, 278, 338, 338, 467, 555, 278, 347, 278, 392,
    555, 555, 555, 555, 555, 555, 555, 555, 555, 555, 278, 278, 555, 555, 555, 474,
    946, 608, 657, 638, 688, 589, 552, 689, 728, 293, 535, 646, 543, 812, 723, 742,
    633, 742, 635, 596, 599, 721, 575, 878, 573, 531, 603, 338, 392, 338, 555, 559,
    606, 563, 618, 510, 620, 554, 325, 564, 607, 275, 275, 552, 284, 926, 610, 606,
    620, 620, 388, 468, 377, 607, 521, 802, 498, 521, 475, 338, 270, 338, 555,
)

HEADER = f"""[Script Info]
Title: {{title}}
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Base,Noto Sans CJK JP,{FS},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3.5,1.5,2,0,0,0,128
Style: Ruby,Noto Sans CJK JP,{RFS},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2.2,1,2,0,0,0,128

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        _TAGGER = fugashi.Tagger()
    return _TAGGER


def _hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _is_kanji(c):
    return "一" <= c <= "鿿" or c in "々〆ヶ"


def _width(c):
    """Advance width of a character, in em."""
    o = ord(c)
    if 0x20 <= o <= 0x7E:
        return _ASCII_ADV[o - 0x20] / 1000
    eaw = unicodedata.east_asian_width(c)
    return 1.0 if eaw in "WFA" else 0.5 if eaw == "H" else 0.55


def ruby_segments(line):
    """[(text, reading|None)] with okurigana split from the kanji part."""
    segs = []
    toks = list(_tagger()(line))
    for i, m in enumerate(toks):
        if m.white_space:  # MeCab drops spaces from surfaces; keep them for the layout
            segs.append((m.white_space, None))
        s = m.surface
        r = OVERRIDES.get(s) or _hira(m.feature.kana or "")
        if s == "何":  # なに vs なん depending on what follows
            nx = toks[i + 1].surface if i + 1 < len(toks) else ""
            r = "なん" if (nx[:1] in "でだのと度泊日百千" or nx.startswith("キロ")) else "なに"
        if not any(_is_kanji(c) for c in s) or not r:
            segs.append((s, None))
            continue
        a = b = 0
        while a < len(s) and not _is_kanji(s[a]) and a < len(r) and _hira(s[a]) == r[a]:
            a += 1
        while (b < len(s) - a and not _is_kanji(s[-1 - b]) and b < len(r) - a
               and _hira(s[-1 - b]) == r[-1 - b]):
            b += 1
        core, core_r = s[a:len(s) - b], r[a:len(r) - b]
        if a:
            segs.append((s[:a], None))
        segs.append((core, core_r if core_r and core_r != _hira(core) else None))
        if b:
            segs.append((s[len(s) - b:], None))
    return segs


_SDH_BRACKETS = re.compile(r"[\[［【〔][^\]］】〕]*[\]］】〕]|[(（][^)）]*[)）]")
_SDH_MARKS = re.compile(r"[♪♫≪≫〈〉＜＞→➡]")


def strip_sdh(text):
    """Remove hearing-impaired annotations: [sound], （sigh）, （Name）speaker tags, ♪ music marks, ≪off-screen≫."""
    out = []
    for line in text.split("\n"):
        line = _SDH_MARKS.sub("", _SDH_BRACKETS.sub("", line))
        line = line.strip(" 　～〜~-－")
        if line:
            out.append(line)
    return "\n".join(out)


def strip_sdh_cues(cues):
    """Same cues without annotations; cues left empty are dropped and the rest renumbered."""
    kept = [(tc, strip_sdh(t)) for _, tc, t in cues]
    kept = [(tc, t) for tc, t in kept if t]
    return [(i + 1, tc, t) for i, (tc, t) in enumerate(kept)]


def _ts(tc):
    h, m, rest = tc.strip().split(":")
    s, ms = rest.split(",")
    return f"{int(h)}:{m}:{s}.{ms[:2]}"


def _base(start, end, x, y, text):
    """Base text left-aligned at x. libass drops leading spaces, so skip over them here."""
    lead = len(text) - len(text.lstrip())
    x += sum(_width(c) for c in text[:lead]) * FS * EM_RATIO
    text = text.strip()
    return f"Dialogue: 0,{start},{end},Base,,0,0,0,,{{\\an1\\pos({x:.0f},{y})}}{text}" if text else None


def _line_events(line, start, end, y):
    """Each kanji word is its own event, with its ruby centred on it: an error in the
    assumed glyph widths shifts whole words slightly but never moves a ruby off its kanji."""
    em, rem = FS * EM_RATIO, RFS * EM_RATIO
    segs = ruby_segments(line)
    widths = [sum(_width(c) for c in s) * em for s, _ in segs]
    x = X0 - sum(widths) / 2
    events, run, run_x = [], "", x
    for (s, r), sw in zip(segs, widths):
        if r:
            events += [_base(start, end, run_x, y, run), _base(start, end, x, y, s)]
            room, need = max(sw, em), len(r) * rem
            sc = min(100, int(100 * room / need)) if need > room * 1.15 else 100
            events.append(
                f"Dialogue: 1,{start},{end},Ruby,,0,0,0,,"
                f"{{\\an2\\pos({x + sw / 2:.0f},{y - FS - 2})\\fscx{sc}}}{r}")
            run, run_x = "", x + sw
        else:
            run += s
        x += sw
    events.append(_base(start, end, run_x, y, run))
    return [e for e in events if e]


def cues_to_ass(cues, title="subs"):
    """cues: [(idx, '00:00:01,000 --> 00:00:02,000', 'texte\\nligne2')]"""
    import re
    events = []
    for _, tc, text in cues:
        start, end = (_ts(x) for x in tc.split("-->"))
        lines = [re.sub(r"</?[ib]>", "", x).replace("- ", "－") for x in text.split("\n") if x.strip()]
        n = len(lines)
        for i, line in enumerate(lines):
            y = 1080 - 50 - (n - 1 - i) * LINE_H
            events.extend(_line_events(line, start, end, y))
    return HEADER.replace("{title}", title) + "\n".join(events) + "\n"
