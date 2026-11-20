"""OCR of image-based PGS subtitles (Blu-ray .sup).

Pipeline: ffmpeg copies the PGS track to a .sup file -> this module parses the PGS
segments and decodes the RLE bitmaps -> each subtitle becomes a black-on-white image ->
text via Tesseract (local) or a vision LLM (Gemini), depending on the language.
"""
import base64
import hashlib
import io
import json
import logging
import struct
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

log = logging.getLogger("subtrans")

PCS, WDS, PDS, ODS, END = 0x16, 0x17, 0x14, 0x15, 0x80
TESSERACT_LANGS = {"en": "eng", "fr": "fra", "ja": "jpn", "es": "spa", "de": "deu", "it": "ita"}


# ---------------------------------------------------------------- PGS parsing
def _rle_decode(data, width, height):
    """PGS RLE -> list of palette indices (row-major)."""
    px = bytearray(width * height)
    i = x = y = 0
    n = len(data)
    while i < n and y < height:
        b = data[i]
        i += 1
        if b:
            if x < width:
                px[y * width + x] = b
            x += 1
            continue
        b2 = data[i]
        i += 1
        if b2 == 0:            # end of line
            x, y = 0, y + 1
            continue
        flag = b2 & 0xC0
        if flag == 0x00:
            run, color = b2 & 0x3F, 0
        elif flag == 0x40:
            run, color = ((b2 & 0x3F) << 8) | data[i], 0
            i += 1
        elif flag == 0x80:
            run, color = b2 & 0x3F, data[i]
            i += 1
        else:
            run, color = ((b2 & 0x3F) << 8) | data[i], data[i + 1]
            i += 2
        if color:
            start = y * width + x
            end = min(start + run, y * width + width)
            px[start:end] = bytes([color]) * (end - start)
        x += run
    return px


def parse_sup(data):
    """Yields (start_s, end_s, [(x, y, w, h, indices, palette)]) for each displayed subtitle."""
    pos, palettes, objects = 0, {}, {}
    ds = None      # display set being read: (pts, [(obj_id, x, y)], palette_id)
    shown = None   # subtitle on screen: (start, [(x, y, w, h, rle)], palette)
    out = []
    while pos + 13 <= len(data):
        if data[pos:pos + 2] != b"PG":
            pos += 1
            continue
        pts = struct.unpack(">I", data[pos + 2:pos + 6])[0] / 90000
        seg_type = data[pos + 10]
        size = struct.unpack(">H", data[pos + 11:pos + 13])[0]
        seg = data[pos + 13:pos + 13 + size]
        pos += 13 + size
        if seg_type == PCS:
            comps, o = [], 11
            for _ in range(seg[10]):
                obj_id, _win, cropped, ox, oy = struct.unpack(">HBBHH", seg[o:o + 8])
                comps.append((obj_id, ox, oy))
                o += 8 + (8 if cropped & 0x40 else 0)
            ds = (pts, comps, seg[9])
        elif seg_type == PDS:
            pal = palettes.setdefault(seg[0], {})
            for e in range(2, len(seg) - 4, 5):
                idx, y_, _cr, _cb, a = seg[e:e + 5]
                pal[idx] = (y_, a)
        elif seg_type == ODS:
            obj_id = struct.unpack(">H", seg[0:2])[0]
            if seg[3] & 0x80:   # first fragment carries the size
                w, h = struct.unpack(">HH", seg[7:11])
                objects[obj_id] = [w, h, bytearray(seg[11:])]
            elif obj_id in objects:
                objects[obj_id][2] += seg[4:]
        elif seg_type == END and ds is not None:
            t, comps, pid = ds
            if shown:  # whatever is on screen ends when a new display set takes effect
                out.append((shown[0], t, shown[1], shown[2]))
                shown = None
            frozen = [(ox, oy, *objects[oid][:2], bytes(objects[oid][2]))
                      for oid, ox, oy in comps if oid in objects]
            if frozen:
                shown = (t, frozen, dict(palettes.get(pid, {})))
            ds = None
    if shown:
        out.append((shown[0], shown[0] + 5, shown[1], shown[2]))
    for start, end, objs, pal in merge_repeats(out):
        if end - start < 0.05:
            continue
        yield start, end, [(ox, oy, w, h, _rle_decode(rle, w, h), pal) for ox, oy, w, h, rle in objs]


def _opacity(pal):
    return sum(a for _, a in pal.values())


def merge_repeats(sets, gap=0.25):
    """Blu-ray PGS re-sends the same bitmap many times (acquisition points, palette-only fades).

    Consecutive display sets with identical bitmaps are one subtitle: merge them, keeping the most
    opaque palette (a fade-in starts transparent, which would hide the text from OCR).
    """
    merged = []
    for start, end, objs, pal in sets:
        prev = merged[-1] if merged else None
        if prev and prev[2] == objs and start - prev[1] <= gap:
            prev[1] = max(prev[1], end)
            if _opacity(pal) > _opacity(prev[3]):
                prev[3] = pal
            continue
        merged.append([start, end, objs, pal])
    return merged


def render(objs, pad=12):
    """Black text on white background, cropped to the objects, for OCR."""
    x0 = min(o[0] for o in objs)
    y0 = min(o[1] for o in objs)
    x1 = max(o[0] + o[2] for o in objs)
    y1 = max(o[1] + o[3] for o in objs)
    img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 255)
    for ox, oy, w, h, idx, pal in objs:
        # bright, opaque palette entries are the glyph fill; dark ones are the outline/shadow
        lut = [0 if (i in pal and pal[i][1] > 128 and pal[i][0] > 128) else 255 for i in range(256)]
        layer = Image.frombytes("L", (w, h), bytes(idx)).point(lut)
        mask = layer.point(lambda v: 255 if v == 0 else 0)
        img.paste(layer, (ox - x0 + pad, oy - y0 + pad), mask)
    return img


def _ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------- OCR engines
def ocr_tesseract(images, lang):
    code = TESSERACT_LANGS.get(lang, lang)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, img in enumerate(images):
            f = Path(tmp) / f"{i}.png"
            # upscale small glyphs: Tesseract works best around 30px x-height
            if img.height < 80:
                img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
            img.save(f)
            r = subprocess.run(["tesseract", str(f), "stdout", "-l", code, "--psm", "6"],
                               capture_output=True, text=True, timeout=60)
            text = "\n".join(line.strip() for line in r.stdout.splitlines() if line.strip())
            if lang == "ja":
                text = text.replace(" ", "")  # Tesseract inserts spaces between CJK glyphs
            out.append(text)
    return out


def ocr_vision(images, lang, call_vision, batch=20):
    """call_vision(prompt, [(id, png_bytes)]) -> {id: text}, provided by main (retries, cache)."""
    out = []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        items = []
        for j, img in enumerate(chunk):
            buf = io.BytesIO()
            img.save(buf, "PNG", optimize=True)
            items.append((str(i + j + 1), buf.getvalue()))
        res = call_vision(lang, items)
        out += [res.get(k, "") for k, _ in items]
        log.info("  OCR %s: %d/%d", lang, min(i + batch, len(images)), len(images))
    return out


def vision_prompt(lang):
    from .main import LANG_NAMES
    name = LANG_NAMES.get(lang, lang)
    ruby = ("Some kanji have small furigana (ruby) printed above them: do NOT transcribe the "
            "furigana, only the main-size text, keeping each kanji as written. " if lang == "ja" else "")
    return (f"Each image is one subtitle from a film, in {name}. Transcribe the text exactly "
            "(no translation, no description). " + ruby + "Keep line breaks as \\n. Output ONLY a JSON "
            "object {id: text} with the ids given before each image; use \"\" for an image without text.")


def images_key(items):
    h = hashlib.sha256()
    for k, png in items:
        h.update(k.encode())
        h.update(hashlib.sha256(png).digest())
    return h.hexdigest()


def b64(png):
    return base64.b64encode(png).decode()


# ---------------------------------------------------------------- entry point
class TooManyImages(Exception):
    pass


def ocr_track(video: Path, track_index, lang, engine, call_vision=None, max_images=None):
    """OCR an embedded PGS track. Returns SRT text, or '' if nothing was recognised."""
    with tempfile.TemporaryDirectory() as tmp:
        sup = Path(tmp) / "track.sup"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video),
                        "-map", f"0:{track_index}", "-c:s", "copy", str(sup)],
                       check=True, timeout=1800)
        sets = list(parse_sup(sup.read_bytes()))
    if not sets:
        return ""
    if max_images and len(sets) > max_images:
        # Same safety net as MAX_CUES for text sources: never send thousands of images to a paid API.
        raise TooManyImages(f"{len(sets)} subtitle images > MAX_CUES={max_images}")
    images = [render(objs) for _, _, objs in sets]
    log.info("OCR %s: %d subtitle images with %s", lang, len(images), engine)
    if engine == "vision":
        texts = ocr_vision(images, lang, call_vision)
    else:
        texts = ocr_tesseract(images, lang)
    cues, n = [], 0
    for (start, end, _), text in zip(sets, texts):
        if text.strip():
            n += 1
            cues.append(f"{n}\n{_ts(start)} --> {_ts(end)}\n{text.strip()}\n")
    return "\n".join(cues) + ("\n" if cues else "")


def dump_json(obj):
    return json.dumps(obj, ensure_ascii=False)
