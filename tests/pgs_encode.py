"""Tiny PGS (.sup) encoder used to build test fixtures: white text, black outline."""
import struct

from PIL import Image, ImageDraw, ImageFont


def _rle(img):
    w, h = img.size
    px = img.load()
    out = bytearray()
    for y in range(h):
        x = 0
        while x < w:
            c = px[x, y]
            run = 1
            while x + run < w and px[x + run, y] == c and run < 16383:
                run += 1
            if c == 0:
                out += bytes([0, run]) if run < 64 else bytes([0, 0x40 | (run >> 8), run & 0xFF])
            elif run < 3:
                out += bytes([c]) * run
            elif run < 64:
                out += bytes([0, 0x80 | run, c])
            else:
                out += bytes([0, 0xC0 | (run >> 8), run & 0xFF, c])
            x += run
        out += b"\x00\x00"
    return bytes(out)


def _seg(pts, kind, payload):
    t = int(pts * 90000)
    return b"PG" + struct.pack(">IIBH", t, t, kind, len(payload)) + payload


def text_image(text, font_path, size=48):
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("P", (1, 1)))
    x0, y0, x1, y1 = probe.multiline_textbbox((0, 0), text, font=font, stroke_width=3)
    img = Image.new("P", (x1 - x0 + 8, y1 - y0 + 8), 0)
    ImageDraw.Draw(img).multiline_text((4 - x0, 4 - y0), text, font=font, fill=1, stroke_width=3, stroke_fill=2)
    return img


def encode(subs, font_path, screen=(1920, 1080), repeats=0):
    """subs: [(start_s, end_s, text)] -> .sup bytes.

    repeats: re-send each subtitle N extra times (acquisition points, like real Blu-rays do),
    the first one with a transparent palette (fade-in).
    """
    out = bytearray()
    for n, (start, end, text) in enumerate(subs):
        img = text_image(text, font_path)
        w, h = img.size
        x, y = (screen[0] - w) // 2, screen[1] - h - 60
        pcs = struct.pack(">HHBHBBBB", screen[0], screen[1], 0x10, n * 2, 0x80, 0, 0, 1)
        pcs += struct.pack(">HBBHH", 0, 0, 0, x, y)
        wds = bytes([1, 0]) + struct.pack(">HHHH", x, y, w, h)
        # palette: 0 transparent, 1 white (Y=235), 2 black outline (Y=16)
        pds = bytes([0, 0]) + bytes([0, 16, 128, 128, 0, 1, 235, 128, 128, 255, 2, 16, 128, 128, 255])
        rle = _rle(img)
        ods = struct.pack(">HBB", 0, 0, 0xC0) + struct.pack(">I", len(rle) + 4)[1:] + struct.pack(">HH", w, h) + rle
        if repeats:  # fade-in: same bitmap, fully transparent palette
            faded = bytes([0, 0]) + bytes([0, 16, 128, 128, 0, 1, 235, 128, 128, 0, 2, 16, 128, 128, 0])
            out += _seg(start, 0x16, pcs) + _seg(start, 0x17, wds) + _seg(start, 0x14, faded)
            out += _seg(start, 0x15, ods) + _seg(start, 0x80, b"")
            start = start + 0.04
        out += _seg(start, 0x16, pcs) + _seg(start, 0x17, wds) + _seg(start, 0x14, pds)
        out += _seg(start, 0x15, ods) + _seg(start, 0x80, b"")
        step = (end - start) / (repeats + 1) if repeats else 0
        for r in range(1, repeats):  # acquisition points: identical display set re-sent
            t = start + r * step
            out += _seg(t, 0x16, pcs) + _seg(t, 0x17, wds) + _seg(t, 0x14, pds)
            out += _seg(t, 0x15, ods) + _seg(t, 0x80, b"")
        clear = struct.pack(">HHBHBBBB", screen[0], screen[1], 0x10, n * 2 + 1, 0x00, 0, 0, 0)
        out += _seg(end, 0x16, clear) + _seg(end, 0x17, wds) + _seg(end, 0x80, b"")
    return bytes(out)
