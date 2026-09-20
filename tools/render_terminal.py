"""Render a captured chat.py session log as a PNG (and an SVG copy).

The input is the verbatim pty capture written by tools/record_chat_session.py.
Nothing is retyped, reordered or edited here: this only draws the captured
characters in a terminal-styled frame so the session can be embedded in the
README. Compare any image with its .txt log - they carry the same characters.

The PNG is drawn directly with Pillow using a real monospace face, so the glyph
advance is measured rather than guessed and no line can overflow the frame. The
SVG is written from the same measured advance for readers who prefer vector.

    python tools/render_terminal.py results/chat/expanded_terminal_session.txt \
        --title "chat.py - expanded-corpus model"
"""
import argparse
import html
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    ("/System/Library/Fonts/Menlo.ttc", 0),
    ("/System/Library/Fonts/SFNSMono.ttf", 0),
    ("/System/Library/Fonts/Courier.ttc", 0),
]
SIZE, SCALE, PAD, BAR, WRAP = 15, 2, 20, 38, 104
BG, BAR_BG, FG = (17, 21, 28), (27, 33, 43), (215, 221, 229)
COLORS = {"you": (127, 209, 185), "model": (240, 198, 116),
          "note": (224, 121, 109), "dim": (124, 135, 152)}
HEX = {"you": "#7fd1b9", "model": "#f0c674", "note": "#e0796d",
       "dim": "#7c8798", None: "#d7dde5"}


def load_font(size):
    for path, index in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=index)
            except OSError:
                continue
    return ImageFont.load_default(size)


def classify(line):
    if line.startswith("You: "):
        return "you"
    if line.startswith("Model: "):
        return "model"
    if line.startswith(("Unknown words:", "Long prompt:")):
        return "note"
    if line.startswith(("number of parameters", "Tiny language model",
                        "Each prompt", "Saved transcript")):
        return "dim"
    return None


def layout(path):
    rows = []
    for raw in path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n").split("\n"):
        kind = classify(raw)
        for index, piece in enumerate(textwrap.wrap(raw, WRAP, subsequent_indent="    ") or [""]):
            rows.append((piece, kind, index > 0))
    return rows


def render(rows, title, png_path, svg_path):
    font = load_font(SIZE * SCALE)
    advance = font.getlength("M")
    line_h = round(SIZE * SCALE * 1.45)
    pad, bar = PAD * SCALE, BAR * SCALE
    width = round(pad * 2 + advance * (WRAP + 1))
    height = bar + pad + line_h * len(rows) + pad

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, bar], fill=BAR_BG)
    for index, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx, r = pad + index * 14 * SCALE, 5 * SCALE
        draw.ellipse([cx - r, bar // 2 - r, cx + r, bar // 2 + r], fill=color)
    title_font = load_font(round(SIZE * SCALE * 0.92))
    draw.text((width // 2, bar // 2), title, font=title_font,
              fill=(139, 149, 165), anchor="mm")
    for index, (text, kind, _) in enumerate(rows):
        draw.text((pad, bar + pad + line_h * index), text, font=font,
                  fill=COLORS.get(kind, FG))
    image = image.resize((width // SCALE, height // SCALE), Image.LANCZOS)
    image.save(png_path)

    # Same layout, expressed as vector, using the measured advance.
    unit = advance / SCALE
    w, h = width // SCALE, height // SCALE
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="{SIZE}">',
           f'<rect width="{w}" height="{h}" rx="10" fill="rgb{BG}"/>',
           f'<rect width="{w}" height="{BAR}" fill="rgb{BAR_BG}"/>']
    for index, color in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        out.append(f'<circle cx="{PAD + index * 14}" cy="{BAR // 2}" r="5" fill="{color}"/>')
    out.append(f'<text x="{w // 2}" y="{BAR // 2 + 5}" fill="#8b95a5" text-anchor="middle" '
               f'font-size="{SIZE * 0.92:.0f}">{html.escape(title)}</text>')
    for index, (text, kind, _) in enumerate(rows):
        if not text:
            continue
        y = BAR + PAD + (line_h // SCALE) * index + SIZE
        out.append(f'<text x="{PAD}" y="{y}" fill="{HEX[kind]}" xml:space="preserve" '
                   f'textLength="{len(text) * unit:.1f}" lengthAdjust="spacingAndGlyphs">'
                   f'{html.escape(text)}</text>')
    out.append("</svg>")
    svg_path.write_text("\n".join(out), encoding="utf-8")
    return image.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--title", default="chat.py")
    args = parser.parse_args()
    rows = layout(args.log)
    size = render(rows, args.title, args.log.with_suffix(".png"), args.log.with_suffix(".svg"))
    print(f"Wrote {args.log.with_suffix('.png')} {size} and {args.log.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
