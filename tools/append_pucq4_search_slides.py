"""Append evidence slides to the existing PUCQ4 deck, preserving prior slides.

The JSON specification supplies titles, text, optional images, and provenance.
This is an offline reporting utility; it never imports acquisition modules.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


def text_box(slide, text, x, y, width, height, size=18, bold=False, color=(35, 46, 63)):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = RGBColor(*color)
        paragraph.space_after = Pt(9)
    return box


def fit_image(slide, path, x, y, width, height):
    with Image.open(path) as image:
        ratio = image.width / image.height
    actual_width = min(width, height * ratio)
    actual_height = actual_width / ratio
    slide.shapes.add_picture(str(path),
                             Inches(x + (width - actual_width) / 2),
                             Inches(y + (height - actual_height) / 2),
                             width=Inches(actual_width))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text(encoding="utf-8-sig"))
    original = args.deck.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    presentation = Presentation(args.deck)
    count = len(presentation.slides)
    width = presentation.slide_width / Inches(1)
    height = presentation.slide_height / Inches(1)
    for entry in specification["slides"]:
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(248, 250, 252)
        text_box(slide, entry["title"], 0.4, 0.18, width - 0.8, 0.7, 27, True)
        text_box(slide, entry.get("subtitle", "M4 / Python index 3 / physical number unassigned"),
                 0.4, 0.92, width - 0.8, 0.45, 13, color=(69, 88, 109))
        images = entry.get("images", [])
        if len(images) > 2:
            raise ValueError("Use at most two evidence images per slide")
        if images:
            available_width = (width - 0.9) / len(images)
            for index, image in enumerate(images):
                fit_image(slide, image, 0.35 + index * available_width,
                          1.43, available_width, height - 3.2)
            text_box(slide, "\n".join(entry.get("lines", [])), 0.4,
                     height - 1.65, width - 0.8, 1.14, 13)
        else:
            text_box(slide, "\n".join(entry.get("lines", [])), 0.5, 1.55,
                     width - 1.0, height - 2.2, entry.get("font_size", 20))
        text_box(slide, "M4 search | 2026-09-08 | " + entry.get("status", "in progress"),
                 0.4, height - 0.39, width - 0.8, 0.25, 10, color=(96, 112, 128))
        slide.notes_slide.notes_text_frame.text = entry.get("provenance", "")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.deck.with_name(args.deck.stem + "_pre_m4_" + stamp + ".pptx")
    pending = args.deck.with_name(args.deck.stem + "_m4_pending_" + stamp + ".pptx")
    presentation.save(pending)
    if len(Presentation(pending).slides) != count + len(specification["slides"]):
        raise RuntimeError("Saved presentation slide count mismatch")
    if hashlib.sha256(args.deck.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("Original deck changed during edit; pending copy retained")
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(args.deck, backup)
    pending.replace(args.deck)
    print(json.dumps(dict(deck=str(args.deck), backup=str(backup),
                          previous_slides=count, slides=len(presentation.slides),
                          added=list(range(count + 1, len(presentation.slides) + 1))), indent=2))


if __name__ == "__main__":
    main()
