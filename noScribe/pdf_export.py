"""Export a noScribe HTML transcript to PDF.

The PDF is built from the same HTML document that the other output formats use,
so speaker labels, timestamps and pause markers come out the same as in the
.txt export. fpdf2 is pure Python, which keeps the PyInstaller builds simple.
"""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path

from . import branding
from .utils import html_to_text

# Unicode TrueType fonts that are usually present. The first one found is used;
# without one, the PDF falls back to Helvetica, which only covers Latin-1.
_FONT_CANDIDATES = {
    "Windows": [
        (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
        (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ],
    "Darwin": [
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ],
    "Linux": [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    ],
}

# "S01: [00:01:02] text" or "Name Surname: text"; the label must be short and
# must not contain sentence punctuation, so ordinary sentences with a colon
# are not mistaken for a speaker turn.
_SPEAKER_RE = re.compile(r"^(?P<label>[^\s:.!?][^:.!?\n]{0,40}):\s+(?P<rest>.*)$", re.S)
_TIMESTAMP_RE = re.compile(r"^(?P<ts>\[\d{2}:\d{2}:\d{2}\])\s*(?P<rest>.*)$", re.S)


def _find_fonts():
    for regular, bold in _FONT_CANDIDATES.get(platform.system(), []):
        if os.path.exists(regular) and os.path.exists(bold):
            return regular, bold
    return None


def _latin1(text: str) -> str:
    """Make text safe for the built-in Helvetica font."""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u20ac": "EUR",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def split_paragraphs(html_str: str) -> list[str]:
    text = html_to_text(html_str, use_only_body=True)
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def html_to_pdf_bytes(html_str: str, *, footer_label: str = "") -> bytes:
    """Render the transcript HTML to PDF and return the file content."""
    from fpdf import FPDF  # imported lazily; only needed for PDF output

    fonts = _find_fonts()
    clean = (lambda s: s) if fonts else _latin1

    class TranscriptPDF(FPDF):
        def header(self):
            # Thin stripe in the four logo colours across the top of each page.
            width = self.w / len(branding.LOGO_STRIPE)
            for i, colour in enumerate(branding.LOGO_STRIPE):
                self.set_fill_color(*branding.hex_to_rgb(colour))
                self.rect(i * width, 0, width, 2.5, style="F")
            self.set_y(14)

        def footer(self):
            self.set_y(-14)
            self.set_font(family, "", 8)
            self.set_text_color(*branding.hex_to_rgb(branding.GREY_TEXT))
            label = f"{footer_label}  |  " if footer_label else ""
            self.cell(0, 6, clean(f"{label}{self.page_no()} / {{nb}}"), align="R")

    pdf = TranscriptPDF(format="A4")
    pdf.set_margins(20, 14, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_title("noScribe transcript")
    pdf.set_creator("noScribe")
    if fonts:
        family = "TranscriptSans"
        pdf.add_font(family, "", fonts[0])
        pdf.add_font(family, "B", fonts[1])
    else:
        family = "Helvetica"
    pdf.alias_nb_pages()
    pdf.add_page()

    blue = branding.hex_to_rgb(branding.BLUE)
    grey = branding.hex_to_rgb(branding.GREY_TEXT)
    black = (33, 37, 41)

    paragraphs = split_paragraphs(html_str)
    if not paragraphs:
        pdf.set_font(family, "", 11)
        pdf.cell(0, 8, "")
        return bytes(pdf.output())

    # Title
    title, *rest = paragraphs
    pdf.set_font(family, "B", 18)
    pdf.set_text_color(*blue)
    pdf.multi_cell(0, 9, clean(title), align="L")
    pdf.ln(2)

    # The block right after the title holds "Transcribed with ...", audio file
    # and options; show it small and grey.
    if rest and "noScribe" in rest[0]:
        meta, *rest = rest
        pdf.set_font(family, "", 8.5)
        pdf.set_text_color(*grey)
        pdf.multi_cell(0, 4.5, clean(meta))
        pdf.ln(4)

    for para in rest:
        match = _SPEAKER_RE.match(para)
        pdf.set_x(pdf.l_margin)
        if match:
            label, body = match.group("label"), match.group("rest")
            pdf.set_font(family, "B", 10.5)
            pdf.set_text_color(*blue)
            pdf.write(5.5, clean(label + ": "))
            ts = _TIMESTAMP_RE.match(body)
            if ts:
                pdf.set_font(family, "", 9)
                pdf.set_text_color(*grey)
                pdf.write(5.5, clean(ts.group("ts") + " "))
                body = ts.group("rest")
            pdf.set_font(family, "", 10.5)
            pdf.set_text_color(*black)
            pdf.write(5.5, clean(body))
        else:
            # Pause markers and other loose paragraphs
            pdf.set_font(family, "", 10)
            pdf.set_text_color(*grey)
            pdf.write(5.5, clean(para))
        pdf.ln(8)

    return bytes(pdf.output())


def write_pdf(html_str: str, path, *, footer_label: str = "") -> None:
    Path(path).write_bytes(html_to_pdf_bytes(html_str, footer_label=footer_label))
