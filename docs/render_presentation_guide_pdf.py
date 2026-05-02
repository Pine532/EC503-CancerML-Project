from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "docs" / "PROJECT_PRESENTATION_GUIDE.md"
OUTPUT_PATH = PROJECT_DIR / "docs" / "EC503_CancerML_Project_Presentation_Guide.pdf"

PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11
LEFT = 0.7
TOP = 10.35
BOTTOM = 0.6
LINE_HEIGHT = 0.18
MONO_LINE_HEIGHT = 0.16


def strip_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text


def wrap_text(text: str, width: int) -> list[str]:
    if not text:
        return [""]

    return textwrap.wrap(
        strip_inline_markdown(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def parse_markdown(path: Path) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    in_code = False
    code_lines: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code:
                blocks.append(("code", "\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line:
            blocks.append(("blank", ""))
        elif line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
        elif line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
        elif line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
        elif line.startswith("- "):
            blocks.append(("bullet", line[2:].strip()))
        else:
            blocks.append(("paragraph", line))

    if code_lines:
        blocks.append(("code", "\n".join(code_lines)))

    return blocks


def new_page(pdf: PdfPages, page_number: int):
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(PAGE_WIDTH / 2, 0.28, str(page_number), ha="center", va="center", fontsize=8)
    return fig, ax, TOP


def finish_page(pdf: PdfPages, fig) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def ensure_space(pdf: PdfPages, fig, ax, y: float, needed: float, page_number: int):
    if y - needed >= BOTTOM:
        return fig, ax, y, page_number

    finish_page(pdf, fig)
    page_number += 1
    fig, ax, y = new_page(pdf, page_number)
    return fig, ax, y, page_number


def render_pdf(blocks: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        page_number = 1
        fig, ax, y = new_page(pdf, page_number)

        for kind, content in blocks:
            if kind == "blank":
                y -= LINE_HEIGHT * 0.6
                continue

            if kind == "title":
                lines = wrap_text(content, 48)
                needed = 0.34 * len(lines) + 0.28
                fig, ax, y, page_number = ensure_space(pdf, fig, ax, y, needed, page_number)
                for line in lines:
                    ax.text(LEFT, y, line, ha="left", va="top", fontsize=20, weight="bold")
                    y -= 0.36
                y -= 0.18
                continue

            if kind == "h2":
                lines = wrap_text(content, 62)
                needed = 0.26 * len(lines) + 0.18
                fig, ax, y, page_number = ensure_space(pdf, fig, ax, y, needed, page_number)
                y -= 0.07
                for line in lines:
                    ax.text(LEFT, y, line, ha="left", va="top", fontsize=13.5, weight="bold")
                    y -= 0.25
                y -= 0.05
                continue

            if kind == "h3":
                lines = wrap_text(content, 68)
                needed = 0.22 * len(lines) + 0.12
                fig, ax, y, page_number = ensure_space(pdf, fig, ax, y, needed, page_number)
                for line in lines:
                    ax.text(LEFT, y, line, ha="left", va="top", fontsize=11.5, weight="bold")
                    y -= 0.22
                continue

            if kind == "bullet":
                lines = wrap_text(content, 88)
                needed = LINE_HEIGHT * len(lines) + 0.04
                fig, ax, y, page_number = ensure_space(pdf, fig, ax, y, needed, page_number)
                ax.text(LEFT + 0.1, y, "-", ha="left", va="top", fontsize=9.5)
                for index, line in enumerate(lines):
                    ax.text(LEFT + 0.28, y, line, ha="left", va="top", fontsize=9.5)
                    y -= LINE_HEIGHT if index < len(lines) - 1 else LINE_HEIGHT * 0.95
                continue

            if kind == "code":
                code_lines = content.splitlines()
                needed = MONO_LINE_HEIGHT * len(code_lines) + 0.1
                fig, ax, y, page_number = ensure_space(pdf, fig, ax, y, needed, page_number)
                for line in code_lines:
                    wrapped = textwrap.wrap(line, width=82, break_long_words=False) or [""]
                    for wrapped_line in wrapped:
                        ax.text(LEFT + 0.18, y, wrapped_line, ha="left", va="top", fontsize=8.2, family="monospace")
                        y -= MONO_LINE_HEIGHT
                y -= 0.04
                continue

            lines = wrap_text(content, 94)
            needed = LINE_HEIGHT * len(lines) + 0.06
            fig, ax, y, page_number = ensure_space(pdf, fig, ax, y, needed, page_number)
            for line in lines:
                ax.text(LEFT, y, line, ha="left", va="top", fontsize=9.7)
                y -= LINE_HEIGHT
            y -= 0.03

        finish_page(pdf, fig)


def main() -> None:
    blocks = parse_markdown(INPUT_PATH)
    render_pdf(blocks, OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
