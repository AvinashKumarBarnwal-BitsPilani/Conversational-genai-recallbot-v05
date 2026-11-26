"""Insert or refresh a clickable table of contents in Markdown files.

Anchors follow GitHub's heading-slug rules (lower-case, punctuation removed,
spaces to hyphens, -1/-2 suffixes for duplicates). Lines inside fenced code
blocks are ignored, so a "# comment" in a PowerShell snippet is not a heading.
The TOC lives between <!-- TOC --> and <!-- /TOC --> markers; re-running the
script replaces it in place.

    python scripts/build_docs_toc.py                      # README + task evidence docs
    python scripts/build_docs_toc.py README.md --levels 1 2
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- TOC -->", "<!-- /TOC -->"
DEFAULT_FILES = {
    "README.md": (1, 2),
    "docs/task-evidence/Task-1-Test-Data-and-Probes.md": (2, 3),
    "docs/task-evidence/Task-2-Dialogue-Pipeline-and-Short-Term-Memory.md": (2, 3),
    "docs/task-evidence/Task-3-Retrieval-and-Long-Term-Memory.md": (2, 3),
    "docs/task-evidence/Task-4-Evaluation-and-Error-Analysis.md": (2, 3),
    "docs/task-evidence/Task-5-Cost-and-Scale.md": (2, 3),
}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def plain_text(heading: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", heading)   # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)       # links
    text = re.sub(r"<[^>]+>", "", text)                        # inline HTML
    text = text.replace("`", "")
    text = re.sub(r"(\*\*|__|\*)", "", text)                   # emphasis markers
    return text.strip()


def slug(text: str) -> str:
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def headings(lines: list[str]) -> list[tuple[int, str, str]]:
    found, seen, in_fence, in_toc = [], {}, False, False
    for line in lines:
        if line.strip() == START:
            in_toc = True
        if line.strip() == END:
            in_toc = False
            continue
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or in_toc:
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        text = plain_text(match.group(2))
        base = slug(text)
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.append((len(match.group(1)), text, base if count == 0 else f"{base}-{count}"))
    return found


def build_toc(lines: list[str], levels: tuple[int, int]) -> list[str]:
    low, high = levels
    entries = [h for h in headings(lines) if low <= h[0] <= high]
    if low == 1:
        entries = entries[1:]  # the document title is not listed
    out = [START, "## Table of contents", ""]
    seen_top = False
    for level, text, anchor in entries:
        seen_top = seen_top or level == low
        # Sub-sections that come before the first top-level entry have no parent: list them flush.
        depth = level - low if seen_top else max(0, level - low - 1)
        out.append(f"{'  ' * depth}- [{text}](#{anchor})")
    out += ["", END]
    return out


def apply(path: Path, levels: tuple[int, int]) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if START in lines:
        start, end = lines.index(START), lines.index(END)
        lines = lines[:start] + lines[end + 1:]
        while start < len(lines) and not lines[start].strip():
            del lines[start]  # drop the spacer from the previous run so reruns are stable
        insert_at = start
    else:
        # Before the first heading after the title (keeps the title, intro and figure on top).
        first = [i for i, line in enumerate(lines) if HEADING_RE.match(line) and not line.startswith("# ")]
        in_fence, insert_at = False, len(lines)
        for i, line in enumerate(lines):
            if FENCE_RE.match(line):
                in_fence = not in_fence
            if not in_fence and i in first:
                insert_at = i
                break
    toc = build_toc(lines, levels)
    lines = lines[:insert_at] + toc + [""] + lines[insert_at:]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sum(1 for line in toc if line.lstrip().startswith("- ["))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--levels", nargs=2, type=int, default=None)
    args = parser.parse_args()
    targets = ({f: tuple(args.levels) if args.levels else DEFAULT_FILES.get(f, (2, 3)) for f in args.files}
               if args.files else DEFAULT_FILES)
    for name, levels in targets.items():
        print(f"{name}: {apply(ROOT / name, levels)} entries (levels {levels[0]}-{levels[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
