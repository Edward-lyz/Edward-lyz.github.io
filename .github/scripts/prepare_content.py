#!/usr/bin/env python3
import re
import shutil
from pathlib import Path
from urllib.parse import quote

IMG_RE = re.compile(r"!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
WIKI_RE = re.compile(r"\[\[([^\]]+)\]\]")
HIGHLIGHT_RE = re.compile(r"(?<![=])==([^=\n]+?)==(?!=)")
PUNCT_RE = re.compile(r"[\\/:*?\"<>|\[\]{}()!@#$%^&+=,.;'`~]")


def anchorize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = PUNCT_RE.sub("", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def convert_obsidian(content: str) -> str:
    def replace_image(match: re.Match) -> str:
        target = match.group(1).strip()
        alt = (match.group(2) or "").strip()
        alt_text = alt or Path(target).stem
        url = "/" + quote(target)
        return f"![{alt_text}]({url})"

    def replace_link(match: re.Match) -> str:
        target = match.group(1).strip()
        if target.startswith("#"):
            label = target[1:].strip() or target
            anchor = anchorize(label)
            return f"[{label}](#{anchor})" if anchor else label
        return target

    lines = content.splitlines(keepends=True)
    out = []
    in_fence = False
    fence_marker = ""

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        line = IMG_RE.sub(replace_image, line)
        line = WIKI_RE.sub(replace_link, line)
        line = HIGHLIGHT_RE.sub(r"\1", line)
        out.append(line)

    return "".join(out)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "学习"
    dest_dir = repo_root / ".github" / "build-content"

    if not src_dir.exists():
        raise SystemExit("Source folder '学习' not found.")

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for path in src_dir.rglob("*"):
        rel = path.relative_to(src_dir)
        target = dest_dir / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            converted = convert_obsidian(text)
            target.write_text(converted, encoding="utf-8")
        else:
            shutil.copy2(path, target)


if __name__ == "__main__":
    main()
