#!/usr/bin/env python3
import html
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import NamedTuple, Optional
from urllib.parse import quote

IMG_RE = re.compile(r"!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
WIKI_RE = re.compile(r"\[\[([^\]]+)\]\]")
HIGHLIGHT_RE = re.compile(r"(?<![=])==([^=\n]+?)==(?!=)")
PUNCT_RE = re.compile(r"[\\/:*?\"<>|\[\]{}()!@#$%^&+=,.;'`~]")
LEGACY_DATE = "2026-01-10"
SITE_URL = "https://edward-lyz.github.io/"


class ArticleInfo(NamedTuple):
    title: str
    rel_path: Path
    date_value: str


def anchorize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = PUNCT_RE.sub("", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def convert_obsidian(content: str) -> str:
    def escape_inline_angle_brackets(line: str) -> str:
        stripped = line.lstrip()
        if stripped.startswith("<"):
            return line
        return re.sub(
            r"<[^>\n]+>",
            lambda match: html.escape(match.group(0)),
            line,
        )

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
        line = escape_inline_angle_brackets(line)
        out.append(line)

    return "".join(out)


def needs_front_matter(content: str) -> bool:
    if content.startswith("\ufeff"):
        content = content.lstrip("\ufeff")
    return not (content.startswith("---") or content.startswith("+++")
                or content.startswith("{"))


def add_front_matter(content: str, title: str, date_value: str) -> str:
    safe_title = title.replace("\"", "\\\"")
    front_matter = (
        "---\n"
        f"title: \"{safe_title}\"\n"
        f"date: {date_value}\n"
        f"lastmod: {date_value}\n"
        "---\n\n"
    )
    return front_matter + content.lstrip("\ufeff")


def load_legacy_list(path: Path) -> set:
    if not path.exists():
        return set()
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line}


def git_commit_date(repo_root: Path, file_path: Path) -> Optional[str]:
    rel_path = file_path.relative_to(repo_root)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--format=%cs", "--", str(rel_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def article_href(rel_path: Path) -> str:
    route = (Path("文章") / rel_path.with_suffix("")).as_posix()
    return "/" + quote(route, safe="/") + ".html"


def write_home_page(dest_dir: Path, articles: list[ArticleInfo]) -> None:
    latest_articles = sorted(
        articles,
        key=lambda article: (article.date_value, article.title),
        reverse=True,
    )[:10]

    lines = [
        "---",
        "title: lyz的博客",
        "aside: false",
        "---",
        "",
        '<div class="home-intro">',
        '<p class="home-kicker">lyz的博客</p>',
        '<h1 class="home-title">AI Infra 与工程笔记</h1>',
        '<p class="home-lead">无限进步。这里整理推理引擎、CUDA 算子、系统工程和 AI 时代的思考，重点保留可复用的工程经验和长文笔记。</p>',
        '<p class="home-links"><a href="/文章/">查看全部文章</a> · <a href="https://github.com/Edward-lyz">GitHub</a></p>',
        '</div>',
        "",
        "## 最新文章",
        "",
        '<ul class="article-list">',
    ]

    for article in latest_articles:
        title = html.escape(article.title)
        href = html.escape(article_href(article.rel_path), quote=True)
        lines.append(
            f'  <li><a href="{href}">{title}</a><span class="article-date">{article.date_value}</span></li>'
        )

    lines.extend(
        [
            '</ul>',
            '',
            f'<p class="home-footnote">当前公开整理 {len(articles)} 篇文章。更多内容按时间持续清理和归档。</p>',
            '',
        ]
    )
    (dest_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def write_article_index(article_dir: Path, articles: list[ArticleInfo]) -> None:
    lines = [
        "---",
        "title: 文章",
        "aside: false",
        "---",
        "",
        "# 文章",
        "",
        "这里收集 AI Infra、推理引擎、CUDA 算子、工程方法论和随笔。",
        "",
        '<ul class="article-list">',
    ]
    for article in sorted(
        articles,
        key=lambda info: (info.date_value, info.title),
        reverse=True,
    ):
        title = html.escape(article.title)
        href = html.escape(article_href(article.rel_path), quote=True)
        lines.append(
            f'  <li><a href="{href}">{title}</a><span class="article-date">{article.date_value}</span></li>'
        )
    lines.extend(["</ul>", ""])
    (article_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def write_public_assets(repo_root: Path, dest_dir: Path) -> None:
    resources_dir = repo_root / "资源"
    if not resources_dir.exists():
        raise SystemExit("Resource folder '资源' not found.")

    shutil.copytree(resources_dir, dest_dir / "资源", dirs_exist_ok=True)

    public_dir = dest_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(resources_dir, public_dir / "资源", dirs_exist_ok=True)

    for child in resources_dir.iterdir():
        target = public_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)

    (public_dir / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}sitemap.xml\n",
        encoding="utf-8",
    )
    (public_dir / ".nojekyll").write_text("", encoding="utf-8")
    (public_dir / "logo.svg").write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" role="img" aria-label="lyz"><rect width="96" height="96" rx="24" fill="#f3ede6"/><path fill="#1f1a17" d="M23 67V25h9v34h18v8H23Zm31 0V25h9v42h-9Zm15 0v-8l19-26H70v-8h29v8L80 59h19v8H69Z" opacity="0.96"/></svg>\n""",
        encoding="utf-8",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "文章"
    dest_dir = repo_root / ".github" / "build-content"
    article_dir = dest_dir / "文章"

    if not src_dir.exists():
        raise SystemExit("Source folder '文章' not found.")

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    article_dir.mkdir(parents=True, exist_ok=True)

    legacy_list = load_legacy_list(repo_root / ".github" / "legacy_notes.txt")
    articles: list[ArticleInfo] = []

    for path in sorted(src_dir.rglob("*")):
        rel = path.relative_to(src_dir)
        if path.is_dir():
            (article_dir / rel).mkdir(parents=True, exist_ok=True)
            continue
        if rel.name == "_index.md":
            continue

        target = article_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            converted = convert_obsidian(text)
            rel_key = rel.as_posix()
            if rel_key in legacy_list:
                date_value = LEGACY_DATE
            else:
                date_value = git_commit_date(repo_root, path) or date.today().isoformat()
            if needs_front_matter(converted):
                converted = add_front_matter(converted, path.stem, date_value)
            target.write_text(converted, encoding="utf-8")
            articles.append(ArticleInfo(path.stem, rel, date_value))
        else:
            shutil.copy2(path, target)

    write_home_page(dest_dir, articles)
    write_article_index(article_dir, articles)
    write_public_assets(repo_root, dest_dir)


if __name__ == "__main__":
    main()
