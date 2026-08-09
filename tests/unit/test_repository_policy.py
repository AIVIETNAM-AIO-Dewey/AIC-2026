from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_committed_notebooks_are_thin_and_cleared() -> None:
    notebooks = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            assert cell.get("outputs", []) == [], path
            assert cell.get("execution_count") is None, path


def test_cloud_runtime_mirrors_core_dependency_pins() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    required = set(project["project"]["dependencies"])
    runtime_lines = {
        line.strip()
        for line in (REPO_ROOT / "requirements" / "runtime-base.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert required <= runtime_lines


def test_importable_package_has_no_cloud_specific_absolute_paths() -> None:
    forbidden = ("/content", "/kaggle", "drive/MyDrive")
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path


def test_markdown_fences_and_relative_links_are_valid() -> None:
    link_pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    paths = list(REPO_ROOT.glob("*.md"))
    for directory in ("docs", "data", "artifacts", "runs", "reports", "tests", ".github"):
        paths.extend((REPO_ROOT / directory).rglob("*.md"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.count("```") % 2 == 0, path
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (path.parent / unquote(target)).exists(), (path, raw_target)
