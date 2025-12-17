#!/usr/bin/env python3
"""
Extract top-level imported modules from a Python package using AST.

- Scans a target directory (default: src/)
- Ignores tests/docs/venv/build folders
- Ignores relative imports (from .foo import bar)
- Filters out stdlib modules (best-effort)
- Writes a newline-separated list to stdout or a file

Usage:
  python src/alastr/backend/tools/extract_imports.py --root src --out requirements.auto.in
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Iterable, Set


EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "env",
    "__pycache__",
    "build", "dist",
    ".tox", ".pytest_cache",
    "site-packages",
    "tests", "test", "docs", "doc",
    "notebooks", "examples",
}

EXCLUDE_MODULE_PREFIXES = {
    # common local-layout names you might not want treated as external deps
    "alastr",
}


def iter_py_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place so os.walk skips them
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def top_level_imports(py_file: Path) -> Set[str]:
    try:
        src = py_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # if some file has a weird encoding, just skip it
        return set()

    try:
        tree = ast.parse(src, filename=str(py_file))
    except SyntaxError:
        # skip files that aren't valid for this interpreter
        return set()

    mods: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                mods.add(name)

        elif isinstance(node, ast.ImportFrom):
            # skip relative imports: from .foo import bar
            if node.level and node.level > 0:
                continue
            if node.module:
                name = node.module.split(".")[0]
                mods.add(name)

    return mods


def is_stdlib(mod: str) -> bool:
    # Python 3.10+ has sys.stdlib_module_names
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is not None:
        return mod in stdlib_names

    # fallback heuristic (not perfect, but okay)
    try:
        __import__(mod)
        m = sys.modules.get(mod)
        if not m or not hasattr(m, "__file__") or m.__file__ is None:
            return True
        p = Path(m.__file__).as_posix()
        return "/site-packages/" not in p and "/dist-packages/" not in p
    except Exception:
        return False


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="src", help="Directory to scan (default: src)")
    ap.add_argument("--out", type=str, default="", help="Output file (default: stdout)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}", file=sys.stderr)
        return 2

    imports: Set[str] = set()
    for py in iter_py_files(root):
        imports |= top_level_imports(py)

    # remove obvious locals
    imports = {m for m in imports if m and m not in EXCLUDE_MODULE_PREFIXES}

    # remove stdlib
    imports = {m for m in imports if not is_stdlib(m)}

    # emit sorted
    lines = sorted(imports)

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        print("\n".join(lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
