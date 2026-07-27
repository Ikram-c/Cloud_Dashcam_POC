#!/usr/bin/env python3
"""Repo style gate: line length, tabs, and trailing whitespace.

Checks every Python file under src, scripts, and tests. Exits
non-zero with a per-violation listing, so CI fails loudly and locally
reproducibly.

Usage:
    python scripts/check_style.py
"""

import sys
from pathlib import Path

MAX_COLUMNS = 99
CHECK_DIRS = ("src", "scripts", "tests")


def line_violations(path: Path) -> list:
    """Find over-length, tab, and trailing-whitespace lines.

    Args:
        path (Path): Python file.

    Returns:
        list: (line, reason) pairs.
    """
    found = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if len(line) > MAX_COLUMNS:
            found.append(
                (number, f"line too long ({len(line)} > {MAX_COLUMNS})"),
            )
        if "\t" in line:
            found.append((number, "tab character"))
        if line != line.rstrip():
            found.append((number, "trailing whitespace"))
    return found


def main() -> int:
    """Entry point.

    Returns:
        int: 0 when clean, 1 when violations exist.
    """
    root = Path(__file__).resolve().parent.parent
    violations = 0
    for dirname in CHECK_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root)
            for line, reason in line_violations(path):
                print(f"{rel}:{line}: {reason}")
                violations += 1
    if violations:
        print(f"{violations} style violation(s)")
        return 1
    print("style clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
