"""Print the first unchecked project step for safe session resumption."""
from __future__ import annotations

import re
from pathlib import Path


def first_unchecked(progress_path: Path = Path("PROGRESS.md")) -> tuple[int, str] | None:
    """Return the first unchecked numbered step and its title, if any."""
    if not progress_path.exists():
        raise FileNotFoundError(f"Progress checklist not found: {progress_path}")
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"- \[ \] Step (\d+) — (.+)", line)
        if match:
            return int(match.group(1)), match.group(2)
    return None


def main() -> int:
    """Print resumable project state and return a useful process code."""
    pending = first_unchecked()
    if pending is None:
        print("All numbered steps are checked.")
        return 0
    number, title = pending
    print(f"First unchecked step: {number} — {title}")
    state = Path("STATE.md")
    if state.exists():
        for line in state.read_text(encoding="utf-8").splitlines():
            if line.startswith("- Exact next command:"):
                print(line)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
