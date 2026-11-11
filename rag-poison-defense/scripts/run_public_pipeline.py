"""Cross-platform runner for public-data preparation and full project evaluation."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikipedia-size", type=int, default=2000)
    parser.add_argument("--per-attack-source", type=int, default=600)
    parser.add_argument("--squad-qa-rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    commands = [
        [sys.executable, "-m", "ragdef.data.prepare_public", "--size", str(args.wikipedia_size),
         "--per-attack-source", str(args.per_attack_source), "--squad-qa-rows", str(args.squad_qa_rows),
         "--seed", str(args.seed)],
        [sys.executable, "-m", "ragdef.eval.public_benchmark"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
