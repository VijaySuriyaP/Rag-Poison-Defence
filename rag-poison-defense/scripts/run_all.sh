#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
resume_flag=""
for arg in "$@"; do
  case "$arg" in
    --resume) resume_flag="--resume" ;;
    -h|--help)
      printf 'Usage: %s [--resume]\nRuns data, attacks, baseline, defense ablations, analysis, figures, and report.\n' "$0"
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
make all RESUME_FLAG="$resume_flag"
