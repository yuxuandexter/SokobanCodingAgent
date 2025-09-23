#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_pytest_script.sh [pytest-args]
# Examples:
#   bash run_pytest_script.sh
#   bash run_pytest_script.sh tests/serving/api_providers_test.py -vv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p cache

# Load env vars from .env (and optional overrides) for this run
# Note: this affects only this script's process; your shell env is unchanged
if [[ -f "$ROOT_DIR/env_setup.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT_DIR/env_setup.sh" >/dev/null 2>&1 || true
fi

ts="$(date +%Y%m%d_%H%M%S)"
log_file="cache/pytest_stream_${ts}.txt"

echo "[run_pytest] Writing streaming output to $log_file"

# Ensure pytest's exit code is preserved with tee
set -o pipefail
pytest -q -s "$@" 2>&1 | tee "$log_file"
exit_code=$?

echo "[run_pytest] Done with exit code $exit_code"
exit "$exit_code"

