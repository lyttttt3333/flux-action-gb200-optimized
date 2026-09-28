#!/bin/bash
set -euo pipefail
root="${FLUX5090_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
output="${1:?supply a new result directory name}"
mode="${2:?supply the validated composition}"
bash "$root/run_lossless_task.sh" benchmark_lossless_selected.py "$output" "$mode"
"$root/venv/bin/python" "$root/compare_5090.py" "$root" --candidate "$output"
