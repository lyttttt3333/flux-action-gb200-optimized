#!/bin/bash
set -euo pipefail
root="${FLUX5090_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
output="${1:?supply a new result directory name}"
exec bash "$root/run_selected_lossless_5090.sh" "$output" all
