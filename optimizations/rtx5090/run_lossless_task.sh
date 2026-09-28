#!/bin/bash
set -euo pipefail
root="${FLUX5090_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export TORCHINDUCTOR_EMULATE_PRECISION_CASTS=0
export HF_HUB_CACHE="$root/hf-cache" PYTHONPYCACHEPREFIX="$root/cache/pycache"
export TRITON_CACHE_DIR="$root/cache/triton" TORCHINDUCTOR_CACHE_DIR="$root/cache/inductor-optimized"
export PYTHONPATH="$root/reference/src"
export PATH="$root/venv/bin:/usr/local/cuda/bin:$PATH"
exec "$root/venv/bin/python" "$root/$1" "$root" "${@:2}"
