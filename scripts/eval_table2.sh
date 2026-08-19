#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/unified31_table2_epoch360.pth" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

python eval.py \
    --checkpoint "$1" \
    --ntu-num-classes 60 \
    --datasets ntu3d ntu2d humanml3d \
    --device "cuda:${HOV_DEVICE:-0}" \
    --workers 16 \
    --json-output work_dirs/table2_eval.json
