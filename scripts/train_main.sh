#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

: "${HUMANML3D_ROOT:?Please set HUMANML3D_ROOT.}"
: "${HOV_NTU120_3D:?Please set HOV_NTU120_3D.}"
: "${HOV_NTU120_2D:?Please set HOV_NTU120_2D.}"
: "${HOV_NTU120_2D_MEAN:?Please set HOV_NTU120_2D_MEAN.}"
: "${HOV_NTU120_2D_STD:?Please set HOV_NTU120_2D_STD.}"

python train.py
