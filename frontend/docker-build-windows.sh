#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"
IMAGE="${MEETILY_WIN_DOCKER_IMAGE:-meetily-windows-portable-builder}"
OUT_DIR="${FRONTEND_DIR}/dist/meetily-portable"
mkdir -p "${OUT_DIR}"
echo "==> Building Docker image: ${IMAGE}"
docker build --platform linux/amd64 -f "${FRONTEND_DIR}/Dockerfile.windows-portable" -t "${IMAGE}" "${FRONTEND_DIR}"
echo "==> Running cross-compile container"
docker run --rm --platform linux/amd64 -v "${ROOT_DIR}:/workspace" -w /workspace -e RUST_BACKTRACE=1 -e CARGO_TERM_COLOR=always -e TAURI_GPU_FEATURE=none "${IMAGE}" bash /workspace/frontend/scripts/docker-windows-inner.sh
echo "==> Host output: ${OUT_DIR}"
if compgen -G "${OUT_DIR}/*.exe" > /dev/null || [ -f "${OUT_DIR}/meetily.exe" ]; then echo SUCCESS; exit 0; fi
echo FAILURE: No exe artifacts under ${OUT_DIR}"
exit 1
