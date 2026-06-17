#!/usr/bin/env bash
set -euo pipefail
exec > >(tee frontend/dist/meetily-portable/docker-build.log) 2>&1
export PATH="/root/.cargo/bin:$PATH"
mkdir -p frontend/src-tauri/binaries frontend/dist/meetily-portable
cargo xwin build --release -p llama-helper --target x86_64-pc-windows-msvc
cp -f target/x86_64-pc-windows-msvc/release/llama-helper.exe frontend/src-tauri/binaries/llama-helper-x86_64-pc-windows-msvc.exe
cd frontend
pnpm install --frozen-lockfile || pnpm install
pnpm tauri build -- --runner cargo-xwin --target x86_64-pc-windows-msvc --bundles nsis
cd ..
profile=target/x86_64-pc-windows-msvc/release
host_out=frontend/dist/meetily-portable
if [ -d "$profile/bundle/nsis" ]; then cp -av "$profile/bundle/nsis/"* "$host_out/" || true; fi
if [ -f "$profile/meetily.exe" ]; then cp -av "$profile/meetily.exe" "$host_out/meetily.exe"; fi
