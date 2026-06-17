$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $Root "..")).Path
Set-Location $Root

$TargetTriple = "x86_64-pc-windows-msvc"
$WorkspaceRelease = Join-Path $RepoRoot "target" "release"
$BinariesDir = Join-Path $Root "src-tauri" "binaries"
$SidecarName = "llama-helper-$TargetTriple.exe"
$SidecarDest = Join-Path $BinariesDir $SidecarName

Write-Host "Building llama-helper sidecar from workspace root..."
Push-Location $RepoRoot
try {
  cargo build --release -p llama-helper
  if ($LASTEXITCODE -ne 0) { throw "llama-helper build failed with exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}

$HelperSrc = Join-Path $WorkspaceRelease "llama-helper.exe"
if (-not (Test-Path $HelperSrc)) {
  throw "llama-helper.exe not found at $HelperSrc"
}

New-Item -ItemType Directory -Force -Path $BinariesDir | Out-Null
Copy-Item -Force $HelperSrc $SidecarDest
Write-Host "Copied llama-helper sidecar to $SidecarDest"

Write-Host "Installing frontend deps and building Tauri..."
pnpm install
if ($env:MEETILY_PORTABLE_GPU -eq "vulkan") {
  pnpm run tauri:build:vulkan
} else {
  pnpm run tauri:build:cpu
}

$OutDir = Join-Path $Root "dist" "meetily-portable"
$ZipPath = Join-Path $Root "dist" "meetily-windows-portable.zip"
$ReadmePath = Join-Path $OutDir "README.txt"

if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$ExeSearchRoots = @(
  $WorkspaceRelease,
  (Join-Path $Root "src-tauri" "target" "release")
)

$Exe = $null
foreach ($dir in $ExeSearchRoots) {
  $candidate = Join-Path $dir "meetily.exe"
  if (Test-Path $candidate) {
    $Exe = $candidate
    break
  }
}

if (-not $Exe) {
  $bundleRoots = @(
    (Join-Path $Root "src-tauri" "target" "release" "bundle"),
    (Join-Path $WorkspaceRelease "bundle")
  )
  foreach ($bundleRoot in $bundleRoots) {
    if (-not (Test-Path $bundleRoot)) { continue }
    $match = Get-ChildItem -Path $bundleRoot -Recurse -Filter "meetily.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($match) {
      $Exe = $match.FullName
      break
    }
  }
}

if (-not $Exe) { throw "meetily.exe not found after build" }
Copy-Item $Exe (Join-Path $OutDir "meetily.exe")
Write-Host "Copied main executable from $Exe"

$ExeDir = Split-Path -Parent $Exe
Get-ChildItem -Path $ExeDir -Filter "*.dll" -File -ErrorAction SilentlyContinue | Where-Object {
  $n = $_.Name.ToLowerInvariant()
  $n -match "sherpa|onnxruntime|kaldi|fst|fbank|kissfft|piper|espeak|ssentencepiece|ucd"
} | ForEach-Object {
  Copy-Item -Force $_.FullName $OutDir
  Write-Host "Copied runtime DLL $($_.Name)"
}

# Copy runtime DLLs
foreach ($rel in @($WorkspaceRelease)) {
  if (-not (Test-Path $rel)) { continue }
  Get-ChildItem -Path $rel -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.Name -match "sherpa|onnxruntime|kaldi|kaldifst|fst|fbank|kissfft|piper|espeak|ssentencepiece|ucd") { Copy-Item $_.FullName $OutDir -Force }
  }
}
# runtime dll copy placeholder

$SidecarCandidates = @(
  (Join-Path $WorkspaceRelease "llama-helper-$TargetTriple.exe"),
  (Join-Path $WorkspaceRelease "ffmpeg-$TargetTriple.exe"),
  (Join-Path $Root "src-tauri" "target" "release" "llama-helper-$TargetTriple.exe"),
  (Join-Path $Root "src-tauri" "target" "release" "ffmpeg-$TargetTriple.exe"),
  (Join-Path $BinariesDir "llama-helper-$TargetTriple.exe"),
  (Join-Path $BinariesDir "ffmpeg-$TargetTriple.exe")
)

foreach ($sidecar in $SidecarCandidates) {
  if (Test-Path $sidecar) {
    Copy-Item $sidecar $OutDir
    Write-Host "Copied sidecar $(Split-Path -Leaf $sidecar)"
  }
}

$ResourceCandidates = @(
  (Join-Path $Root "src-tauri" "target" "release" "bundle" "resources"),
  (Join-Path $WorkspaceRelease "bundle" "resources")
)

foreach ($resources in $ResourceCandidates) {
  if (Test-Path $resources) {
    Copy-Item -Recurse $resources (Join-Path $OutDir "resources")
    Write-Host "Copied bundle resources from $resources"
    break
  }
}

@'
Meetily Windows Portable Build
==============================

1. Extract this folder anywhere on your PC.
2. Run meetily.exe.
3. Keep sidecar binaries (llama-helper-*.exe, ffmpeg-*.exe) in the same folder as meetily.exe.

GPU builds: set MEETILY_PORTABLE_GPU=vulkan before running build-portable-windows.bat
'@ | Set-Content -Path $ReadmePath -Encoding UTF8

if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path (Join-Path $OutDir "*") -DestinationPath $ZipPath
Write-Host "Portable package: $ZipPath"
