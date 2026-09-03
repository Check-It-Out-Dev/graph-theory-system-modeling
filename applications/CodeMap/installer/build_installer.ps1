# CodeMap - build the installer payload, then compile setup.exe (Inno Setup 6).
# The geodoc pattern (tools/build_installer.ps1 there), simplified by the app's
# zero-dependency law: the ONLY vendored package is real-ladybug (the embedded
# graph DB the cypher() tier queries); everything else is stdlib.
#
# Produces: installer\python-embed  (PSF embeddable 3.12 + real-ladybug)
#           installer\llama-bin     (llama.cpp b10155 CPU, runtime dispatch)
#           installer\Output\codemap-setup-<ver>.exe - ONE file; the 2.5 GB
#           model is downloaded by the wizard (SHA-256) or read side-by-side
#
# Usage: powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$inst = $PSScriptRoot
$py   = Join-Path $inst "python-embed"
$pyver = "3.12.8"

Write-Host "[1/4] python embeddable $pyver"
if (-not (Test-Path (Join-Path $py "python.exe"))) {
    $zip = Join-Path $env:TEMP "python-embed-$pyver.zip"
    if (-not (Test-Path $zip)) {
        Invoke-WebRequest "https://www.python.org/ftp/python/$pyver/python-$pyver-embed-amd64.zip" -OutFile $zip
    }
    Expand-Archive $zip -DestinationPath $py -Force
    $pth = Get-ChildItem $py -Filter "python3*._pth" | Select-Object -First 1
    (Get-Content $pth.FullName) -replace '#import site', 'import site' |
        Set-Content $pth.FullName
}

Write-Host "[2/4] vendor real-ladybug (the one non-stdlib runtime dep)"
$site = Join-Path $py "Lib\site-packages"
New-Item -ItemType Directory -Force $site | Out-Null
if (-not (Test-Path (Join-Path $site "real_ladybug"))) {
    python -m pip install "real-ladybug==0.15.3" --target $site --quiet
}
& (Join-Path $py "python.exe") -c "import real_ladybug; print('real-ladybug', real_ladybug.__version__, 'OK in embed')"

Write-Host "[3/4] llama.cpp binaries (b10155, CPU runtime dispatch)"
$lb = Join-Path $inst "llama-bin"
New-Item -ItemType Directory -Force $lb | Out-Null
Copy-Item (Join-Path $root "bin\llama\b10155\*") $lb -Force

Write-Host "[4/4] compile setup.exe (Inno Setup 6)"
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $p = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    if (Test-Path $p) { $iscc = $p }
}
if ($iscc) {
    & $iscc (Join-Path $inst "codemap.iss")
    Write-Host "DONE: installer\Output\codemap-setup-*.exe (single file)"
    Write-Host "(unsigned - Certum signing is the owner's manual step, as in geodoc)"
} else {
    Write-Host "iscc not found - install Inno Setup 6 to compile setup.exe"
}
