param(
    [string]$Python = "python",
    [string]$Config = "configs\paper_mainline_best.yaml"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path $Config)) {
    throw "Config not found: $Config"
}

& $Python main_new.py --config $Config
exit $LASTEXITCODE
