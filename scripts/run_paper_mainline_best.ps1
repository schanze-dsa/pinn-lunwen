param(
    [string]$Python = "python",
    [string]$Config = "configs\paper_mainline_best.yaml"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (Test-Path $Python) {
    $PythonExe = (Resolve-Path $Python).Path
} else {
    $cmd = Get-Command $Python -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Python executable not found: $Python"
    }
    $PythonExe = $cmd.Source
}
if (-not (Test-Path $Config)) {
    throw "Config not found: $Config"
}

& $PythonExe main_new.py --config $Config
exit $LASTEXITCODE
