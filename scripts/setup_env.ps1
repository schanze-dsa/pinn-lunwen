param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv"
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

& $PythonExe -m venv $VenvDir
$VenvPython = Join-Path $Root ($VenvDir + "\Scripts\python.exe")
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python not found: $VenvPython"
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

Write-Host "Environment ready."
Write-Host "Activate with: $VenvDir\\Scripts\\Activate.ps1"
