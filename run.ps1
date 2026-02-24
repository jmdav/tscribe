Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    Write-Host "python not found. Please install Python 3.10+ and try again."
    exit 1
}

$VenvDir = Join-Path $RootDir ".venv"
if (-not (Test-Path $VenvDir)) {
    & $Python.Source -m venv $VenvDir
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Host "Venv activation script not found: $ActivateScript"
    exit 1
}

. $ActivateScript

python -m pip install --upgrade pip

$Requirements = Join-Path $RootDir "requirements.txt"
if (Test-Path $Requirements) {
    python -m pip install -r $Requirements
} else {
    python -m pip install streamlit faster-whisper torch requests
}

python -m streamlit run app.py
