$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSCommandPath)
Set-Location ..
if (!(Test-Path ".venv")) {
    python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .
Write-Host "Setup complete."
