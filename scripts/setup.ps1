# scripts\setup.ps1
$ErrorActionPreference = "Stop"

# Go to repo root
Set-Location -Path (Split-Path -Parent $PSCommandPath)
Set-Location ..

# Create venv if missing
if (!(Test-Path ".venv")) {
    python -m venv .venv
}

# Activate venv
. .\.venv\Scripts\Activate.ps1

# Upgrade tooling
python -m pip install --upgrade pip setuptools wheel

# Install the project in editable mode (reads dependencies from pyproject.toml)
pip install -e .

Write-Host "Setup complete. Run 'dlab' to launch (or use your VS Code launch)."
