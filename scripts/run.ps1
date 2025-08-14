$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSCommandPath); Set-Location ..
.\.venv\Scripts\Activate.ps1
python -m dlab
