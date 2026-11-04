# VisionGuard web app on Windows (Tasks 4, 5, 8): Flask server, debug OFF, SQLite history,
# inference by the cloud model API.
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -ApiUrl https://<ip>.sslip.io -ApiKey <key>
param(
  [Parameter(Mandatory = $true)][string]$ApiUrl,
  [Parameter(Mandatory = $true)][string]$ApiKey,
  [int]$Port = 5000
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) {
  Write-Host "== creating virtual environment .venv" -ForegroundColor Cyan
  py -3 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install -q --disable-pip-version-check -r requirements-serve.txt

# 12-factor config: everything from environment variables, nothing hard-coded
$env:INFERENCE_MODE = "remote"
$env:MODEL_API_URL  = $ApiUrl
$env:MODEL_API_KEY  = $ApiKey
$env:DATABASE_URL   = "sqlite:///visionguard.db"
$env:SECRET_KEY     = [guid]::NewGuid().ToString()
$env:FLASK_DEBUG    = "0"
$env:PORT           = "$Port"

Write-Host "== VisionGuard on http://127.0.0.1:$Port  (inference: $ApiUrl)  user: $env:USERNAME  $(Get-Date)" -ForegroundColor Green
& .\.venv\Scripts\python.exe -m app.wsgi
