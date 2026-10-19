# Reproduce Component 02 on Windows

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path .venv)) {
    py -3 -m venv .venv
}

& .\.venv\Scripts\python -m pip install --upgrade pip
& .\.venv\Scripts\python -m pip install -r requirements-dev.txt
& .\.venv\Scripts\python -m pytest
& .\.venv\Scripts\python scripts\train_benchmark.py --profile synthetic

Write-Host "Synthetic pipeline check finished. This is not an IEEE-CIS result."
Write-Host "For the research benchmark, follow docs\DATASETS.md and then run:"
Write-Host "  .\.venv\Scripts\python scripts\train_benchmark.py --profile dev --dataset ieee"
