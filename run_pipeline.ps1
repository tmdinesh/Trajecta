$ErrorActionPreference = "Stop"

# Define paths relative to the script location
$ExperimentDir = Join-Path $PSScriptRoot "experiment"
$VenvDir = Join-Path $ExperimentDir ".venv"

if (-not (Test-Path -LiteralPath $ExperimentDir)) {
    throw "Target directory '$ExperimentDir' does not exist."
}

# Move working directory to experiment folder once at the start
Set-Location -LiteralPath $ExperimentDir

# Determine python executable inside the virtual environment (Windows vs. Unix/macOS PowerShell)
$VenvPython = if ($IsWindows -or ($env:OS -like "*Windows*")) {
    Join-Path $VenvDir "Scripts\python.exe"
} else {
    Join-Path $VenvDir "bin/python"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Host "`n==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

# 1. Virtual Environment Setup
Invoke-Step "Set up virtual environment" {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host "Virtual environment not found. Creating '.venv' inside experiment directory..."
        python -m venv .venv
    } else {
        Write-Host "Virtual environment detected at $VenvDir"
    }
}

# 2. Dependency Installation
Invoke-Step "Install requirements" {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
}

# 3. Experiment Pipeline Execution
Invoke-Step "Recreate synthetic experiments" { & $VenvPython recreate.py }
Invoke-Step "Download public benchmarks" { & $VenvPython download_benchmarks.py }
Invoke-Step "Run public benchmark experiments" { & $VenvPython run_benchmarks.py }
Invoke-Step "Generate figures and tables" { & $VenvPython generate_artifacts.py }

Write-Host "`nPipeline completed. See experiment/results/figures and experiment/results/tables."