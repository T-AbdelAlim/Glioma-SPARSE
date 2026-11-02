# One-time setup on a new machine. Creates .venv with the exact package versions
# from requirements-lock.txt, installs glioma_sparse, then checks which files that
# live outside git still need to be copied over. From the repo root:
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-Python314 {
    foreach ($cand in @(@("py", "-3.14"), @("python"), @("python3.14"))) {
        try {
            $exe = $cand[0]; $pre = @($cand | Select-Object -Skip 1)
            $out = & $exe @pre -c "import sys; print(sys.version_info[:2] == (3, 14))" 2>$null
            if ($out -eq "True") { return ,$cand }
        } catch {}
    }
    return $null
}

function Invoke-Step($what, [scriptblock]$cmd) {
    Write-Host "`n== $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    $py = Get-Python314
    if (-not $py) {
        Write-Host "Python 3.14 not found. Install it, then rerun this script:" -ForegroundColor Yellow
        Write-Host "  winget install -e --id Python.Python.3.14"
        exit 1
    }
    $exe = $py[0]; $pre = @($py | Select-Object -Skip 1)
    Invoke-Step "Creating .venv" { & $exe @pre -m venv .venv }
}

Invoke-Step "Upgrading pip" { & $venvPy -m pip install --upgrade pip }
Invoke-Step "Installing pinned packages (PyTorch alone is ~2.5 GB)" { & $venvPy -m pip install -r requirements-lock.txt }
Invoke-Step "Installing glioma_sparse (editable)" { & $venvPy -m pip install -e . --no-deps }

Write-Host ""
& $venvPy -m scripts.check_setup
