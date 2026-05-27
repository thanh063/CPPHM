param(
  [int]$Pages = 789,
  [double]$Delay = 2.5,
  [int]$MinRows = 50,
  [int]$SyntheticRows = 3000,
  [switch]$SkipInstall,
  [switch]$SkipCrawl,
  [switch]$StartWeb
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
  param([string]$Message)
  Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Resolve-Python {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    return @("py", "-3")
  }
  if (Get-Command python -ErrorAction SilentlyContinue) {
    return @("python")
  }
  throw "Khong tim thay Python trong PATH."
}

function Invoke-Python {
  param(
    [string[]]$PythonCmd,
    [string[]]$PythonArgs
  )
  $displayCmd = ($PythonCmd + $PythonArgs) -join " "
  Write-Host "[RUN] $displayCmd" -ForegroundColor DarkGray

  if ($PythonCmd.Count -gt 1) {
    & $PythonCmd[0] @($PythonCmd[1..($PythonCmd.Count - 1)]) @PythonArgs
  }
  else {
    & $PythonCmd[0] @PythonArgs
  }
}

$projectRoot = $PSScriptRoot
if (-not $projectRoot) {
  $projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

Push-Location $projectRoot
try {
  Write-Host "[INFO] Working directory: $(Get-Location)" -ForegroundColor DarkCyan

  $pythonCmd = Resolve-Python

  if (-not $SkipInstall) {
    Write-Step "Nang cap pip va cai thu vien"
    Invoke-Python -PythonCmd $pythonCmd -PythonArgs @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Python -PythonCmd $pythonCmd -PythonArgs @("-m", "pip", "install", "-r", "requirements.txt")
  }

  if (-not $SkipCrawl) {
    Write-Step "Crawl du lieu bat dong san"
    Invoke-Python -PythonCmd $pythonCmd -PythonArgs @(
      "crawler.py",
      "--pages", "$Pages",
      "--sources", "mogi", "homedy", "alonhadat", "batdongsan", "nhatot",
      "--categories", "nha-dat", "can-ho", "biet-thu", "dat-nen", "nha-pho",
      "--output", "vietnam_house_raw.csv",
      "--delay", "$Delay"
    )
  }

  Write-Step "Kiem tra so dong du lieu crawl"
  $rowCheck = & {
    $code = @'
import pandas as pd
from pathlib import Path
p = Path("vietnam_house_raw.csv")
if not p.exists():
    print(-1)
else:
    try:
        print(len(pd.read_csv(p, encoding="utf-8-sig")))
    except Exception:
        print(-1)
'@
    $tmp = Join-Path $env:TEMP "check_rows_pipeline.py"
    Set-Content -Path $tmp -Value $code -Encoding UTF8
    try {
      if ($pythonCmd.Count -gt 1) {
        & $pythonCmd[0] @($pythonCmd[1..($pythonCmd.Count - 1)]) $tmp
      }
      else {
        & $pythonCmd[0] $tmp
      }
    }
    finally {
      Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
  }

  [int]$rowCount = [int]($rowCheck | Select-Object -Last 1)
  Write-Host "So dong hien tai: $rowCount"

  if ($rowCount -lt $MinRows) {
    Write-Step "Du lieu crawl qua it. Fallback sinh du lieu tong hop"
    Invoke-Python -PythonCmd $pythonCmd -PythonArgs @("generate_dataset.py", "--rows", "$SyntheticRows", "--output", "vietnam_house_raw.csv")
  }

  Write-Step "Lam sach du lieu (loai outlier, loi don vi, trung lap)"
  Invoke-Python -PythonCmd $pythonCmd -PythonArgs @(
    "clean_data.py",
    "--input", "vietnam_house_raw.csv",
    "--output", "vietnam_house_clean.csv"
  )

  Write-Step "Train model Bagging + bao cao + bieu do"
  Invoke-Python -PythonCmd $pythonCmd -PythonArgs @(
    "bagging_train.py",
    "--data", "vietnam_house_clean.csv",
    "--output", "house_bagging_model.joblib",
    "--report", "evaluation_report.json",
    "--plot",
    "--compare",
    "--n-estimators", "200",
    "--max-depth", "15"
  )

  Write-Step "Predict mau bang CLI"
  Invoke-Python -PythonCmd $pythonCmd -PythonArgs @(
    "predict.py",
    "--model", "house_bagging_model.joblib",
    "--area", "80",
    "--district", "q7",
    "--bedrooms", "3",
    "--bathrooms", "2",
    "--floors", "4",
    "--house-type", "nha pho"
  )

  if ($StartWeb) {
    Write-Step "Chay Flask web demo tai http://127.0.0.1:5000"
    $env:FLASK_DEBUG = "0"
    Invoke-Python -PythonCmd $pythonCmd -PythonArgs @("app.py")
  }
  else {
    Write-Step "Pipeline hoan tat"
    Write-Host "Model: house_bagging_model.joblib"
    Write-Host "Report: evaluation_report.json"
    Write-Host "Mo web thu cong bang lenh: python app.py"
  }
}
finally {
  Pop-Location
}
