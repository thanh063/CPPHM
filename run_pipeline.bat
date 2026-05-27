@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

set PYTHONUTF8=1
set PAGES=789
set DELAY=2
set MIN_ROWS=500
set SYNTHETIC_ROWS=10000

echo.
echo ============================================================
echo   PIPELINE: Thu thap - Lam sach - Train - Bao cao
echo ============================================================
echo.

:: ── Buoc 1: Cai thu vien ─────────────────────────────────────
echo.
echo [1/5] Cai dat thu vien...
python -m pip install -r requirements.txt
if errorlevel 1 ( echo [LOI] Cai thu vien that bai & exit /b 1 )

:: ── Buoc 2: Thu thap du lieu thuc te ─────────────────────────────
echo.
echo [Buoc 2] Thu thap du lieu tu 5 nguon...
python crawler.py --pages %PAGES% --categories nha-dat can-ho biet-thu dat-nen nha-pho --sources mogi homedy alonhadat batdongsan nhatot --output vietnam_house_raw.csv --delay %DELAY% --detail --fresh

:: ── Buoc 3: Lam sach du lieu ──────────────────────────────────
echo.
echo [Buoc 3] Lam sach du lieu (loai outlier, loi don vi, trung lap)...
python clean_data.py --input vietnam_house_raw.csv --output vietnam_house_clean.csv
if errorlevel 1 (
    echo [WARN] Lam sach that bai - dung du lieu raw thay the
    copy vietnam_house_raw.csv vietnam_house_clean.csv
)

:: ── Buoc 3: Kiem tra so dong, fallback neu qua it ───────────
echo.
echo [3/5] Kiem tra chat luong du lieu...
for /f %%i in ('python -c "import pandas as pd; print(len(pd.read_csv(\"vietnam_house_raw.csv\", encoding=\"utf-8-sig\")))"') do set ROW_COUNT=%%i

echo       So ban ghi thu thap duoc: !ROW_COUNT!

if !ROW_COUNT! LSS %MIN_ROWS% (
    echo       [CANH BAO] Chi co !ROW_COUNT! ban ghi, qua it ^(min %MIN_ROWS%^).
    echo       [FALLBACK] Dung du lieu tong hop %SYNTHETIC_ROWS% ban ghi...
    python generate_dataset.py --rows %SYNTHETIC_ROWS% --output vietnam_house_raw.csv
)

:: ── Buoc 4: Train model ──────────────────────────────────────
echo.
echo [4/5] Huan luyen mo hinh Bagging...
python bagging_train.py ^
    --data vietnam_house_clean.csv ^
    --output house_bagging_model.joblib ^
    --n-estimators 200 ^
    --max-depth 20 ^
    --test-size 0.05 ^
    --compare ^
    --plot ^
    --report evaluation_report.json
if errorlevel 1 ( echo [LOI] Train that bai & exit /b 1 )

:: ── Buoc 5: Du doan mau ─────────────────────────────────────
echo.
echo [5/5] Du doan mau (nha pho 80m2, Quan 7, 3PN, 2WC, 4 tang)...
python predict.py ^
    --model house_bagging_model.joblib ^
    --area 80 ^
    --district q7 ^
    --bedrooms 3 ^
    --bathrooms 2 ^
    --floors 4 ^
    --house-type "nha pho"

echo.
echo ============================================================
echo   HOAN THANH!
echo   Model : house_bagging_model.joblib
echo   Report: evaluation_report.json
echo   Bieu do: plots/
echo   Chay web: python app.py
echo ============================================================