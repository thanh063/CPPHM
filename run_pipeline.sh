#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt

if [ ! -f "vietnam_house_raw.csv" ]; then
  python generate_dataset.py --rows 600 --output vietnam_house_raw.csv
fi

python bagging_train.py --data vietnam_house_raw.csv --output house_bagging_model.joblib --n-estimators 100 --max-depth 15 --plot --compare --report evaluation_report.json
python auto_report.py
python app.py