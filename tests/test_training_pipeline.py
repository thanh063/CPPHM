import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from bagging_train import load_and_clean, train


def test_load_and_clean_filters_and_builds_features(tmp_path):
    csv_path = tmp_path / "raw.csv"

    df_raw = pd.DataFrame(
        [
            {
                "price": "2 ty 500 trieu",
                "area": "100 m2",
                "location": "Quan 7, TPHCM",
                "bedrooms": "3",
                "bathrooms": "2",
                "floors": "4",
                "house_type": "nha pho",
            },
            {
                "price": "thoa thuan",
                "area": "120 m2",
                "location": "Quan 1, TPHCM",
                "bedrooms": "2",
                "bathrooms": "2",
                "floors": "2",
                "house_type": "can ho",
            },
            {
                "price": "50 trieu",
                "area": "6 m2",
                "location": "Binh Thanh, TPHCM",
                "bedrooms": "1",
                "bathrooms": "1",
                "floors": "1",
                "house_type": "nha hem",
            },
        ]
    )
    df_raw.to_csv(csv_path, index=False, encoding="utf-8-sig")

    cleaned = load_and_clean(csv_path)

    assert len(cleaned) == 1
    assert "district" in cleaned.columns
    assert "price_million" in cleaned.columns
    assert cleaned["price_million"].notna().all()
    assert cleaned["area_m2"].notna().all()


def test_train_creates_model_and_report(tmp_path):
    data_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    rows = []
    for i in range(80):
        rows.append(
            {
                "price": int((1500 + i * 30) * 1_000_000),
                "area": 50 + (i % 20),
                "location": f"Duong {i}, Quan {(i % 12) + 1}, TPHCM",
                "district": f"q{(i % 12) + 1}",
                "bedrooms_n": 2 + (i % 3),
                "bathrooms_n": 1 + (i % 2),
                "floors_n": 1 + (i % 4),
                "house_type": "nha pho" if i % 2 == 0 else "can ho",
            }
        )

    pd.DataFrame(rows).to_csv(data_path, index=False, encoding="utf-8-sig")

    args = argparse.Namespace(
        data=str(data_path),
        output=str(model_path),
        test_size=0.2,
        n_estimators=6,
        max_depth=8,
        compare=False,
        plot=False,
        plot_dir=str(tmp_path / "plots"),
        report=str(report_path),
    )

    train(args)

    assert model_path.exists()
    assert report_path.exists()

    bundle = joblib.load(model_path)
    assert "pipeline" in bundle
    assert "feature_cols" in bundle

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "test_metrics" in report
    assert "oob_r2" in report
    assert report["n_estimators"] == 6
