import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from bagging_train import load_and_clean, train


def test_load_and_clean_filters_and_builds_features(tmp_path):
    csv_path = tmp_path / "raw.csv"
import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import bagging_train as bt
from generate_dataset import generate


def test_parse_price_mixed():
    assert bt.parse_price("2 tỷ 500 triệu") == 2500.0


def test_parse_price_ty_only():
    assert bt.parse_price("1.5 tỷ") == 1500.0


def test_parse_price_trieu_only():
    assert bt.parse_price("500 triệu") == 500.0


def test_load_clean_removes_dirty_rows(tmp_path):
    csv_path = tmp_path / "dirty.csv"

    pd.DataFrame(
        [
            {
                "price": "2 tỷ 500 triệu",
                "area": "100 m2",
                "location": "Quan 7, TPHCM",
                "bedrooms": "3",
                "bathrooms": "2",
                "floors": "4",
                "house_type": "nha pho",
            },
            {
                "price": "60 tỷ",
                "area": "150 m2",
                "location": "Quan 1, TPHCM",
                "bedrooms": "4",
                "bathrooms": "3",
                "floors": "5",
                "house_type": "biet thu",
            },
            {
                "price": "1 tỷ",
                "area": "80 m2",
                "location": "Quan 3, TPHCM",
                "bedrooms": "11",
                "bathrooms": "2",
                "floors": "2",
                "house_type": "can ho",
            },
            {
                "price": "500 triệu",
                "area": "60 m2",
                "location": "Quan 5, TPHCM",
                "bedrooms": "2",
                "bathrooms": "1",
                "floors": "1",
                "house_type": "can ho",
            },
        ]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    cleaned = bt.load_and_clean(csv_path)

    assert len(cleaned) == 2
    assert cleaned["price_million"].tolist() == [2500.0, 500.0]
    assert cleaned["bedrooms_n"].max() <= 10
    assert (cleaned["price_million"] <= 50000).all()
    assert "district" in cleaned.columns


def test_no_data_leakage(tmp_path, monkeypatch):
    data_path = tmp_path / "train_data.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    generate(80, data_path, seed=123)
    cleaned_len = len(bt.load_and_clean(data_path))

    seen_lengths = []

    def fake_cross_validate(pipe, X, y, cv, scoring, n_jobs):
        seen_lengths.append(len(X))
        assert len(X) < cleaned_len
        return {
            "test_mae": pd.Series([10.0, 11.0, 12.0, 13.0, 14.0]).to_numpy(),
            "test_r2": pd.Series([0.7, 0.72, 0.71, 0.73, 0.74]).to_numpy(),
        }

    monkeypatch.setattr(bt, "cross_validate", fake_cross_validate)

    args = argparse.Namespace(
        data=str(data_path),
        output=str(model_path),
        test_size=0.2,
        n_estimators=8,
        max_depth=8,
        compare=True,
        plot=False,
        plot_dir=str(tmp_path / "plots"),
        report=str(report_path),
    )

    bt.train(args)

    assert seen_lengths
    assert len(set(seen_lengths)) == 1
    assert seen_lengths[0] < cleaned_len


def test_pipeline_r2(tmp_path):
    data_path = tmp_path / "generated.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    generate(320, data_path, seed=42)

    args = argparse.Namespace(
        data=str(data_path),
        output=str(model_path),
        test_size=0.2,
        n_estimators=50,
        max_depth=15,
        compare=False,
        plot=False,
        plot_dir=str(tmp_path / "plots"),
        report=str(report_path),
    )

    bt.train(args)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["test_r2"] > 0.7
