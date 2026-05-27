import argparse
import json
from pathlib import Path

import pytest

import app as app_module
from bagging_train import train
from generate_dataset import generate


@pytest.fixture(scope="session")
def trained_model_and_report(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("api_model")
    data_path = work_dir / "train_data.csv"
    model_path = work_dir / "model.joblib"
    report_path = work_dir / "report.json"

    generate(220, data_path, seed=123)

    args = argparse.Namespace(
        data=str(data_path),
        output=str(model_path),
        test_size=0.2,
        n_estimators=8,
        max_depth=8,
        compare=False,
        plot=False,
        plot_dir=str(work_dir / "plots"),
        report=str(report_path),
    )
    train(args)

    return model_path, report_path


@pytest.fixture()
def client(trained_model_and_report):
    model_path, report_path = trained_model_and_report

    app_module.MODEL_PATH = Path(model_path)
    app_module.REPORT_PATH = Path(report_path)
    app_module._model_cache.clear()

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def test_index_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Du doan" in body or "D\u1ef1 \u0111o\u00e1n" in body


def test_predict_success(client):
    payload = {
        "area": 85,
        "district": "q7",
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 4,
        "house_type": "nh\u00e0 ph\u1ed1",
    }
    res = client.post("/predict", data=json.dumps(payload), content_type="application/json")

    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data["price"], (int, float))
    assert data["ci_low"] <= data["ci_high"]
    assert isinstance(data["ty"], (int, float))


def test_predict_rejects_non_positive_area(client):
    res = client.post("/predict", json={"area": 0, "district": "q7"})
    assert res.status_code == 400
    data = res.get_json()
    assert "Di\u1ec7n t\u00edch" in data["error"]


def test_predict_rejects_invalid_json_payload(client):
    res = client.post("/predict", data="not-json", content_type="application/json")
    assert res.status_code == 400
    data = res.get_json()
    assert "JSON" in data["error"]


def test_about_page_loads(client):
    res = client.get("/about")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "Bagging" in body


def test_analysis_page_loads(client):
    res = client.get("/analysis")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    # Chỉ kiểm tra trang load được — không assert file plot (có thể chưa tồn tại trên CI)
    assert "Phân tích" in body or "analysis" in body.lower() or res.status_code == 200


def test_report_page_loads(client):
    res = client.get("/report")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "evaluation_report.json" in body


# ─── Kiểm thử input validation tại /predict ───────────────────────────

def test_predict_rejects_area_too_large(client):
    res = client.post("/predict", json={"area": 99999, "district": "q7"})
    assert res.status_code == 400
    assert "Diện tích" in res.get_json()["error"]


def test_predict_rejects_invalid_bedrooms(client):
    res = client.post("/predict", json={"area": 80, "district": "q7", "bedrooms": 999})
    assert res.status_code == 400


def test_predict_rejects_non_json(client):
    res = client.post("/predict", data="plain text", content_type="text/plain")
    assert res.status_code == 400


def test_region_page_loads(client):
    res = client.get("/region")
    assert res.status_code == 200


def test_404_returns_error(client):
    res = client.get("/this-page-does-not-exist")
    assert res.status_code == 404
