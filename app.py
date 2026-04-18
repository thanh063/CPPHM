#!/usr/bin/env python3
"""
Flask web demo – Dự đoán giá nhà Việt Nam bằng Bagging.

Chạy:
    python app.py
Sau đó mở: http://127.0.0.1:5000
"""

import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("house_bagging_model.joblib")
REPORT_PATH = Path("evaluation_report.json")
PLOTS_DIR   = Path("plots")
DATA_PATHS  = [Path("vietnam_house_clean.csv"), Path("vietnam_house_raw.csv")]

# Cache model
_model_cache = {}


def get_model():
    if "model" not in _model_cache:
        if not MODEL_PATH.exists():
            return None
        _model_cache["model"] = joblib.load(MODEL_PATH)
    return _model_cache["model"]


def get_report():
    if not REPORT_PATH.exists():
        return {}
    with open(REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)

def _normalize_city(raw: str) -> str:
    if not isinstance(raw, str):
        return "Khac"
    t = raw.strip().lower()
    if not t:
        return "Khac"

    # Common aliases first.
    alias_map = {
        "tphcm": "TP. Ho Chi Minh",
        "tp hcm": "TP. Ho Chi Minh",
        "tp.hcm": "TP. Ho Chi Minh",
        "ho chi minh": "TP. Ho Chi Minh",
        "ha noi": "Ha Noi",
        "hanoi": "Ha Noi",
        "da nang": "Da Nang",
        "danang": "Da Nang",
        "can tho": "Can Tho",
        "hai phong": "Hai Phong",
    }
    compact = re.sub(r"[^a-z0-9]+", " ", t).strip()
    if compact in alias_map:
        return alias_map[compact]

    compact = compact.replace("thanh pho ", "").replace("tp ", "")
    if compact in alias_map:
        return alias_map[compact]

    return " ".join(w.capitalize() for w in compact.split()) if compact else "Khac"

def _extract_district_slug(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    lo = text.lower()

    m = re.search(r"(quan|q\.?)\s*([0-9]+)", lo)
    if m:
        return "q" + m.group(2)

    named = {
        "binh thanh": "binh_thanh",
        "phu nhuan": "phu_nhuan",
        "tan binh": "tan_binh",
        "tan phu": "tan_phu",
        "binh tan": "binh_tan",
        "go vap": "go_vap",
        "thu duc": "thu_duc",
        "nha be": "nha_be",
        "hoc mon": "hoc_mon",
        "binh chanh": "binh_chanh",
        "cu chi": "cu_chi",
    }
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lo)
    for k, v in named.items():
        if k in cleaned:
            return v
    return "unknown"

def _parse_location_parts(location: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in str(location).split(",") if p and p.strip()]
    if not parts:
        return "Khac", "Khong ro", "Khong ro"

    street = parts[0]
    city = _normalize_city(parts[-1])

    ward = "Khong ro"
    if len(parts) >= 3:
        middle = parts[1:-1]
        preferred = next((p for p in middle if any(k in p.lower() for k in ["phuong", "xa", "thi tran"])), None)
        ward = preferred or middle[-1]
    elif len(parts) == 2:
        ward = parts[0]

    return city, ward, street

def get_location_tree() -> dict:
    for p in DATA_PATHS:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, encoding="utf-8-sig")
        except Exception:
            continue

        if "location" not in df.columns:
            continue

        tree: dict = {}
        for loc in df["location"].dropna().astype(str):
            city, ward, street = _parse_location_parts(loc)
            district = _extract_district_slug(loc + ", " + ward)

            city_node = tree.setdefault(city, {})
            ward_node = city_node.setdefault(ward, {"district": district, "streets": []})

            if ward_node.get("district", "unknown") == "unknown" and district != "unknown":
                ward_node["district"] = district

            if street and street not in ward_node["streets"]:
                if len(ward_node["streets"]) < 30:
                    ward_node["streets"].append(street)

        # Keep payload compact and deterministic for frontend.
        compact = {}
        for city in sorted(tree.keys()):
            wards = tree[city]
            compact[city] = {}
            for ward in sorted(wards.keys()):
                compact[city][ward] = {
                    "district": wards[ward].get("district", "unknown"),
                    "streets": sorted(wards[ward].get("streets", [])),
                }
        return compact

    return {}

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    report = get_report()
    model  = get_model()
    model_loaded = model is not None
    location_tree = get_location_tree()

    plot_files = []
    if PLOTS_DIR.exists():
        plot_files = sorted(
            f.name for f in PLOTS_DIR.glob("*.png")
        )

    return render_template(
        "index.html",
        report=report,
        model_loaded=model_loaded,
        plot_files=plot_files,
        location_tree=location_tree,
    )


@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()
    if model is None:
        return jsonify({"error": "Chưa có mô hình. Hãy chạy bagging_train.py trước."}), 400

    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Payload phải là JSON object hợp lệ"}), 400

        area       = float(data.get("area", 0))
        district   = str(data.get("district", "unknown"))
        bedrooms   = data.get("bedrooms")
        bathrooms  = data.get("bathrooms")
        floors     = data.get("floors")
        house_type = str(data.get("house_type", "nhà ở"))

        if area <= 0:
            return jsonify({"error": "Diện tích phải lớn hơn 0"}), 400

        pipe         = model["pipeline"]
        feature_cols = model.get("feature_cols", ["area_m2", "district"])

        sample = {
            "area_m2":    area,
            "district":   district,
            "bedrooms_n":  float(bedrooms)  if bedrooms  not in (None, "", "null") else np.nan,
            "bathrooms_n": float(bathrooms) if bathrooms not in (None, "", "null") else np.nan,
            "floors_n":    float(floors)    if floors    not in (None, "", "null") else np.nan,
            "house_type":  house_type,
        }

        row = {col: sample.get(col, None) for col in feature_cols}
        df  = pd.DataFrame([row])[feature_cols]
        price = float(pipe.predict(df)[0])

        # Lấy dự đoán từng cây để tính khoảng tin cậy bootstrap
        pre_step   = pipe.named_steps["pre"]
        bag_step   = pipe.named_steps["model"]
        X_t        = pre_step.transform(df)
        tree_preds = np.array([e.predict(X_t)[0] for e in bag_step.estimators_])
        ci_low  = float(np.percentile(tree_preds, 10))
        ci_high = float(np.percentile(tree_preds, 90))

        return jsonify({
            "price":   round(price, 0),
            "ci_low":  round(ci_low, 0),
            "ci_high": round(ci_high, 0),
            "ty":      round(price / 1000, 3),
        })

    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": f"Lỗi xử lý: {e}"}), 400


@app.route("/plots/<filename>")
def serve_plot(filename):
    # Chỉ phục vụ file .png trong thư mục plots (tránh path traversal)
    safe_name = Path(filename).name
    if not safe_name.endswith(".png"):
        return "Not found", 404
    return send_from_directory(PLOTS_DIR.resolve(), safe_name)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n  Demo web: http://127.0.0.1:{port}\n")
    app.run(debug=debug, port=port)
