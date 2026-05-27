#!/usr/bin/env python3
"""
Flask web demo – Dự đoán giá nhà Việt Nam bằng Bagging.

Chạy:
    python app.py
Sau đó mở: http://127.0.0.1:5000
"""

import json
import logging
import os
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from constants import CITY_ALIAS as _CITY_ALIAS_IMPORT, VALID_CITY_NAMES

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("house_bagging_model.joblib")
REPORT_PATH = Path("evaluation_report.json")
PLOTS_DIR   = Path("plots")
DATA_PATHS  = [
    Path("vietnam_house_clean.csv"),   # ưu tiên file đã làm sạch
    Path("vietnam_house_raw.csv"),      # fallback sang raw
]

# VALID_CITY_NAMES và CITY_TOKEN_TO_NAME xây từ constants.py
CITY_TOKEN_TO_NAME = {
    _normalize: name
    for name in VALID_CITY_NAMES
    for _normalize in [
        re.sub(r"^(thanh pho|tp\.?|tinh)\s+", "", re.sub(r"[^a-z0-9\s._-]", " ",
            unicodedata.normalize("NFD", name.replace("đ", "d").replace("Đ", "D")).encode("ascii", "ignore").decode("ascii").lower()
        )).strip()
    ]
}

# ─── City alias map (module-level, build once) ───────────────────────────────
_CITY_ALIAS: dict[str, str] = _CITY_ALIAS_IMPORT

# Cache model và location tree
_model_cache: dict = {}
_location_tree_cache: dict | None = None

def get_model():
    """Load model từ disk. Tự động reload nếu file model đã thay đổi (môi trường dev)."""
    model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
    cached_mtime = _model_cache.get("mtime", -1)
    if _model_cache.get("model") is None or model_mtime != cached_mtime:
        if not MODEL_PATH.exists():
            return None
        loaded = joblib.load(MODEL_PATH)
        try:
            loaded["pipeline"].named_steps["model"].set_params(n_jobs=1)
        except Exception:
            pass
        _model_cache["model"] = loaded
        _model_cache["mtime"] = model_mtime
    return _model_cache["model"]


def get_report():
    if not REPORT_PATH.exists():
        return {}
    with open(REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _collect_plot_files() -> list[str]:
    if not PLOTS_DIR.exists():
        return []
    return sorted(f.name for f in PLOTS_DIR.glob("*.png"))


def _build_report_view(report: dict) -> dict:
    test_metrics = report.get("test_metrics") if isinstance(report.get("test_metrics"), dict) else {}
    if not test_metrics:
        test_metrics = {
            "mae": report.get("test_mae", 0),
            "rmse": report.get("rmse", 0),
            "r2": report.get("test_r2", 0),
            "mape": report.get("mape", 0),
        }

    cv_metrics = {
        "mae": report.get("cv_mae_mean", report.get("cv_mae", 0)),
        "mae_std": report.get("cv_mae_std", 0),
        "r2": report.get("cv_r2_mean", report.get("cv_r2", 0)),
        "r2_std": report.get("cv_r2_std", 0),
    }

    oob_score = report.get("oob_score", report.get("oob_r2", 0))
    compare = report.get("compare", {}) if isinstance(report.get("compare", {}), dict) else {}

    compare_rows = []
    for name, values in compare.items():
        if not isinstance(values, dict):
            continue
        compare_rows.append({
            "name": name,
            "cv_mae": values.get("cv_mae", 0),
            "cv_mae_std": values.get("cv_mae_std", 0),
            "cv_r2": values.get("cv_r2", 0),
        })

    compare_rows.sort(key=lambda item: item["cv_mae"])

    return {
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "oob_score": oob_score,
        "compare_rows": compare_rows,
        "n_estimators": report.get("n_estimators", 0),
        "n_train": report.get("n_train", 0),
        "n_test": report.get("n_test", 0),
    }


HOUSE_TYPE_MULTIPLIERS = {
    "nhà phố": 1.08,
    "căn hộ": 0.92,
    "biệt thự": 1.28,
    "nhà hẻm": 0.86,
    "đất nền": 1.12,
    "nhà trọ / phòng": 0.72,
}


def _stable_ratio(text: str, low: float = 0.92, high: float = 1.08) -> float:
    token = _normalize_token_for_match(text)
    if not token:
        return 1.0
    span = max(high - low, 0.0)
    if span == 0:
        return low
    bucket = sum(ord(ch) for ch in token) % 1000
    return low + (bucket / 999.0) * span


def _estimate_price_per_m2(base_price_per_m2: float, area: float, district: str, bedrooms, bathrooms, floors, house_type: str) -> float:
    area = max(float(area or 0), 1.0)
    rooms = 0.0
    for value, weight in ((bedrooms, 0.025), (bathrooms, 0.018), (floors, 0.015)):
        try:
            if value not in (None, "", "null"):
                rooms += float(value) * weight
        except (TypeError, ValueError):
            continue

    area_factor = 1.0 + np.clip(np.log(area / 80.0) * 0.08, -0.12, 0.18)
    district_factor = _stable_ratio(district, 0.90, 1.14)
    type_factor = HOUSE_TYPE_MULTIPLIERS.get(str(house_type).strip().lower(), 1.0)
    room_factor = 1.0 + np.clip((rooms - 0.10), -0.10, 0.18)

    estimated = float(base_price_per_m2) * area_factor * district_factor * type_factor * room_factor
    return max(estimated, 1.0)


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_accents(text: str) -> str:
    s = _clean_spaces(text).replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _normalize_token_for_match(text: str) -> str:
    s = _strip_accents(text).lower()
    s = re.sub(r"[^a-z0-9\s._-]", " ", s)
    return _clean_spaces(s)


def _normalize_city(raw: str) -> str:
    if not isinstance(raw, str):
        return "Khác"

    display = _clean_spaces(raw)
    if not display:
        return "Khác"

    token = _normalize_token_for_match(raw)
    if not token:
        return "Khác"

    if token in _CITY_ALIAS:
        return _CITY_ALIAS[token]

    token_stripped = re.sub(r"^(thanh pho|tp\.?|tinh)\s+", "", token).strip()
    if token_stripped in _CITY_ALIAS:
        return _CITY_ALIAS[token_stripped]

    # Kiểm tra trong danh mục 63 tỉnh/thành hợp lệ
    if token in CITY_TOKEN_TO_NAME:
        return CITY_TOKEN_TO_NAME[token]
    if token_stripped in CITY_TOKEN_TO_NAME:
        return CITY_TOKEN_TO_NAME[token_stripped]

    # Fallback: giữ nguyên chuỗi gốc nếu hợp lý (không quá ngắn, không phải số)
    if len(display) >= 3 and not display.isdigit():
        return display

    return "Khác"


def _is_street_like(text: str) -> bool:
    t = _normalize_token_for_match(text)
    return any(k in t for k in ("duong", "street", "ngo", "hem", "ap", "khu pho", "to"))


def _is_admin_like(text: str) -> bool:
    t = _normalize_token_for_match(text)
    return any(k in t for k in ("phuong", "xa", "thi tran", "quan", "huyen", "thi xa", "thanh pho", "tp "))


def _normalize_ward_display(text: str) -> str:
    w = _clean_spaces(text)
    if not w:
        return "Không rõ"
    w = re.sub(r"^(tinh|thanh\s*pho|tp\.?)+\s+", "", w, flags=re.IGNORECASE).strip()
    return w or "Không rõ"


def _is_valid_ward(ward: str, city: str) -> bool:
    if not ward or ward == "Không rõ":
        return False

    w = _normalize_token_for_match(ward)
    c = _normalize_token_for_match(city)
    if not w or w in {"unknown", "khong ro", "khac", "null", "none", "nan"}:
        return False

    if _is_street_like(ward):
        return False

    if c and (w == c or w in c or c in w):
        return False

    if len(w) < 2:
        return False

    return True


def _is_valid_street(street: str, ward: str) -> bool:
    if not street or street == "Không rõ":
        return False

    s = _normalize_token_for_match(street)
    w = _normalize_token_for_match(ward)
    if not s or s in {"unknown", "khong ro", "null", "none", "nan"}:
        return False

    if _is_admin_like(street):
        return False

    if w and s == w:
        return False

    return len(s) >= 3


def _extract_district_slug(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    lo = _normalize_token_for_match(text)

    m = re.search(r"(quan|q\.?)\s*([0-9]+)", lo)
    if m:
        return "q" + m.group(2)

    m = re.search(r"\bq\s*([0-9]+)\b", lo)
    if m:
        return "q" + m.group(1)

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
    for k, v in named.items():
        if k in lo or k.replace(" ", "_") in lo:
            return v
    return "unknown"


def _parse_location_parts(location: str) -> tuple[str, str, str]:
    parts = [_clean_spaces(p) for p in str(location).split(",") if _clean_spaces(p)]
    if not parts:
        return "Khác", "Không rõ", "Không rõ"

    city = _normalize_city(parts[-1])
    street = "Không rõ"
    ward = "Không rõ"

    if len(parts) >= 3:
        if _is_street_like(parts[0]) or not _is_admin_like(parts[0]):
            street = parts[0]

        middle = parts[1:-1]
        preferred = next((p for p in middle if _is_admin_like(p)), None)
        ward = preferred or middle[-1]
    elif len(parts) == 2:
        first = parts[0]
        if _is_admin_like(first):
            ward = first
        else:
            street = first

    if ward == street:
        ward = "Không rõ"

    ward = _normalize_ward_display(ward)
    return city, ward or "Không rõ", street or "Không rõ"

def get_location_tree() -> dict:
    """Build location tree từ CSV. Kết quả được cache trong RAM (xóa cache khi restart)."""
    global _location_tree_cache
    if _location_tree_cache is not None:
        return _location_tree_cache

    result: dict = {}
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
        for _, row in df.iterrows():
            loc = str(row.get("location", "") or "")
            if not loc.strip():
                continue

            city, ward, street = _parse_location_parts(loc)
            if city == "Khác":
                continue
            if not _is_valid_ward(ward, city):
                continue

            district_raw = str(row.get("district", "") or "").strip()
            district = district_raw if district_raw else "unknown"

            city_node = tree.setdefault(city, {})
            ward_node = city_node.setdefault(ward, {"district": district, "streets": []})

            if ward_node.get("district", "unknown") == "unknown" and district != "unknown":
                ward_node["district"] = district

            if _is_valid_street(street, ward) and street not in ward_node["streets"]:
                if len(ward_node["streets"]) < 30:
                    ward_node["streets"].append(street)

        # Keep payload compact and deterministic for frontend.
        compact: dict = {}
        for city in sorted(tree.keys()):
            wards = tree[city]
            compact[city] = {}
            for ward in sorted(wards.keys()):
                if not wards[ward].get("streets") and wards[ward].get("district", "unknown") == "unknown":
                    continue
                compact[city][ward] = {
                    "district": wards[ward].get("district", "unknown"),
                    "streets":  sorted(wards[ward].get("streets", [])),
                }
            if not compact[city]:
                compact.pop(city, None)

        if compact:
            result = compact
            break   # dùng file dữ liệu đầu tiên hợp lệ

    _location_tree_cache = result
    return result

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    report = get_report()
    report_view = _build_report_view(report)
    model  = get_model()
    model_loaded = model is not None
    location_tree = get_location_tree()

    return render_template(
        "index.html",
        report=report,
        report_view=report_view,
        model_loaded=model_loaded,
        plot_files=_collect_plot_files(),
        location_tree=location_tree,
    )


@app.route("/about")
def about():
    report = get_report()
    return render_template(
        "about.html",
        report=report,
        model_loaded=get_model() is not None,
    )


@app.route("/analysis")
def analysis():
    try:
        report = get_report()
        plot_files = [name for name in ["price_analysis.png", "1_prediction_vs_actual.png", "2_residuals.png", "3_model_comparison.png", "4_learning_curve.png", "5_feature_importance.png"] if (PLOTS_DIR / name).exists()]
        report_view = _build_report_view(report) if report else {}
        return render_template(
            "analysis.html",
            report=report,
            report_view=report_view,
            model_loaded=get_model() is not None,
            plot_files=plot_files,
        )
    except Exception as e:
        log.warning("analysis route error: %s", e)
        return render_template("analysis.html", report={}, report_view={}, model_loaded=False, plot_files=[])


@app.route("/report")
def report_page():
    report = get_report()
    return render_template(
        "report.html",
        report=report,
        report_view=_build_report_view(report),
        model_loaded=get_model() is not None,
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

        # Validate input existence and types
        try:
            area       = float(data.get("area", 0))
            district   = str(data.get("district", "unknown"))
            house_type = str(data.get("house_type", "nhà ở"))
        except (ValueError, TypeError):
             return jsonify({"error": "Dữ liệu đầu vào không hợp lệ"}), 400

        if area <= 0 or area > 10_000:
            return jsonify({"error": "Diện tích phải trong khoảng 1 – 10.000 m²"}), 400

        # Validate bedrooms, bathrooms, floors
        for field, max_val in (("bedrooms", 20), ("bathrooms", 15), ("floors", 40)):
            val = data.get(field)
            if val not in (None, "", "null"):
                try:
                    v = int(float(val))
                    if v < 0 or v > max_val:
                        return jsonify({"error": f"{field} phải trong khoảng 0 – {max_val}"}), 400
                except (ValueError, TypeError):
                    return jsonify({"error": f"{field} không hợp lệ"}), 400

        pipe         = model["pipeline"]
        feature_cols = model.get("feature_cols", ["area_m2", "district"])
        use_log      = model.get("use_log_target", False)

        # District encoding – hỗ trợ cả model cũ (district_map) lẫn model mới (district_log_map)
        district_log_map    = model.get("district_log_map", {})
        district_map_model  = model.get("district_map", {})
        global_log_median   = float(model.get("global_log_median", np.log1p(5000)))
        global_median_price = float(model.get("global_median_price",
                                              model.get("median_price_per_m2", 5000)))
        district_log_val = float(district_log_map.get(district, global_log_median))
        district_median  = float(district_map_model.get(district, global_median_price))

        sample = {
            "area_m2":               area,
            "log_area":              float(np.log1p(area)),
            "district_log_median":   district_log_val,
            "district_median_price": district_median,
            "district":              district,
            "bedrooms_n":  float(data["bedrooms"])  if data.get("bedrooms")  not in (None, "", "null") else np.nan,
            "bathrooms_n": float(data["bathrooms"]) if data.get("bathrooms") not in (None, "", "null") else np.nan,
            "floors_n":    float(data["floors"])    if data.get("floors")    not in (None, "", "null") else np.nan,
            "house_type":  house_type,
            # backward-compat với model cũ
            "price_per_m2": district_median / max(area, 1),
        }

        row = {col: sample.get(col, None) for col in feature_cols}
        df  = pd.DataFrame([row])[feature_cols]

        # Dự đoán (với log-transform nếu model mới)
        raw_pred = float(pipe.predict(df)[0])
        price    = float(np.expm1(raw_pred)) if use_log else raw_pred

        # Khoảng tin cậy từ từng cây
        pre_step       = pipe.named_steps["pre"]
        bag_step       = pipe.named_steps["model"]
        X_t            = pre_step.transform(df)
        tree_preds_raw = np.array([e.predict(X_t)[0] for e in bag_step.estimators_])
        tree_preds     = np.expm1(tree_preds_raw) if use_log else tree_preds_raw

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


@app.route('/region')
def region():
    try:
        # Tìm file dữ liệu hợp lệ theo DATA_PATHS
        df = None
        for p in DATA_PATHS:
            try:
                if p.exists():
                    df = pd.read_csv(p, encoding='utf-8-sig')
                    break
            except Exception:
                continue

        if df is None:
            return render_template('region.html', stats=[], top_expensive=[], top_cheap=[], summary={}, error='Dữ liệu không tìm thấy')

        # Kiểm tra cột cần thiết
        for c in ('price', 'area', 'location'):
            if c not in df.columns:
                return render_template('region.html', stats=[], top_expensive=[], top_cheap=[], summary={}, error=f'Thiếu cột: {c}')

        df = df.copy()
        
        # Trích xuất số thực từ chuỗi (xử lý "60 m2" -> 60.0)
        df['price'] = df['price'].astype(str).str.replace(',', '.').str.extract(r'(\d+(?:\.\d+)?)')[0]
        df['area']  = df['area'].astype(str).str.replace(',', '.').str.extract(r'(\d+(?:\.\d+)?)')[0]
        
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area']  = pd.to_numeric(df['area'],  errors='coerce')
        
        df = df.dropna(subset=['price', 'area'])
        df = df[df['area'] >= 10]        # loại diện tích < 10m² (lỗi crawl)
        
        # Chuẩn hoá giá về triệu VND
        # Nếu giá trung vị > 1.000 thì đơn vị đồng VND, cần chia 1.000.000
        price_med = df['price'].median()
        if price_med > 500_000:
            df['price'] = df['price'] / 1_000_000
        
        # Đưa về đơn vị triệu VND để lọc outlier
        df['price_million'] = df['price']
        df['price_per_m2'] = df['price_million'] / df['area']
        
        # Lọc bỏ tin ảo theo ngưỡng thực tế Việt Nam:
        # Giá/m²: tối thiểu 3 triệu (đất tỉnh lẻ), tối đa 300 triệu (nội ô HN/HCM cao cấp)
        # Tổng giá: tối thiểu 200 triệu, tối đa 100 tỷ (= 100.000 triệu)
        df = df[
            (df['price_million'] >= 200) &
            (df['price_million'] <= 100_000) &
            (df['price_per_m2'] >= 3.0) &
            (df['price_per_m2'] <= 300.0)
        ]

        # Chuẩn hoá tên tỉnh bằng hàm đã có
        df['city'] = df['location'].apply(
            lambda loc: _normalize_city(str(loc).rsplit(',', 1)[-1].strip())
        )
        df = df[df['city'] != 'Khác']

        # ── Tính toán thống kê theo tỉnh (giá đơn vị triệu VND) ──────────────
        province_stats = df.groupby('city').agg(
            count        =('price',       'count'),
            median       =('price',       'median'),
            min_price    =('price',       'min'),
            max_price    =('price',       'max'),
            price_per_m2 =('price_per_m2','median'),
        ).reset_index()

        # Lọc tỉnh có ít nhất 5 BĐS
        province_stats = province_stats[province_stats['count'] >= 5].copy()

        # Chuyển về tỷ đồng để hiển thị (tổng giá trị nhà)
        province_stats['median_ty']  = (province_stats['median']       / 1e9).round(3)
        province_stats['min_ty']     = (province_stats['min_price']    / 1e9).round(3)
        province_stats['max_ty']     = (province_stats['max_price']    / 1e9).round(3)
        
        # price_per_m2 đã được quy đổi sang triệu/m2 ở bước trước
        province_stats['ppm2_m']     = province_stats['price_per_m2'].round(2)

        # Xếp hạng tỉnh thành dựa trên giá trên 1m2 để phản ánh đúng mức độ đắt đỏ
        province_stats = province_stats.sort_values('price_per_m2', ascending=False)

        top5_expensive = province_stats.nlargest(5, 'price_per_m2')
        top5_cheap     = province_stats.nsmallest(5, 'price_per_m2')

        # ── Summary toàn quốc ────────────────────────────────────────────────
        global_median_ty = round(df['price'].median() / 1e9, 3)
        global_ppm2_m    = round(df['price_per_m2'].median(), 2)
        n_provinces      = int(province_stats.shape[0])

        # Tổng BĐS = n_train + n_test từ evaluation_report.json
        # (phản ánh đúng số mẫu đã dùng để huấn luyện, sau khi lọc outlier)
        _rpt = get_report()
        n_total_report = _rpt.get('n_train', 0) + _rpt.get('n_test', 0)
        n_total = int(n_total_report) if n_total_report > 0 else int(df.shape[0])

        summary = {
            'total_bds':   n_total,
            'n_provinces': n_provinces,
            'median_ty':   global_median_ty,
            'ppm2_m':      global_ppm2_m,
        }

        # ── Highlight tỉnh đắt nhất / rẻ nhất ───────────────────────────────
        max_median_city = province_stats.iloc[0]['city']  if not province_stats.empty else ''
        min_median_city = province_stats.iloc[-1]['city'] if not province_stats.empty else ''

        def _row(r):

            return {
                'province':    r['city'],
                'median_price': r['median_ty'],
                'min_price':   r['min_ty'],
                'max_price':   r['max_ty'],
                'per_m2':      r['ppm2_m'],
                'count':       int(r['count']),
                'is_max':      r['city'] == max_median_city,
                'is_min':      r['city'] == min_median_city,
            }

        stats_list      = [_row(r) for _, r in province_stats.iterrows()]
        top_expensive   = [_row(r) for _, r in top5_expensive.iterrows()]
        top_cheap       = [_row(r) for _, r in top5_cheap.iterrows()]

        return render_template(
            'region.html',
            stats=stats_list,
            top_expensive=top_expensive,
            top_cheap=top_cheap,
            summary=summary,
        )
    except Exception as e:
        log.exception("region route error")
        return render_template('region.html', stats=[], top_expensive=[], top_cheap=[], summary={}, error=str(e))


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html") if (Path(app.root_path) / "templates" / "404.html").exists() \
        else (jsonify({"error": "Không tìm thấy trang"}), 404)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n  Demo web: http://127.0.0.1:{port}\n")
    app.run(debug=debug, port=port)
