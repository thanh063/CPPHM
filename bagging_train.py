#!/usr/bin/env python3
"""
Huấn luyện mô hình Bagging (Bootstrap Aggregating) dự đoán giá nhà Việt Nam.

═══════════════════════════════════════════════════════════════════
 BAGGING – Bootstrap Aggregating (Leo Breiman, 1996)
═══════════════════════════════════════════════════════════════════
Ý tưởng:
  1. Bootstrap Sampling: từ tập D (n mẫu), tạo B tập con D_b bằng
     cách lấy mẫu CÓ HOÀN LẠI (mỗi D_b ~63.2% mẫu là duy nhất).
  2. Huấn luyện B mô hình cơ sở (base estimator) độc lập trên mỗi D_b.
  3. Kết hợp: dự đoán cuối = trung bình (regression) của B mô hình.

Lợi ích:
  - Giảm Variance: Var(ensemble) ≈ Var(single)/B × (1 + (B-1)ρ)
  - OOB score: ~36.8% mẫu không dùng khi train có thể đánh giá mô hình
    mà không cần tập validation riêng.
  - Ổn định hơn cây đơn lẻ: loại bỏ nhiễu do dữ liệu training cụ thể.

Sử dụng:
    python bagging_train.py --data vietnam_house_raw.csv
    python bagging_train.py --data vietnam_house_raw.csv --plot --compare --report
"""

import argparse
import json
import logging
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold, cross_val_score, cross_validate, learning_curve, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PARSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_price(text) -> float | None:
    """
    Chuyển giá tiền → triệu VND (float).

    CSV thô / đã xử lý đều lưu giá bằng đồng (VND), ví dụ:
        65_900_000_000  → 65900.0  (triệu)
        3_699_000_000   →  3699.0
    """
    if pd.isna(text):
        return None
    if isinstance(text, (int, float)):
        # Giá lưu dạng số đồng VND – chuyển sang triệu
        return round(float(text) / 1_000_000, 1)

    t = str(text).strip().lower().replace("\xa0", " ")
    if t in ("thoả thuận", "thỏa thuận", "liên hệ", ""):
        return None

    try:
        if "tỷ" in t or " ty" in t:
            m = re.search(r"([\d,.]+)\s*t", t)
            if m:
                return round(float(m.group(1).replace(",", "")) * 1000, 1)
        if any(x in t for x in ("triệu", "trieu", " tr")):
            m = re.search(r"([\d,.]+)\s*(tri|tr)", t)
            if m:
                return round(float(m.group(1).replace(",", "")), 1)
        if "vnd" in t or "đ" in t:
            m = re.search(r"([\d,.]+)", t)
            if m:
                val = float(m.group(1).replace(",", "").replace(".", "").strip())
                return round(val / 1_000_000, 1) if val > 10_000 else val
        m = re.search(r"([\d,.]+)", t)
        if m:
            raw = m.group(1).replace(",", "")
            num = float(raw)
            return round(num / 1_000_000, 1) if num > 100_000 else round(num, 1)
    except (ValueError, AttributeError):
        pass
    return None


def parse_area(text) -> float | None:
    """Chuyển chuỗi diện tích → float m²."""
    if pd.isna(text):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    m = re.search(r"([\d,.]+)", str(text).strip())
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            pass
    return None


def parse_int_col(text) -> float | None:
    """Cột số nguyên (bedrooms, bathrooms, floors)."""
    if pd.isna(text) or str(text).strip() == "":
        return None
    m = re.search(r"\d+", str(text))
    return float(m.group()) if m else None


def extract_district(loc: str) -> str:
    """Trích quận/huyện từ chuỗi địa chỉ."""
    if not isinstance(loc, str) or not loc.strip():
        return "unknown"
    lo = loc.lower()
    m = re.search(r"(qu[aậ]n|q\.?)\s*([0-9]+|[a-zàáâãèéêìíòóôõùúưăđ]+)", lo)
    if m:
        return "q" + m.group(2).strip()
    for kw in (
        "bình thạnh", "phú nhuận", "tân bình", "tân phú",
        "bình tân", "gò vấp", "thủ đức", "nhà bè",
        "hóc môn", "bình chánh", "củ chi",
    ):
        if kw in lo:
            return kw.replace(" ", "_")
    toks = [t.strip() for t in re.split(r"[,/\-]", loc) if t.strip()]
    return toks[0] if toks else "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_clean(path: Path) -> pd.DataFrame:
    """Đọc CSV, parse giá/diện tích, làm sạch outlier.

    Tự động nhận dạng hai định dạng:
    - Dữ liệu ĐÃ XỬ LÝ: có cột price (đồng), area (m², số), district
    - Dữ liệu THÔ     : có cột price (đồng, số), area (chuỗi "325 m 2")
    Cả hai đều lưu price theo đồng VND.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info("Đọc %d hàng | Cột: %s", len(df), list(df.columns))

    # parse 'area' trước để kiểm tra is_processed chính xác ("325 m 2" → 325.0)
    if "area" in df.columns:
        df["area"] = df["area"].apply(parse_area)

    # ── Nhận dạng định dạng đầu vào ──
    # Dữ liệu đã xử lý: có cột 'district' và 'area' là số thực
    is_processed = "district" in df.columns and pd.to_numeric(df["area"], errors="coerce").notna().mean() > 0.9

    if is_processed:
        log.info("Phát hiện dữ liệu đã xử lý – bỏ qua bước parse chuỗi.")
        # price đơn vị đồng → chuyển sang triệu
        df["price_million"] = pd.to_numeric(df["price"], errors="coerce") / 1_000_000
        df["area_m2"]       = pd.to_numeric(df["area"],  errors="coerce")
    else:
        df["price_million"] = df["price"].apply(parse_price)
        df["area_m2"]       = df["area"].apply(parse_area)

    before = len(df)
    df = df.dropna(subset=["price_million", "area_m2"])
    df = df[df["area_m2"] > 8]
    df = df[df["price_million"] > 100]
    log.info("Sau lọc NaN/cận biên: %d/%d hàng", len(df), before)

    # Loại outlier cực đoan (±3σ trên log-price).
    # Với tập quá nhỏ hoặc std=0/NaN thì bỏ qua để tránh làm rỗng dữ liệu.
    log_p = np.log1p(df["price_million"])
    sigma = float(log_p.std())
    if len(df) >= 10 and np.isfinite(sigma) and sigma > 0:
        df = df[np.abs(log_p - log_p.mean()) < 3 * sigma]
        log.info("Sau loại outlier 3σ: %d hàng", len(df))
    else:
        log.info("Bỏ qua lọc outlier 3σ (mẫu ít hoặc sigma không hợp lệ): %d hàng", len(df))

    if is_processed:
        # district đã có sẵn; đảm bảo kiểu chuỗi
        if "district" not in df.columns:
            df["location"] = df.get("location", pd.Series("", index=df.index)).fillna("").astype(str)
            df["district"] = df["location"].apply(extract_district)
        else:
            df["district"] = df["district"].fillna("unknown").astype(str)
        # bedrooms_n / bathrooms_n / floors_n đã là số
        for col in ("bedrooms_n", "bathrooms_n", "floors_n"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan
    else:
        df["location"] = df.get("location", pd.Series("", index=df.index)).fillna("").astype(str)
        df["district"] = df["location"].apply(extract_district)
        for col in ("bedrooms", "bathrooms", "floors"):
            src = col if col in df.columns else None
            df[col + "_n"] = df[src].apply(parse_int_col) if src else np.nan

    if "house_type" not in df.columns:
        df["house_type"] = "nhà ở"
    df["house_type"] = df["house_type"].fillna("nhà ở").astype(str)

    df["price_per_m2"] = df["price_million"] / df["area_m2"]
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame):
    """Tự động chọn đặc trưng sẵn có."""
    num_cols = ["area_m2"]
    for c in ("bedrooms_n", "bathrooms_n", "floors_n"):
        if c in df.columns and df[c].notna().mean() > 0.10:
            num_cols.append(c)
    cat_cols = ["district"]
    if "house_type" in df.columns:
        cat_cols.append("house_type")
    return num_cols, cat_cols


def _make_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def build_pipeline(
    num_cols: list,
    cat_cols: list,
    n_estimators: int = 50,
    max_depth: int = 10,
    max_samples: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """
    Pipeline: Preprocessor → BaggingRegressor(base=DecisionTree).

    Tham số Bagging:
      n_estimators : số cây B — càng nhiều variance càng giảm
      max_samples  : tỷ lệ mẫu bootstrap mỗi lần (80%)
      bootstrap    : True → lấy mẫu có hoàn lại (Bootstrap Sampling)
      oob_score    : True → ước lượng lỗi không cần tập validation riêng
    """
    pre  = _make_preprocessor(num_cols, cat_cols)
    base = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=5,
        random_state=random_state,
    )
    bag = BaggingRegressor(
        estimator=base,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", bag)])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs(
        (np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + 1e-8)
    )) * 100)
    return {"mae": round(mae, 2), "rmse": round(rmse, 2),
            "r2": round(r2, 4),  "mape": round(mape, 2)}


def print_metrics(metrics: dict, label: str = "") -> None:
    log.info(
        "%-10s  MAE=%8.1f tr  RMSE=%8.1f tr  R²=%.4f  MAPE=%.1f%%",
        label, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["mape"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_models(X: pd.DataFrame, y: pd.Series,
                   num_cols: list, cat_cols: list) -> dict:
    """
    So sánh 6 mô hình qua 5-fold CV để minh hoạ tác dụng giảm Variance.
    """
    log.info("\n--- So sánh mô hình (5-fold CV) ---")

    def _pipe(n_est=None, use_rf=False):
        pre = _make_preprocessor(num_cols, cat_cols)
        if n_est is None:
            mdl = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
        elif use_rf:
            mdl = RandomForestRegressor(n_estimators=n_est, max_depth=10,
                                        min_samples_leaf=5, random_state=42, n_jobs=-1)
        else:
            base = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
            mdl  = BaggingRegressor(estimator=base, n_estimators=n_est,
                                    max_samples=0.8, bootstrap=True,
                                    oob_score=False, random_state=42, n_jobs=-1)
        return Pipeline([("pre", pre), ("model", mdl)])

    configs = [
        ("DecisionTree (đơn lẻ)",  _pipe()),
        ("Bagging (n=10)",         _pipe(10)),
        ("Bagging (n=25)",         _pipe(25)),
        ("Bagging (n=50)",         _pipe(50)),
        ("Bagging (n=100)",        _pipe(100)),
        ("RandomForest (n=100)",   _pipe(100, use_rf=True)),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, pipe in configs:
        cv = cross_validate(pipe, X, y, cv=kf,
                            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
                            n_jobs=-1)
        mae = float(-cv["test_mae"].mean())
        r2  = float(cv["test_r2"].mean())
        results[name] = {
            "cv_mae":     round(mae, 2),
            "cv_mae_std": round(float(cv["test_mae"].std()), 2),
            "cv_r2":      round(r2, 4),
            "cv_r2_std":  round(float(cv["test_r2"].std()), 4),
        }
        log.info("  %-30s  CV-MAE=%7.1f (±%5.1f)  CV-R²=%.4f",
                 name, mae, cv["test_mae"].std(), r2)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def make_plots(pipe: Pipeline,
               X_train, X_test, y_train, y_test,
               X_all, y_all,
               compare_results: dict,
               plot_dir: Path) -> None:
    """Tạo 5 biểu đồ đánh giá mô hình Bagging."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib chưa cài. Bỏ qua. Chạy: pip install matplotlib")
        return

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    plot_dir.mkdir(exist_ok=True)
    y_pred = pipe.predict(X_test)

    # 1. Prediction vs Actual
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.45, s=18, color="steelblue", label="Dự đoán")
    lo = min(float(y_test.min()), float(y_pred.min())) * 0.9
    hi = max(float(y_test.max()), float(y_pred.max())) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Lý tưởng (y=x)")
    ax.set_xlabel("Giá thực tế (triệu VND)", fontsize=11)
    ax.set_ylabel("Giá dự đoán (triệu VND)", fontsize=11)
    ax.set_title("Bagging: Dự đoán vs Thực tế", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / "1_prediction_vs_actual.png", dpi=130)
    plt.close(fig)
    log.info("Biểu đồ 1: %s", plot_dir / "1_prediction_vs_actual.png")

    # 2. Residuals
    residuals = y_pred - y_test.values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Sai số (triệu VND)"); axes[0].set_title("Phân phối Residuals")
    axes[1].scatter(y_pred, residuals, alpha=0.35, s=14, color="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Giá dự đoán"); axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual vs Predicted")
    fig.tight_layout()
    fig.savefig(plot_dir / "2_residuals.png", dpi=130)
    plt.close(fig)
    log.info("Biểu đồ 2: %s", plot_dir / "2_residuals.png")

    # 3. Model comparison
    if compare_results:
        names  = list(compare_results.keys())
        maes   = [compare_results[n]["cv_mae"]     for n in names]
        stds   = [compare_results[n]["cv_mae_std"] for n in names]
        r2s    = [compare_results[n]["cv_r2"]      for n in names]
        colors = (["#e74c3c"] + ["#3498db"] * (len(names) - 2) + ["#27ae60"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(names, maes, xerr=stds, color=colors,
                     alpha=0.85, height=0.55, capsize=4)
        axes[0].set_xlabel("CV MAE (triệu VND) – thấp hơn tốt hơn")
        axes[0].set_title("So sánh mô hình – MAE", fontsize=13)
        axes[0].invert_yaxis()

        axes[1].barh(names, r2s, color=colors, alpha=0.85, height=0.55)
        axes[1].set_xlabel("CV R² – cao hơn tốt hơn")
        axes[1].set_title("So sánh mô hình – R²", fontsize=13)
        axes[1].set_xlim(0, 1); axes[1].invert_yaxis()

        fig.tight_layout()
        fig.savefig(plot_dir / "3_model_comparison.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        log.info("Biểu đồ 3: %s", plot_dir / "3_model_comparison.png")

    # 4. Learning curve
    tr_sz, tr_sc, val_sc = learning_curve(
        pipe, X_all, y_all,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    tr_mae = -tr_sc.mean(axis=1);  val_mae = -val_sc.mean(axis=1)
    tr_std = tr_sc.std(axis=1);    val_std = val_sc.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tr_sz, tr_mae,  "o-",  color="steelblue", label="Train MAE")
    ax.plot(tr_sz, val_mae, "s--", color="coral",     label="Validation MAE")
    ax.fill_between(tr_sz, tr_mae - tr_std,  tr_mae + tr_std,  alpha=0.12, color="steelblue")
    ax.fill_between(tr_sz, val_mae - val_std, val_mae + val_std, alpha=0.12, color="coral")
    ax.set_xlabel("Số mẫu huấn luyện"); ax.set_ylabel("MAE (triệu VND)")
    ax.set_title("Learning Curve – BaggingRegressor", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "4_learning_curve.png", dpi=130)
    plt.close(fig)
    log.info("Biểu đồ 4: %s", plot_dir / "4_learning_curve.png")

    # 5. Permutation feature importance
    try:
        from sklearn.inspection import permutation_importance
        perm  = permutation_importance(pipe, X_test, y_test,
                                       n_repeats=15, random_state=42, n_jobs=-1)
        names_fi = list(X_test.columns)
        order = np.argsort(perm.importances_mean)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            [names_fi[i] for i in order[::-1]],
            perm.importances_mean[order[::-1]],
            xerr=perm.importances_std[order[::-1]],
            color="steelblue", alpha=0.80, capsize=3,
        )
        ax.set_xlabel("Mức giảm MAE khi xáo trộn đặc trưng")
        ax.set_title("Permutation Feature Importance", fontsize=13)
        fig.tight_layout()
        fig.savefig(plot_dir / "5_feature_importance.png", dpi=130)
        plt.close(fig)
        log.info("Biểu đồ 5: %s", plot_dir / "5_feature_importance.png")
    except Exception as e:
        log.warning("Bỏ qua feature importance: %s", e)

    log.info("Tất cả biểu đồ đã lưu vào: %s/", plot_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    df = load_and_clean(Path(args.data))
    if len(df) < 30:
        raise ValueError(
            f"Quá ít dữ liệu ({len(df)} hàng). "
            "Hãy chạy: python generate_dataset.py --rows 600"
        )

    num_cols, cat_cols = get_feature_cols(df)
    log.info("Đặc trưng số: %s | Đặc trưng phân loại: %s", num_cols, cat_cols)

    X = df[num_cols + cat_cols]
    y = df["price_million"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42,
    )
    log.info("Train: %d mẫu | Test: %d mẫu", len(X_train), len(X_test))

    pipe = build_pipeline(num_cols, cat_cols,
                          n_estimators=args.n_estimators,
                          max_depth=args.max_depth)

    log.info(
        "Huấn luyện BaggingRegressor  "
        "(n_estimators=%d, base=DecisionTree(depth=%d), bootstrap=True)…",
        args.n_estimators, args.max_depth,
    )
    pipe.fit(X_train, y_train)

    # ── Đánh giá ──────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}\n  KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH BAGGING\n{sep}")

    train_metrics = compute_metrics(y_train, pipe.predict(X_train))
    test_metrics  = compute_metrics(y_test,  pipe.predict(X_test))
    print_metrics(train_metrics, "Train  ")
    print_metrics(test_metrics,  "Test   ")

    # OOB score (ước lượng không cần tập validation riêng)
    oob_r2 = pipe.named_steps["model"].oob_score_
    log.info("OOB R² = %.4f  (không cần tập validation riêng)", oob_r2)

    # Minh hoạ lợi ích ensemble: so sánh cây đơn lẻ vs ensemble
    pre_step = pipe.named_steps["pre"]
    bag_step = pipe.named_steps["model"]
    X_test_t = pre_step.transform(X_test)
    individual_maes = [mean_absolute_error(y_test, e.predict(X_test_t))
                       for e in bag_step.estimators_]
    avg_single  = float(np.mean(individual_maes))
    pct_reduced = (avg_single - test_metrics["mae"]) / avg_single * 100
    log.info(
        "Cây đơn lẻ avg MAE=%.1f tr | Ensemble MAE=%.1f tr | "
        "Bagging giảm %.1f%% variance",
        avg_single, test_metrics["mae"], pct_reduced,
    )

    # 5-fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = cross_val_score(pipe, X, y, cv=kf,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_r2  = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    log.info("5-fold CV  MAE=%.1f (±%.1f)  R²=%.4f (±%.4f)",
             -cv_mae.mean(), cv_mae.std(), cv_r2.mean(), cv_r2.std())

    # ── So sánh mô hình (tuỳ chọn) ───────────────────────────────────────────
    compare_results: dict = {}
    if args.compare:
        compare_results = compare_models(X, y, num_cols, cat_cols)

    # ── Biểu đồ (tuỳ chọn) ───────────────────────────────────────────────────
    if args.plot:
        make_plots(pipe, X_train, X_test, y_train, y_test,
                   X, y, compare_results, Path(args.plot_dir))

    # ── Lưu mô hình ──────────────────────────────────────────────────────────
    out = Path(args.output)
    joblib.dump(
        {
            "pipeline":     pipe,
            "num_cols":     num_cols,
            "cat_cols":     cat_cols,
            "feature_cols": num_cols + cat_cols,
            "n_estimators": args.n_estimators,
            "train_size":   len(X_train),
        },
        out,
    )
    log.info("Mô hình đã lưu: %s", out)

    # ── Báo cáo JSON ─────────────────────────────────────────────────────────
    if args.report:
        report = {
            "train_metrics":           train_metrics,
            "test_metrics":            test_metrics,
            "oob_r2":                  round(oob_r2, 4),
            "cv_mae_mean":             round(float(-cv_mae.mean()), 2),
            "cv_mae_std":              round(float(cv_mae.std()), 2),
            "cv_r2_mean":              round(float(cv_r2.mean()), 4),
            "cv_r2_std":               round(float(cv_r2.std()), 4),
            "n_train":                 len(X_train),
            "n_test":                  len(X_test),
            "features":                num_cols + cat_cols,
            "n_estimators":            args.n_estimators,
            "avg_single_tree_mae":     round(avg_single, 2),
            "variance_reduction_pct":  round(pct_reduced, 2),
            "compare":                 compare_results,
        }
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Báo cáo JSON: %s", args.report)

    print(f"\n{sep}")
    print(f"  Mô hình: {out}  |  Cây: {args.n_estimators}")
    print(f"  Test MAE  = {test_metrics['mae']:>8.1f} triệu VND")
    print(f"  Test RMSE = {test_metrics['rmse']:>8.1f} triệu VND")
    print(f"  Test R²   = {test_metrics['r2']:>8.4f}")
    print(f"  OOB R²    = {oob_r2:>8.4f}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện BaggingRegressor dự đoán giá nhà Việt Nam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",          default="vietnam_house_raw.csv",
                        help="File CSV dữ liệu đầu vào")
    parser.add_argument("--output",        default="house_bagging_model.joblib",
                        help="File lưu mô hình")
    parser.add_argument("--test-size",     type=float, default=0.2,
                        help="Tỷ lệ tập test")
    parser.add_argument("--n-estimators",  type=int,   default=50,
                        help="Số cây B trong Bagging")
    parser.add_argument("--max-depth",     type=int,   default=10,
                        help="Độ sâu tối đa cây cơ sở (DecisionTree)")
    parser.add_argument("--compare",       action="store_true",
                        help="So sánh Single DT / Bagging(10,25,50,100) / RandomForest")
    parser.add_argument("--plot",          action="store_true",
                        help="Tạo 5 biểu đồ đánh giá (cần matplotlib)")
    parser.add_argument("--plot-dir",      default="plots",
                        help="Thư mục lưu biểu đồ")
    parser.add_argument("--report",        default="evaluation_report.json",
                        help="File báo cáo JSON (để trống = bỏ qua)")
    args = parser.parse_args()
    train(args)
