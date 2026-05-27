#!/usr/bin/env python3
"""Tạo báo cáo tự động sau khi huấn luyện mô hình Bagging.

Script này đọc model đã lưu, tải dữ liệu gốc, tính các chỉ số đánh giá,
vẽ biểu đồ và xuất JSON báo cáo để dùng cho web app và slide thuyết trình.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from bagging_train import build_pipeline, compare_models, get_feature_cols, load_and_clean


MODEL_PATH = Path("house_bagging_model.joblib")
DATA_PATH = Path("vietnam_house_raw.csv")
REPORT_PATH = Path("evaluation_report.json")
PLOTS_DIR = Path("plots")


def _load_bundle(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("File mô hình không đúng định dạng bundle của dự án")
    return bundle


def _format_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [title, sep, fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    lines.append(sep)
    return "\n".join(lines)


def _save_price_analysis(df: pd.DataFrame, output_path: Path) -> None:
    if "district" not in df.columns:
        return

    top_districts = df["district"].value_counts().head(12).index.tolist()
    subset = df[df["district"].isin(top_districts)].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=subset, x="price_million", y="district", order=top_districts, ax=ax, palette="viridis")
    ax.set_title("Phân phối giá theo khu vực")
    ax.set_xlabel("Giá (triệu VND)")
    ax.set_ylabel("Quận/Huyện")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_prediction_vs_actual(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.55, s=26, color="#2563eb", edgecolor="white", linewidth=0.4)
    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.8)
    ax.set_title("Dự đoán vs Thực tế")
    ax.set_xlabel("Giá thực tế (triệu VND)")
    ax.set_ylabel("Giá dự đoán (triệu VND)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_residuals(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> np.ndarray:
    residuals = np.asarray(y_true - y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=24, kde=True, color="#0f766e", ax=ax)
    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_title("Phân phối residuals")
    ax.set_xlabel("Residual = y_thực - y_dự đoán (triệu VND)")
    ax.set_ylabel("Số lượng")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return residuals


def _save_feature_importance(pipe, output_path: Path) -> list[dict]:
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    if not hasattr(model, "estimators_"):
        return []

    try:
        feature_names = list(pre.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{idx}" for idx in range(len(model.estimators_[0].feature_importances_))]

    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, palette="crest")
    ax.set_title("Feature importance")
    ax.set_xlabel("Mức độ quan trọng")
    ax.set_ylabel("Đặc trưng")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return [
        {"feature": row.feature, "importance": float(row.importance)}
        for row in importance_df.itertuples(index=False)
    ]


def _save_learning_curve(X_train, y_train, num_cols, cat_cols, base_max_depth: int, output_path: Path) -> list[dict]:
    """Vẽ validation curve theo số cây (n_estimators): Train MAE vs Validation MAE."""
    train_sub, val_sub, y_train_sub, y_val_sub = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    estimator_grid = [10, 25, 50, 100]
    records: list[dict] = []

    for n_estimators in estimator_grid:
        pipe = build_pipeline(num_cols, cat_cols, n_estimators=n_estimators, max_depth=base_max_depth)
        pipe.fit(train_sub, y_train_sub)
        train_pred = pipe.predict(train_sub)
        val_pred = pipe.predict(val_sub)
        records.append({
            "n_estimators": n_estimators,
            "train_mae": float(mean_absolute_error(y_train_sub, train_pred)),
            "val_mae": float(mean_absolute_error(y_val_sub, val_pred)),
        })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["n_estimators"] for r in records], [r["train_mae"] for r in records], marker="o", linewidth=2.2, label="Train MAE")
    ax.plot([r["n_estimators"] for r in records], [r["val_mae"] for r in records], marker="o", linewidth=2.2, label="Validation MAE")
    ax.set_title("Validation Curve theo số cây (n_estimators)")
    ax.set_xlabel("Số cây (n_estimators)")
    ax.set_ylabel("MAE (triệu VND)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return records


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_PATH}")

    bundle = _load_bundle(MODEL_PATH)
    pipe = bundle["pipeline"]

    df = load_and_clean(DATA_PATH)
    num_cols, cat_cols = get_feature_cols(df)
    feature_cols = bundle.get("feature_cols", num_cols + cat_cols)

    X = df[feature_cols]
    y = df["price_million"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = pipe.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    test_r2 = float(r2_score(y_test, y_pred))
    test_mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-9))) * 100)
    oob_score = float(getattr(pipe.named_steps["model"], "oob_score_", getattr(pipe.named_steps["model"], "oob_score", float("nan"))))

    compare = compare_models(X_train, y_train, num_cols, cat_cols)
    cv_reference = compare.get("Bagging(100)") or next(iter(compare.values()), {}) if compare else {}

    PLOTS_DIR.mkdir(exist_ok=True)
    _save_price_analysis(df, PLOTS_DIR / "price_analysis.png")
    _save_prediction_vs_actual(y_test, y_pred, PLOTS_DIR / "1_prediction_vs_actual.png")
    residuals = _save_residuals(y_test, y_pred, PLOTS_DIR / "2_residuals.png")
    feature_importance = _save_feature_importance(pipe, PLOTS_DIR / "5_feature_importance.png")
    learning_curve_data = _save_learning_curve(X_train, y_train, num_cols, cat_cols, int(getattr(pipe.named_steps["model"].estimator, "max_depth", 15)), PLOTS_DIR / "4_learning_curve.png")

    report_data = {
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_metrics": {
            "mae": test_mae,
            "rmse": test_rmse,
            "r2": test_r2,
            "mape": test_mape,
        },
        "oob_r2": oob_score,
        "oob_score": oob_score,
        "n_estimators": int(getattr(pipe.named_steps["model"], "n_estimators", 0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_mae_mean": float(cv_reference.get("cv_mae", 0)),
        "cv_mae_std": float(cv_reference.get("cv_mae_std", 0)),
        "cv_r2_mean": float(cv_reference.get("cv_r2", 0)),
        "cv_r2_std": float(cv_reference.get("cv_r2_std", 0)),
        "compare": compare,
        "feature_importance": feature_importance,
        "plots": {
            "price_analysis":        "price_analysis.png",
            "prediction_vs_actual":  "1_prediction_vs_actual.png",
            "residuals":             "2_residuals.png",
            "feature_importance":    "5_feature_importance.png",
            "learning_curve":        "4_learning_curve.png",
        },
        "residual_mean": float(np.mean(residuals)),
        "residual_std":  float(np.std(residuals)),
        "learning_curve": learning_curve_data,
    }

    REPORT_PATH.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = [
        ["Test MAE", f"{test_mae:,.2f}"],
        ["Test RMSE", f"{test_rmse:,.2f}"],
        ["Test R²", f"{test_r2:.4f}"],
        ["MAPE", f"{test_mape:.2f}%"],
        ["OOB score", f"{oob_score:.4f}"],
        ["CV MAE", f"{report_data['cv_mae_mean']:,.2f} ± {report_data['cv_mae_std']:,.2f}"],
        ["CV R²", f"{report_data['cv_r2_mean']:.4f} ± {report_data['cv_r2_std']:.4f}"],
    ]
    print()
    print(_format_table("BẢNG TỔNG KẾT MÔ HÌNH", ["Chỉ số", "Giá trị"], summary_rows))

    if compare:
        compare_rows = [[name, f"{row['cv_mae']:,.2f}", f"{row['cv_mae_std']:,.2f}", f"{row['cv_r2']:.4f}"] for name, row in compare.items()]
        print()
        print(_format_table("BẢNG SO SÁNH MÔ HÌNH", ["Mô hình", "CV MAE", "±", "CV R²"], compare_rows))

    print()
    print(f"Đã lưu báo cáo: {REPORT_PATH}")
    print(f"Đã lưu biểu đồ trong: {PLOTS_DIR}")


if __name__ == "__main__":
    main()