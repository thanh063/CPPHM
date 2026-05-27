#!/usr/bin/env python3
"""
clean_data.py — Làm sạch dữ liệu bất động sản thô.

Bộ lọc thực tế Việt Nam:
  - Giá: 200 triệu ≤ price ≤ 100 tỷ
  - Diện tích: 10m² ≤ area ≤ 5000m²
  - Giá/m²: 3 triệu ≤ ppm2 ≤ 300 triệu  (thực tế VN, kể cả trung tâm HN/HCM)
  - Loại bỏ hàng thiếu location hoặc price
  - Loại bỏ outlier cực đoan theo tỉnh (IQR × 5 per-province)

Sử dụng:
    python clean_data.py
    python clean_data.py --input vietnam_house_raw.csv --output vietnam_house_clean.csv
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Ngưỡng lọc cứng ─────────────────────────────────────────────────────────
MIN_PRICE_M   = 200        # triệu VND
MAX_PRICE_M   = 100_000    # triệu VND (= 100 tỷ)
MIN_AREA      = 10         # m²
MAX_AREA      = 5_000      # m²
MIN_PPM2      = 3.0        # triệu/m²  — đất tỉnh lẻ rẻ nhất
MAX_PPM2      = 300.0      # triệu/m²  — cao cấp HN/HCM nội đô

# Một số tỉnh trung tâm cho phép ppm2 cao hơn
HIGH_VALUE_CITIES = {
    "hà nội", "tp. hồ chí minh", "hồ chí minh", "đà nẵng",
    "hải phòng", "bình dương", "đồng nai",
}
MAX_PPM2_PREMIUM = 300.0   # vẫn giữ 300 để loại rác hoàn toàn


def parse_number(text) -> float | None:
    """Trích số thực đầu tiên từ chuỗi."""
    if pd.isna(text):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    m = re.search(r"[\d]+(?:[.,][\d]+)?", str(text).strip())
    if m:
        try:
            return float(m.group().replace(",", "."))
        except ValueError:
            pass
    return None


def normalize_city(loc: str) -> str:
    """Lấy tên tỉnh từ chuỗi location (phần cuối sau dấu phẩy cuối cùng)."""
    if not isinstance(loc, str):
        return "unknown"
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    return parts[-1].lower() if parts else "unknown"


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    total_raw = len(df)

    # ── 1. Ép kiểu ────────────────────────────────────────────────────────────
    df = df.copy()
    df["price_m"]  = df["price"].apply(parse_number)
    df["area_n"]   = df["area"].apply(parse_number)

    # Nếu price được lưu dạng đồng (>= 1_000_000 mỗi triệu), quy đổi
    if df["price_m"].median() > 1_000:
        # Kiểm tra: nếu median > 1000 thì đơn vị là đồng, cần chia 1_000_000
        if df["price_m"].median() > 500_000:
            df["price_m"] = df["price_m"] / 1_000_000
            if verbose:
                print("[INFO] Phát hiện price lưu dạng đồng VND → chia 1.000.000 → triệu")

    # ── 2. Lọc NaN cơ bản ─────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["price_m", "area_n", "location"])
    df = df[df["location"].astype(str).str.strip() != ""]
    if verbose:
        print(f"[2] Loại NaN price/area/location: {before - len(df):,} hàng → còn {len(df):,}")

    # ── 3. Lọc diện tích vô lý ────────────────────────────────────────────────
    before = len(df)
    df = df[(df["area_n"] >= MIN_AREA) & (df["area_n"] <= MAX_AREA)]
    if verbose:
        print(f"[3] Lọc area [{MIN_AREA}–{MAX_AREA}m²]: loại {before - len(df):,} → còn {len(df):,}")

    # ── 4. Lọc giá vô lý ──────────────────────────────────────────────────────
    before = len(df)
    df = df[(df["price_m"] >= MIN_PRICE_M) & (df["price_m"] <= MAX_PRICE_M)]
    if verbose:
        print(f"[4] Lọc giá [{MIN_PRICE_M}–{MAX_PRICE_M} triệu]: loại {before - len(df):,} → còn {len(df):,}")

    # ── 5. Tính giá/m² và lọc outlier cứng ───────────────────────────────────
    df["ppm2"] = df["price_m"] / df["area_n"]

    before = len(df)
    df["_city"] = df["location"].apply(normalize_city)
    mask = (df["ppm2"] >= MIN_PPM2)
    city_mask = df["_city"].isin(HIGH_VALUE_CITIES)
    mask &= np.where(city_mask, df["ppm2"] <= MAX_PPM2_PREMIUM, df["ppm2"] <= MAX_PPM2)
    df = df[mask]
    if verbose:
        print(f"[5] Lọc ppm2 [{MIN_PPM2}–{MAX_PPM2} tr/m²]: loại {before - len(df):,} → còn {len(df):,}")

    # ── 6. Lọc outlier per-province (IQR × 4) ─────────────────────────────────
    before = len(df)
    clean_rows = []
    for city, grp in df.groupby("_city"):
        if len(grp) < 5:
            clean_rows.append(grp)
            continue
        q1 = grp["ppm2"].quantile(0.10)
        q3 = grp["ppm2"].quantile(0.90)
        iqr = q3 - q1
        lo  = max(MIN_PPM2, q1 - 4 * iqr)
        hi  = min(MAX_PPM2_PREMIUM, q3 + 4 * iqr)
        filtered = grp[(grp["ppm2"] >= lo) & (grp["ppm2"] <= hi)]
        clean_rows.append(filtered)

    df = pd.concat(clean_rows, ignore_index=True) if clean_rows else df.iloc[0:0]
    if verbose:
        print(f"[6] Lọc outlier per-province (IQR×4): loại {before - len(df):,} → còn {len(df):,}")

    # ── 7. Loại bản ghi trùng lặp ─────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["price_m", "area_n", "location"])
    if verbose:
        print(f"[7] Loại trùng lặp: loại {before - len(df):,} → còn {len(df):,}")

    # ── 8. Cập nhật lại cột price và area chuẩn hoá ──────────────────────────
    # Ghi đè price = giá triệu VND (số thực) → phù hợp với bagging_train.py
    df["price"] = df["price_m"]
    df["area"]  = df["area_n"]

    # Drop cột phụ
    df = df.drop(columns=["price_m", "area_n", "ppm2", "_city"], errors="ignore")

    if verbose:
        pct = 100 * (1 - len(df) / total_raw) if total_raw > 0 else 0
        print(f"\n✅ Kết quả: {total_raw:,} → {len(df):,} hàng sạch (loại {pct:.1f}%)")
        # Thống kê tỉnh
        city_col = df["location"].apply(normalize_city)
        top_prov = city_col.value_counts().head(10)
        print("\nTop 10 tỉnh/TP nhiều BĐS nhất:")
        for city, cnt in top_prov.items():
            print(f"  {city:<30} {cnt:>5} BĐS")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Làm sạch dữ liệu BĐS thô theo ngưỡng thực tế Việt Nam"
    )
    parser.add_argument("--input",  default="vietnam_house_raw.csv")
    parser.add_argument("--output", default="vietnam_house_clean.csv")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Không tìm thấy file: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Đọc dữ liệu từ: {input_path}")
    df_raw = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"Tổng {len(df_raw):,} hàng | Cột: {df_raw.columns.tolist()}\n")

    df_clean = clean(df_raw, verbose=not args.quiet)

    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Đã lưu: {output_path} ({output_path.stat().st_size // 1024:,} KB)")


if __name__ == "__main__":
    main()
