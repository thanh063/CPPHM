#!/usr/bin/env python3
"""
Tạo tập dữ liệu giá nhà Việt Nam tổng hợp để huấn luyện và thử nghiệm.

Sinh ra dữ liệu thực tế với mô hình giá:
    price = base_price_per_m2(district) * area * type_factor * feature_adj * noise

Sử dụng:
    python generate_dataset.py --rows 600 --output vietnam_house_raw.csv
"""

import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Tham số dữ liệu ──────────────────────────────────────────────────────────
# (tên quận, giá_min_per_m2, giá_max_per_m2, trọng_số_sinh_mẫu)
# Đơn vị: triệu VND / m²
DISTRICTS = [
    ("Quận 1",           300, 500, 1),
    ("Quận 2",           150, 280, 2),
    ("Quận 3",           220, 400, 1),
    ("Quận 4",           110, 190, 2),
    ("Quận 5",           160, 280, 1),
    ("Quận 6",            90, 160, 2),
    ("Quận 7",           120, 240, 2),
    ("Quận 8",            75, 135, 2),
    ("Quận 9",            60, 120, 3),
    ("Quận 10",          150, 260, 1),
    ("Quận 11",          130, 220, 1),
    ("Quận 12",           65, 125, 3),
    ("Bình Thạnh",       110, 200, 2),
    ("Phú Nhuận",        160, 300, 1),
    ("Tân Bình",         110, 190, 2),
    ("Tân Phú",           85, 150, 2),
    ("Bình Tân",          65, 125, 3),
    ("Gò Vấp",            90, 170, 3),
    ("Thủ Đức",           75, 155, 3),
    ("Nhà Bè",            60, 110, 2),
    ("Hóc Môn",           50,  95, 2),
    ("Bình Chánh",        45,  90, 2),
    ("Củ Chi",            35,  75, 1),
    ("Cần Giờ",           30,  65, 1),
]

# (loại nhà, hệ_số_giá, diện_tích_min, diện_tích_max,
#  phòngngủ_min, phòngngủ_max, phòng_tắm_min, phòng_tắm_max,
#  tầng_min, tầng_max, trọng_số)
HOUSE_TYPES = [
    ("nhà phố",          1.05,  45, 200, 2, 5, 1, 4, 2, 5, 25),
    ("biệt thự",         1.40, 150, 600, 3, 7, 2, 6, 2, 4,  5),
    ("căn hộ",           0.85,  30, 150, 1, 4, 1, 3, 0, 0, 30),
    ("nhà hẻm",          0.80,  25, 100, 2, 4, 1, 3, 1, 4, 20),
    ("đất nền",          1.10,  60, 500, 0, 0, 0, 0, 0, 0,  8),
    ("nhà trọ / phòng",  0.65,  20,  60, 1, 3, 1, 2, 1, 3, 12),
]

STREETS = [
    "Nguyễn Huệ", "Lê Lợi", "Trần Hưng Đạo", "Đinh Tiên Hoàng",
    "Cách Mạng Tháng 8", "Hoàng Văn Thụ", "Phan Xích Long",
    "Nguyễn Thị Minh Khai", "Điện Biên Phủ", "Võ Văn Tần",
    "Nam Kỳ Khởi Nghĩa", "Lý Thường Kiệt", "Trường Chinh",
    "Tô Hiến Thành", "Lê Văn Sỹ", "Phạm Văn Đồng", "Ngô Gia Tự",
    "Nguyễn Văn Trỗi", "Bùi Thị Xuân", "Hai Bà Trưng",
    "Bà Huyện Thanh Quan", "Phó Đức Chính", "Nguyễn Đình Chiểu",
]


def _generate_one(rng: random.Random) -> dict:
    # ── Chọn quận/huyện ──
    dist_weights = [d[3] for d in DISTRICTS]
    dist = rng.choices(DISTRICTS, weights=dist_weights, k=1)[0]
    dist_name, price_min, price_max, _ = dist

    # ── Chọn loại nhà ──
    type_weights = [h[10] for h in HOUSE_TYPES]
    ht = rng.choices(HOUSE_TYPES, weights=type_weights, k=1)[0]
    (house_type, type_factor, a_min, a_max,
     bed_min, bed_max, bath_min, bath_max, fl_min, fl_max, _) = ht

    # ── Diện tích ──
    area = rng.uniform(a_min, a_max)

    # ── Phòng ngủ / phòng tắm / tầng ──
    bedrooms  = rng.randint(bed_min,  bed_max)  if bed_max  > 0 else 0
    bathrooms = rng.randint(bath_min, bath_max) if bath_max > 0 else 0
    floors    = rng.randint(fl_min,   fl_max)   if fl_max   > 0 else 0

    # ── Điều chỉnh theo tiện nghi ──
    feature_adj = 1.0
    feature_adj += bedrooms  * 0.04   # mỗi phòng ngủ thêm +4%
    feature_adj += bathrooms * 0.025  # mỗi phòng tắm thêm +2.5%
    feature_adj += floors    * 0.015  # mỗi tầng thêm +1.5%

    # ── Giá = giá_per_m2 × diện_tích × hệ_số_loại × điều_chỉnh × nhiễu ──
    price_per_m2 = rng.uniform(price_min, price_max)
    price = price_per_m2 * area * type_factor * feature_adj
    price *= rng.uniform(0.82, 1.18)   # ±18% nhiễu ngẫu nhiên
    price = max(200.0, price)           # tối thiểu 200 triệu VND

    # ── Format ──
    street  = rng.choice(STREETS)
    number  = rng.randint(1, 250)
    location = f"{number} {street}, {dist_name}, TP.HCM"

    return {
        "price":      int(round(price * 1_000_000)),
        "area":       round(area, 1),
        "location":   location,
        "district":   dist_name,
        "bedrooms_n":  bedrooms  if bedrooms  > 0 else None,
        "bathrooms_n": bathrooms if bathrooms > 0 else None,
        "floors_n":    floors    if floors    > 0 else None,
        "house_type":  house_type,
    }


def generate(n: int, output: Path, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)

    rows = [_generate_one(rng) for _ in range(n)]
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"\n✓ Đã tạo {n} bản ghi  →  {output}")
    print("\n── Thống kê tổng quan ──────────────────────────────────")
    print(f"  Phân bố loại nhà:\n{df['house_type'].value_counts().to_string()}")
    print(f"\n  Top 5 quận:\n{df['district'].value_counts().head().to_string()}")
    print(f"\n  Giá (đồng VND) – min/mean/max: {df['price'].min():,.0f} / {df['price'].mean():,.0f} / {df['price'].max():,.0f}")
    print(f"  Diện tích (m²) – min/mean/max: {df['area'].min():.0f} / {df['area'].mean():.0f} / {df['area'].max():.0f}")
    print(f"\n  Cột: {list(df.columns)}")
    print(f"  Số hàng: {len(df)}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tạo tập dữ liệu giá nhà Việt Nam tổng hợp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rows",   type=int, default=600,
                        help="Số bản ghi cần tạo")
    parser.add_argument("--output", default="vietnam_house_real.csv",
                        help="File CSV đầu ra")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed để tái lặp")
    args = parser.parse_args()

    generate(args.rows, Path(args.output), args.seed)
