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

# ─── Cấu hình các tỉnh/thành phố ──────────────────────────────────────────────
# (tên_hiển_thị, chuỗi_city_trong_location, hệ_số_giá, trọng_số_sinh_mẫu)
# Hệ số giá = nhân với DISTRICTS để điều chỉnh mặt bằng giá từng tỉnh
PROVINCES = [
    ("TP. Hồ Chí Minh", "TP.HCM",        1.00, 35),
    ("Hà Nội",           "Hà Nội",        0.92, 25),
    ("Đà Nẵng",          "Đà Nẵng",       0.60, 10),
    ("Bình Dương",       "Bình Dương",    0.45,  8),
    ("Đồng Nai",         "Đồng Nai",      0.38,  7),
    ("Cần Thơ",          "Cần Thơ",       0.30,  5),
    ("Khánh Hòa",        "Khánh Hòa",     0.40,  4),
    ("Hải Phòng",        "Hải Phòng",     0.38,  4),
    ("Long An",          "Long An",       0.25,  2),
]

# ─── Quận/huyện theo từng tỉnh ───────────────────────────────────────────────
# (tên_quận, giá_min/m², giá_max/m², trọng_số)  — đơn vị triệu VND/m²
# Giá sẽ được nhân thêm hệ số tỉnh (PROVINCES[2]) khi sinh dữ liệu
HN_DISTRICTS = [
    ("Hoàn Kiếm",    350, 550, 1),
    ("Ba Đình",      280, 450, 1),
    ("Đống Đa",      220, 380, 2),
    ("Cầu Giấy",     180, 320, 2),
    ("Hai Bà Trưng", 200, 350, 2),
    ("Thanh Xuân",   160, 280, 2),
    ("Nam Từ Liêm",  120, 220, 3),
    ("Bắc Từ Liêm",  100, 190, 3),
    ("Hà Đông",      110, 200, 3),
    ("Long Biên",    120, 210, 2),
    ("Gia Lâm",       85, 160, 2),
    ("Đông Anh",      80, 150, 2),
]

DN_DISTRICTS = [   # Đà Nẵng
    ("Hải Châu",        120, 220, 2),
    ("Sơn Trà",          90, 170, 2),
    ("Ngũ Hành Sơn",     80, 150, 2),
    ("Thanh Khê",        70, 130, 2),
    ("Liên Chiểu",       65, 120, 2),
    ("Cẩm Lệ",           60, 115, 2),
    ("Hòa Vang",         45,  90, 1),
]

BD_DISTRICTS = [   # Bình Dương
    ("Thuận An",         80, 140, 3),
    ("Dĩ An",            70, 130, 3),
    ("Thủ Dầu Một",      90, 160, 2),
    ("Bến Cát",          55, 100, 2),
    ("Tân Uyên",         50,  95, 2),
    ("Phú Giáo",         40,  80, 1),
]

DNA_DISTRICTS = [  # Đồng Nai
    ("Biên Hòa",         70, 130, 3),
    ("Long Khánh",       50,  95, 2),
    ("Nhơn Trạch",       55, 105, 2),
    ("Trảng Bom",        45,  90, 2),
    ("Long Thành",       50, 100, 2),
    ("Xuân Lộc",         35,  70, 1),
]

CT_DISTRICTS = [   # Cần Thơ
    ("Ninh Kiều",        60, 115, 3),
    ("Bình Thủy",        50,  95, 2),
    ("Cái Răng",         45,  90, 2),
    ("Ô Môn",            40,  80, 2),
    ("Thốt Nốt",         35,  70, 1),
]

KH_DISTRICTS = [   # Khánh Hòa
    ("Nha Trang",       100, 180, 3),
    ("Cam Ranh",         60, 110, 2),
    ("Ninh Hòa",         45,  90, 2),
    ("Diên Khánh",       40,  80, 1),
]

HP_DISTRICTS = [   # Hải Phòng
    ("Hồng Bàng",        70, 130, 2),
    ("Ngô Quyền",        75, 140, 2),
    ("Lê Chân",          65, 125, 2),
    ("Kiến An",          55, 100, 2),
    ("Hải An",           60, 115, 2),
    ("Đồ Sơn",           50,  95, 1),
]

LA_DISTRICTS = [   # Long An
    ("Tân An",           55, 100, 3),
    ("Bến Lức",          45,  85, 2),
    ("Đức Hòa",          40,  80, 2),
    ("Cần Giuộc",        40,  78, 2),
    ("Cần Đước",         35,  70, 1),
]

_PROVINCE_DISTRICTS: dict[str, list] = {
    "TP. Hồ Chí Minh": DISTRICTS,
    "Hà Nội":          HN_DISTRICTS,
    "Đà Nẵng":         DN_DISTRICTS,
    "Bình Dương":      BD_DISTRICTS,
    "Đồng Nai":        DNA_DISTRICTS,
    "Cần Thơ":         CT_DISTRICTS,
    "Khánh Hòa":       KH_DISTRICTS,
    "Hải Phòng":       HP_DISTRICTS,
    "Long An":         LA_DISTRICTS,
}


def _get_districts_for_province(prov_name: str) -> list:
    """Trả về danh sách quận/huyện chính xác cho từng tỉnh/thành."""
    return _PROVINCE_DISTRICTS.get(prov_name, DISTRICTS)


def _generate_one(rng: random.Random) -> dict:
    # ── Chọn tỉnh/thành ──
    prov_weights = [p[3] for p in PROVINCES]
    prov = rng.choices(PROVINCES, weights=prov_weights, k=1)[0]
    prov_name, city_str, price_factor, _ = prov

    # ── Chọn quận/huyện theo tỉnh ──
    districts_pool = _get_districts_for_province(prov_name)
    dist_weights = [d[3] for d in districts_pool]
    dist = rng.choices(districts_pool, weights=dist_weights, k=1)[0]
    dist_name, price_min_raw, price_max_raw, _ = dist

    # Điều chỉnh giá theo hệ số tỉnh
    price_min = price_min_raw * price_factor
    price_max = price_max_raw * price_factor

    # ── Chọn loại nhà ──
    type_weights = [h[10] for h in HOUSE_TYPES]
    ht = rng.choices(HOUSE_TYPES, weights=type_weights, k=1)[0]
    (house_type, type_factor_ht, a_min, a_max,
     bed_min, bed_max, bath_min, bath_max, fl_min, fl_max, _) = ht

    # ── Diện tích ──
    area = rng.uniform(a_min, a_max)

    # ── Phòng ngủ / phòng tắm / tầng ──
    bedrooms  = rng.randint(bed_min,  bed_max)  if bed_max  > 0 else 0
    bathrooms = rng.randint(bath_min, bath_max) if bath_max > 0 else 0
    floors    = rng.randint(fl_min,   fl_max)   if fl_max   > 0 else 0

    # ── Năm xây dựng (1975-2024) ──
    year_built = rng.randint(1975, 2024)
    age_factor = 1.0 + (year_built - 1990) * 0.002   # nhà mới hơn → giá cao hơn
    age_factor = max(0.85, min(age_factor, 1.20))

    # ── Chiều rộng mặt tiền (m) ──
    if house_type in ("nhà phố", "biệt thự"):
        facade_width = round(rng.uniform(3.5, 12.0), 1)
    elif house_type == "nhà hẻm":
        facade_width = round(rng.uniform(2.5, 5.0), 1)
    elif house_type == "căn hộ":
        facade_width = None
    else:
        facade_width = round(rng.uniform(4.0, 20.0), 1)
    facade_factor = (1.0 + (facade_width - 5.0) * 0.015) if facade_width else 1.0
    facade_factor = max(0.90, min(facade_factor, 1.35))

    # ── Pháp lý ──
    legal_choices = ["sổ đỏ", "sổ hồng", "giấy tay", "hợp đồng mua bán"]
    legal_weights = [40, 40, 10, 10]
    legal_status  = rng.choices(legal_choices, weights=legal_weights, k=1)[0]
    legal_factor  = {"sổ đỏ": 1.05, "sổ hồng": 1.03, "giấy tay": 0.88, "hợp đồng mua bán": 0.92}[legal_status]

    # ── Điều chỉnh theo tiện nghi ──
    feature_adj = 1.0
    feature_adj += bedrooms  * 0.04
    feature_adj += bathrooms * 0.025
    feature_adj += floors    * 0.015

    # price_per_m2 bien dong +-10% quanh trung vi cua quan
    # (thay vi uniform(min, max) +-25-33% -> model khong the hoc duoc)
    price_mid = (price_min + price_max) / 2.0
    price_per_m2 = rng.uniform(price_mid * 0.90, price_mid * 1.10)
    price = price_per_m2 * area * type_factor_ht * feature_adj * age_factor * facade_factor * legal_factor
    price *= rng.uniform(0.95, 1.05)   # noise +-5%
    price = max(200.0, price)

    # ── Format ──
    street   = rng.choice(STREETS)
    number   = rng.randint(1, 250)
    location = f"{number} {street}, {dist_name}, {city_str}"

    return {
        "price":        int(round(price * 1_000_000)),
        "area":         round(area, 1),
        "location":     location,
        "district":     dist_name,
        "bedrooms_n":   bedrooms  if bedrooms  > 0 else None,
        "bathrooms_n":  bathrooms if bathrooms > 0 else None,
        "floors_n":     floors    if floors    > 0 else None,
        "house_type":   house_type,
        "year_built":   year_built,
        "facade_width": facade_width,
        "legal_status": legal_status,
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
    parser.add_argument("--rows",   type=int, default=5000,
                        help="Số bản ghi cần tạo")
    parser.add_argument("--output", default="vietnam_house_raw.csv",
                        help="File CSV đầu ra")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed để tái lặp")
    args = parser.parse_args()

    generate(args.rows, Path(args.output), args.seed)
