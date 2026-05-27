#!/usr/bin/env python3
"""
Dự đoán giá nhà bằng mô hình Bagging đã huấn luyện.

Sử dụng:
    python predict.py --area 80 --district "Quận 7"
    python predict.py --area 120 --district "Bình Thạnh" --bedrooms 3 --bathrooms 2 --floors 4 --house-type "nhà phố"
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def predict_single(model_path: str, sample: dict) -> float:
    d = joblib.load(model_path)
    pipe         = d["pipeline"]
    feature_cols = d.get("feature_cols", ["area_m2", "district"])
    use_log      = d.get("use_log_target", False)

    # Bổ sung district_log_median nếu model mới yêu cầu
    if "district_log_median" in feature_cols and "district_log_median" not in sample:
        district_log_map  = d.get("district_log_map", {})
        global_log_median = float(d.get("global_log_median", np.log1p(5000)))
        dist = sample.get("district", "unknown")
        sample["district_log_median"] = float(
            district_log_map.get(dist, global_log_median)
        )

    row = {col: sample.get(col, None) for col in feature_cols}
    df  = pd.DataFrame([row])[feature_cols]
    raw = float(pipe.predict(df)[0])
    return float(np.expm1(raw)) if use_log else raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dự đoán giá nhà với mô hình Bagging đã huấn luyện",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default="house_bagging_model.joblib",
                        help="File .joblib của mô hình")
    parser.add_argument("--area",       type=float, required=True,
                        help="Diện tích (m²)")
    parser.add_argument("--district",   type=str,   default="unknown",
                        help="Quận/huyện, ví dụ: 'Quận 7', 'Bình Thạnh'")
    parser.add_argument("--bedrooms",   type=float, default=None,
                        help="Số phòng ngủ")
    parser.add_argument("--bathrooms",  type=float, default=None,
                        help="Số phòng tắm")
    parser.add_argument("--floors",     type=float, default=None,
                        help="Số tầng")
    parser.add_argument("--house-type", type=str,   default="nha o",
                        help="Loại nhà: nha pho / can ho / biet thu / nha hem / dat nen")
    parser.add_argument("--city",       type=str,   default="unknown",
                        help="Thành phố")
    parser.add_argument("--province",   type=str,   default="unknown",
                        help="Tỉnh/Thành phố")
    args = parser.parse_args()

    # ── Tính price_per_m2 ước lượng (triệu/m²) ──────────────────────────────
    bundle = joblib.load(args.model)
    base_ppm2 = float(bundle.get("median_price_per_m2", args.area * 20))
    type_factors = {
        "nha pho": 1.08, "can ho": 0.92, "biet thu": 1.28,
        "nha hem": 0.86, "dat nen": 1.12, "nha tro": 0.72,
    }
    tf = type_factors.get(args.house_type.strip().lower(), 1.0)
    rooms_adj = 1.0
    if args.bedrooms:  rooms_adj += args.bedrooms  * 0.025
    if args.bathrooms: rooms_adj += args.bathrooms * 0.018
    if args.floors:    rooms_adj += args.floors    * 0.015
    price_per_m2 = base_ppm2 * tf * rooms_adj

    district_map = bundle.get("district_map", {})
    city_map = bundle.get("city_map", {})
    global_median = bundle.get("global_median_price", 5000)
    
    district_median = district_map.get(args.district, global_median)
    city_median = city_map.get(args.city, global_median)

    bedrooms = args.bedrooms if args.bedrooms else 0
    bathrooms = args.bathrooms if args.bathrooms else 0
    total_rooms = bedrooms + bathrooms
    area_per_room = args.area / max(total_rooms, 1)

    tier_map = {"TP.HCM": "tier1", "Hà Nội": "tier1",
                "Đà Nẵng": "tier2", "Hải Phòng": "tier2", "Cần Thơ": "tier2"}
    city_tier = tier_map.get(args.city, "tier3")

    sample = {
        "area_m2":     args.area,
        "log_area":    float(np.log1p(args.area)),
        "total_rooms": total_rooms,
        "area_per_room": area_per_room,
        "log_area_per_room": float(np.log1p(area_per_room)),
        "price_per_m2": price_per_m2,
        "district":    args.district,
        "district_median_price": district_median,
        "city":        args.city,
        "city_median_price": city_median,
        "province":    args.province,
        "city_tier":   city_tier,
        "bedrooms_n":  args.bedrooms if args.bedrooms else np.nan,
        "bathrooms_n": args.bathrooms if args.bathrooms else np.nan,
        "floors_n":    args.floors if args.floors else np.nan,
        "house_type":  args.house_type,
    }

    price = predict_single(args.model, sample)

    print(f"\n{'─'*48}")
    print(f"  Diện tích  : {args.area} m²")
    print(f"  Quận/Huyện : {args.district}")
    if args.bedrooms:   print(f"  Phòng ngủ  : {int(args.bedrooms)}")
    if args.bathrooms:  print(f"  Phòng tắm  : {int(args.bathrooms)}")
    if args.floors:     print(f"  Số tầng    : {int(args.floors)}")
    print(f"  Loại nhà   : {args.house_type}")
    print(f"{'─'*48}")
    print(f"  Giá dự đoán: {price:>10,.0f} triệu VND")
    print(f"             (~{price/1000:>8.3f} tỷ VND)")
    print(f"{'─'*48}\n")
