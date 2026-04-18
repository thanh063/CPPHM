#!/usr/bin/env python3
"""
Dự đoán giá nhà bằng mô hình Bagging đã huấn luyện.

Sử dụng:
    python predict.py --area 80 --district "Quận 7"
    python predict.py --area 120 --district "Bình Thạnh" --bedrooms 3 --bathrooms 2 --floors 4 --house-type "nhà phố"
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path


def predict_single(model_path: str, sample: dict) -> float:
    d = joblib.load(model_path)
    pipe         = d["pipeline"]
    feature_cols = d.get("feature_cols", ["area_m2", "district"])

    # Tạo DataFrame với đúng cột theo thứ tự model đã học
    row = {col: sample.get(col, None) for col in feature_cols}
    df  = pd.DataFrame([row])[feature_cols]
    return float(pipe.predict(df)[0])


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
    parser.add_argument("--house-type", type=str,   default="nhà ở",
                        help="Loại nhà: nhà phố / căn hộ / biệt thự / nhà hẻm / đất nền")
    args = parser.parse_args()

    sample = {
        "area_m2":    args.area,
        "district":   args.district,
        "bedrooms_n":  args.bedrooms,
        "bathrooms_n": args.bathrooms,
        "floors_n":    args.floors,
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
