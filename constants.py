#!/usr/bin/env python3
"""
Hằng số dùng chung giữa app.py và crawler.py.
Sửa alias tại đây sẽ có tác dụng cho toàn bộ project.
"""

# ─── Danh sách 63 tỉnh/thành hợp lệ ─────────────────────────────────────────
VALID_CITY_NAMES = [
    "An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh",
    "Bến Tre", "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau",
    "Cần Thơ", "Cao Bằng", "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên",
    "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội",
    "Hà Tĩnh", "Hải Dương", "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên",
    "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn",
    "Lào Cai", "Long An", "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận",
    "Phú Thọ", "Phú Yên", "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh",
    "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình", "Thái Nguyên",
    "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang", "TP. Hồ Chí Minh", "Trà Vinh",
    "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái",
]

# ─── Alias map (chuỗi không dấu → tên chuẩn) ─────────────────────────────────
# Dùng chung cho cả crawler.py và app.py
CITY_ALIAS: dict[str, str] = {
    # TP. Hồ Chí Minh
    "tphcm":                  "TP. Hồ Chí Minh",
    "tp hcm":                 "TP. Hồ Chí Minh",
    "tp.hcm":                 "TP. Hồ Chí Minh",
    "hcm":                    "TP. Hồ Chí Minh",
    "ho chi minh":            "TP. Hồ Chí Minh",
    "tp ho chi minh":         "TP. Hồ Chí Minh",
    "ho chi minh city":       "TP. Hồ Chí Minh",
    # Hà Nội
    "ha noi":                 "Hà Nội",
    "hanoi":                  "Hà Nội",
    # Đà Nẵng
    "da nang":                "Đà Nẵng",
    "danang":                 "Đà Nẵng",
    # Các tỉnh thành
    "can tho":                "Cần Thơ",
    "hai phong":              "Hải Phòng",
    "binh duong":             "Bình Dương",
    "dong nai":               "Đồng Nai",
    "bien hoa":               "Đồng Nai",
    "tp bien hoa":            "Đồng Nai",
    "thanh pho bien hoa":     "Đồng Nai",
    "bien ho":                "Đồng Nai",
    "khanh hoa":              "Khánh Hòa",
    "nha trang":              "Khánh Hòa",
    "long an":                "Long An",
    "ba ria vung tau":        "Bà Rịa - Vũng Tàu",
    "vung tau":               "Bà Rịa - Vũng Tàu",
    "binh thuan":             "Bình Thuận",
    "lam dong":               "Lâm Đồng",
    "da lat":                 "Lâm Đồng",
    "tay ninh":               "Tây Ninh",
    "binh phuoc":             "Bình Phước",
    "tien giang":             "Tiền Giang",
    "vinh long":              "Vĩnh Long",
    "an giang":               "An Giang",
    "kien giang":             "Kiên Giang",
    "ben tre":                "Bến Tre",
    "tra vinh":               "Trà Vinh",
    "hau giang":              "Hậu Giang",
    "soc trang":              "Sóc Trăng",
    "bac lieu":               "Bạc Liêu",
    "ca mau":                 "Cà Mau",
    "quang ninh":             "Quảng Ninh",
    "hai duong":              "Hải Dương",
    "hung yen":               "Hưng Yên",
    "bac ninh":               "Bắc Ninh",
    "vinh phuc":              "Vĩnh Phúc",
    "ha nam":                 "Hà Nam",
    "ninh binh":              "Ninh Bình",
    "thanh hoa":              "Thanh Hóa",
    "nghe an":                "Nghệ An",
    "ha tinh":                "Hà Tĩnh",
    "quang binh":             "Quảng Bình",
    "quang tri":              "Quảng Trị",
    "thua thien hue":         "Thừa Thiên Huế",
    "hue":                    "Thừa Thiên Huế",
    "quang nam":              "Quảng Nam",
    "quang ngai":             "Quảng Ngãi",
    "binh dinh":              "Bình Định",
    "phu yen":                "Phú Yên",
    "ninh thuan":             "Ninh Thuận",
    "gia lai":                "Gia Lai",
    "dak lak":                "Đắk Lắk",
    "dak nong":               "Đắk Nông",
    "kon tum":                "Kon Tum",
}

# ─── Giới hạn validation dữ liệu ────────────────────────────────────────────
MIN_PRICE_VND    = 100_000_000       # 100 triệu
MAX_PRICE_VND    = 200_000_000_000   # 200 tỷ
MAX_BEDROOMS     = 20
MAX_BATHROOMS    = 15
MAX_FLOORS       = 30
MAX_AREA_M2      = 2000
MIN_AREA_M2      = 10
