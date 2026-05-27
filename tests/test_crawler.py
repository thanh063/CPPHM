"""
Unit tests cho crawler helpers: parse_price, normalize_city, extract_district.
Dùng HTML mẫu tĩnh để test parse_mogi_page — không cần kết nối mạng.
"""
import pytest
from crawler import (
    _parse_price_to_vnd,
    _normalize_area_to_m2,
    _normalize_city_name,
    _extract_district_from_location,
    _infer_house_type,
    parse_mogi_page,
)


# ─── _parse_price_to_vnd ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("3 tỷ 500 triệu",  3_500_000_000),
    ("2.5 tỷ",          2_500_000_000),
    ("850 triệu",         850_000_000),
    ("1 tỷ",            1_000_000_000),
    ("thoả thuận",               None),
    ("liên hệ",                  None),
    ("",                         None),
    (None,                       None),
])
def test_parse_price_to_vnd(text, expected):
    assert _parse_price_to_vnd(text) == expected


# ─── _normalize_area_to_m2 ───────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("80 m²",    "80 m2"),
    ("120.5 m2", "120.5 m2"),
    ("55,5m2",   "55.5 m2"),
    ("",         ""),
    (None,       ""),
])
def test_normalize_area(text, expected):
    assert _normalize_area_to_m2(text) == expected


# ─── _normalize_city_name ────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("TPHCM",              "TP. Hồ Chí Minh"),
    ("TP.HCM",             "TP. Hồ Chí Minh"),
    ("Hồ Chí Minh",        "TP. Hồ Chí Minh"),
    ("Hà Nội",             "Hà Nội"),
    ("ha noi",             "Hà Nội"),
    ("Da Nang",            "Đà Nẵng"),
    ("Nha Trang",          "Khánh Hòa"),
    ("",                   "Khác"),
])
def test_normalize_city_name(raw, expected):
    assert _normalize_city_name(raw) == expected


# ─── _extract_district_from_location ─────────────────────────────────────────

@pytest.mark.parametrize("location, expected", [
    ("123 Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP. Hồ Chí Minh", "Quận 1"),
    ("456 Lê Lợi, Quận 3, TP. Hồ Chí Minh",                      "Quận 3"),
    ("Quận 7, TP. Hồ Chí Minh",                                   "Quận 7"),
    ("TP. Hồ Chí Minh",                                           ""),
    ("",                                                           ""),
])
def test_extract_district(location, expected):
    assert _extract_district_from_location(location) == expected


# ─── _infer_house_type ────────────────────────────────────────────────────────

@pytest.mark.parametrize("title, category, expected", [
    ("Biệt thự sang trọng Q7",   "biet-thu", "biệt thự"),
    ("Căn hộ 2PN tầng cao",      "can-ho",   "căn hộ"),
    ("Đất nền sổ đỏ Bình Chánh", "dat-nen",  "đất nền"),
    ("Nhà phố 5 tầng mới xây",   "nha-dat",  "nhà phố"),
    ("Nhà hẻm 1 trệt 1 lầu",     "nha-dat",  "nhà hẻm"),
    ("Bán nhà chính chủ",        "nha-dat",  "nhà ở"),
])
def test_infer_house_type(title, category, expected):
    assert _infer_house_type(title, category) == expected


# ─── parse_mogi_page với HTML mẫu ────────────────────────────────────────────

SAMPLE_MOGI_HTML = """
<html><body>
<div class="prop-info">
  <a class="link-overlay" href="/nha-dat/123-nguyen-hue-q1"></a>
  <h2 class="prop-title">Nhà phố đẹp 5 tầng Quận 1</h2>
  <div class="price">5 tỷ 200 triệu</div>
  <div class="prop-addr">123 Nguyễn Huệ, Quận 1, TP. Hồ Chí Minh</div>
  <ul class="prop-attr">
    <li>80 m²</li>
    <li>3 PN</li>
    <li>2 WC</li>
  </ul>
</div>
</body></html>
"""


def test_parse_mogi_page_returns_items():
    items = parse_mogi_page(SAMPLE_MOGI_HTML, "nha-dat")
    assert len(items) == 1
    item = items[0]
    assert item["price"] == 5_200_000_000
    assert item["area"] == "80 m2"
    assert item["bedrooms"] == "3"
    assert item["bathrooms"] == "2"
    assert item["house_type"] == "nhà phố"


def test_parse_mogi_page_empty_html():
    assert parse_mogi_page("<html><body></body></html>", "nha-dat") == []
