#!/usr/bin/env python3
"""
Thu thập dữ liệu bất động sản Việt Nam từ mogi.vn.

Sử dụng:
    python crawler.py --pages 200 --output vietnam_house_raw.csv
    python crawler.py --pages 50 --categories nha-dat can-ho biet-thu --delay 2.5
"""

import argparse
import csv
import logging
import random
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from constants import CITY_ALIAS as _CITY_ALIAS_CRAWLER_IMPORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Hằng số ──────────────────────────────────────────────────────────────────
BASE_URL      = "https://mogi.vn"
BASE_URL_HMDY = "https://homedy.com"
BASE_URL_ALN  = "https://alonhadat.com.vn"

# mogi.vn categories
CATEGORIES_MOGI = {
    "nha-dat":  f"{BASE_URL}/mua-nha-dat",
    "can-ho":   f"{BASE_URL}/mua-can-ho-chung-cu",
    "biet-thu": f"{BASE_URL}/mua-nha-biet-thu-lien-ke",
    "dat-nen":  f"{BASE_URL}/mua-dat-nen-du-an",
    "nha-pho":  f"{BASE_URL}/mua-nha-mat-pho",
}

# homedy.com categories
CATEGORIES_HMDY = {
    "nha-dat":  f"{BASE_URL_HMDY}/ban-nha-dat",
    "can-ho":   f"{BASE_URL_HMDY}/ban-can-ho-chung-cu",
    "biet-thu": f"{BASE_URL_HMDY}/ban-biet-thu-lien-ke",
    "dat-nen":  f"{BASE_URL_HMDY}/ban-dat-nen",
    "nha-pho":  f"{BASE_URL_HMDY}/ban-nha-mat-pho",
}

# alonhadat.com.vn categories
CATEGORIES_ALN = {
    "nha-dat":  f"{BASE_URL_ALN}/can-ban-nha-dat",
    "can-ho":   f"{BASE_URL_ALN}/can-ban-can-ho-chung-cu",
    "biet-thu": f"{BASE_URL_ALN}/can-ban-biet-thu",
    "dat-nen":  f"{BASE_URL_ALN}/can-ban-dat",
    "nha-pho":  f"{BASE_URL_ALN}/can-ban-nha-mat-pho",
}

# batdongsan.com.vn
BASE_URL_BDS = "https://batdongsan.com.vn"
CATEGORIES_BDS = {
    "nha-dat":  f"{BASE_URL_BDS}/ban-nha-dat",
    "can-ho":   f"{BASE_URL_BDS}/ban-can-ho-chung-cu",
    "biet-thu": f"{BASE_URL_BDS}/ban-biet-thu-lien-ke",
    "dat-nen":  f"{BASE_URL_BDS}/ban-dat-nen-du-an",
    "nha-pho":  f"{BASE_URL_BDS}/ban-nha-rieng",
}

# nhatot.com / Chợ Tốt — dùng JSON API công khai (không bị chặn, không cần JS)
NHATOT_API_URL = "https://gateway.chotot.com/v1/public/ad-listing"

# Category codes Cho Tot
NHATOT_CATEGORIES = {
    "nha-dat":  1020,  # Nhà đất tổng hợp
    "can-ho":   1010,  # Căn hộ / Chung cư
    "biet-thu": 1040,  # Biệt thự / Nhà liền kề
    "dat-nen":  1030,  # Đất nền
    "nha-pho":  1020,  # Nhà riêng / nhà phố (dùng category tổng hợp)
}

# Region codes Cho Tot → tên tỉnh
NHATOT_REGIONS = {
    13000: "Tp Hồ Chí Minh",
    12000: "Hà Nội",
    52000: "Bình Dương",
    56000: "Đồng Nai",
    48000: "Đà Nẵng",
    62000: "Long An",
    60000: "Tiền Giang",
    58000: "Bà Rịa - Vũng Tàu",
    42000: "Khánh Hòa",
    64000: "Cần Thơ",
    31000: "Hải Phòng",
    38000: "Nghệ An",
    54000: "Lâm Đồng",
    40000: "Thừa Thiên Huế",
    50000: "Quảng Nam",
    66000: "An Giang",
    70000: "Khánh Hòa",
    0:     "Toàn quốc",
}

# Cho phép dùng key chung
CATEGORIES = CATEGORIES_MOGI  # dùng mogi làm default cho backward compat

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
    "Mozilla/5.0 (Android 14; Mobile; rv:124.0) Gecko/124.0 Firefox/124.0",
]

# ─── Schema thống nhất với generate_dataset.py ───────────────────────────────
CSV_FIELDS = [
    "price", "area", "location", "district",
    "bedrooms_n", "bathrooms_n", "floors_n", "house_type",
    "year_built", "facade_width", "legal_status",
]

# Validation thresholds (theo yêu cầu giảng viên)
MIN_PRICE_VND = 100_000_000      # 100 triệu VND
MAX_PRICE_VND = 200_000_000_000 # 200 tỷ VND
MAX_BEDROOMS = 20
MAX_BATHROOMS = 15
MAX_FLOORS = 30

# Counter for filtered records
_filtered_count = {"price_low": 0, "price_high": 0, "bedrooms": 0, "bathrooms": 0, "floors": 0}


def _validate_record(price, bedrooms, bathrooms, floors) -> bool:
    """
    Validate a record against business rules.
    Returns True if valid, False if should be filtered out.
    Also updates module-level counters for logging.
    Note: dict mutation does not need 'global' keyword.
    """
    
    # Validate price
    if price is not None:
        if price < MIN_PRICE_VND:
            _filtered_count["price_low"] += 1
            return False
        if price > MAX_PRICE_VND:
            _filtered_count["price_high"] += 1
            return False
    
    # Validate bedrooms
    if bedrooms:
        try:
            br = int(bedrooms)
            if br > MAX_BEDROOMS:
                _filtered_count["bedrooms"] += 1
                return False
        except (ValueError, TypeError):
            pass
    
    # Validate bathrooms
    if bathrooms:
        try:
            ba = int(bathrooms)
            if ba > MAX_BATHROOMS:
                _filtered_count["bathrooms"] += 1
                return False
        except (ValueError, TypeError):
            pass
    
    # Validate floors
    if floors:
        try:
            fl = int(floors)
            if fl > MAX_FLOORS:
                _filtered_count["floors"] += 1
                return False
        except (ValueError, TypeError):
            pass
    
    return True


def _log_filter_stats():
    """Log statistics about filtered records."""
    total = sum(_filtered_count.values())
    log.info("=== Validation Filter Stats ===")
    log.info("  Giá < 100 triệu:   %d", _filtered_count["price_low"])
    log.info("  Giá > 200 tỷ:       %d", _filtered_count["price_high"])
    log.info("  Phòng ngủ > 20:     %d", _filtered_count["bedrooms"])
    log.info("  Phòng tắm > 15:     %d", _filtered_count["bathrooms"])
    log.info("  Số tầng > 30:       %d", _filtered_count["floors"])
    log.info("  Tổng bị loại:       %d", total)
    log.info("================================")


# ─── HTTP helpers ─────────────────────────────────────────────────────────────
def _headers(url: str | None = None) -> dict:
    referer = BASE_URL
    if url:
        try:
            p = urlparse(url)
            if p.scheme and p.netloc:
                referer = f"{p.scheme}://{p.netloc}"
        except Exception:
            pass

    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
        "Referer": referer,
    }


def _is_robot_page(html: str) -> bool:
    lo = html.lower()
    return (
        "xác minh không phải robot" in lo
        or "xac minh khong phai robot" in lo
        or "verify you are human" in lo
    )


def _extract_region_links_homedy(html: str, base_url: str) -> list[str]:
    """Lấy các trang theo tỉnh/thành của Homedy từ trang category."""
    soup = BeautifulSoup(html, "html.parser")
    base_path = urlparse(base_url).path.strip("/")
    links: set[str] = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href or href.startswith("javascript:"):
            continue
        abs_url = href if href.startswith("http") else (BASE_URL_HMDY + href)
        p = urlparse(abs_url)
        path = p.path.strip("/")

        # Ví dụ: /ban-nha-dat-ha-noi
        # Lọc chẩn: path mở rộng từ base_path, không có subdirectory,
        # không phải trang phân trang (không chứa /p\d+ hoặc 'page')
        if (
            path.startswith(base_path + "-")
            and "/" not in path
            and not re.search(r"(?:/p\d+|page|trang)$", path)
        ):
            links.add(f"{p.scheme}://{p.netloc}/{path}")

    return sorted(links)


def _extract_region_links_alonhadat(html: str, base_url: str) -> list[str]:
    """Lấy các trang theo tỉnh/thành của Alonhadat từ trang category."""
    soup = BeautifulSoup(html, "html.parser")
    base_path = urlparse(base_url).path.strip("/")
    links: set[str] = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href or href.startswith("javascript:"):
            continue
        abs_url = href if href.startswith("http") else (BASE_URL_ALN + href)
        p = urlparse(abs_url)
        path = p.path.strip("/")

        # Ví dụ: /can-ban-nha-dat/phu-yen
        if not path.startswith(base_path + "/"):
            continue
        suffix = path[len(base_path) + 1:]
        if not suffix:
            continue
        if "trang-" in suffix or suffix.endswith(".html") or "/" in suffix:
            continue
        links.add(f"{p.scheme}://{p.netloc}/{path}")

    return sorted(links)



# Các lỗi DNS / mất mạng — phân biệt với lỗi server (4xx/5xx)
_NET_ERROR_KEYWORDS = (
    "getaddrinfo failed",
    "nameresolution",
    "name or service not known",
    "failed to resolve",
    "network is unreachable",
    "connection timed out",
    "connect timeout",
    "connection refused",
    "remotedisconnected",
    "errno 11001",
    "errno 111",
    "errno 110",
)


def _is_network_error(exc: Exception) -> bool:
    """Kiểm tra xem lỗi có phải mất internet/DNS không (khác với lỗi server 4xx)."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _NET_ERROR_KEYWORDS)


def _wait_for_internet(check_url: str = "https://mogi.vn", interval: int = 15) -> None:
    """
    Chờ cho đến khi internet phục hồi.
    - Hiển thị bộ đếm ngược sau mỗi lần thử thất bại.
    - Tự động tiếp tục khi kết nối được phục hồi.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            import socket
            socket.setdefaulttimeout(8)
            socket.getaddrinfo("mogi.vn", 443)
            log.info("✅ Internet đã phục hồi! Tiếp tục crawl...")
            return
        except OSError:
            pass

        wait_sec = min(interval * attempt, 120)   # tăng dần, tối đa 2 phút
        for remaining in range(wait_sec, 0, -5):
            log.warning(
                "⚠️  Mất internet — thử lại sau %ds  (lần %d)...", remaining, attempt
            )
            time.sleep(5)


def _get(session: requests.Session, url: str, retries: int = 3) -> str | None:
    """GET với retry + exponential back-off.
    
    Khi gặp lỗi mạng/DNS: tự động chờ internet phục hồi rồi thử lại.
    Khi gặp lỗi server (4xx/5xx): retry thông thường rồi trả None.
    """
    attempt = 0
    while attempt < retries:
        try:
            r = session.get(url, headers=_headers(url), timeout=20, allow_redirects=True)
            if r.status_code == 200:
                if _is_robot_page(r.text):
                    wait = 5 * (attempt + 1)
                    log.warning("Bị chặn robot-check: %s | chờ %ds rồi thử lại", url, wait)
                    time.sleep(wait)
                    attempt += 1
                    continue
                return r.text
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("Rate-limited (429). Chờ %ds...", wait)
                time.sleep(wait)
                attempt += 1
                continue
            log.warning("HTTP %s – %s (lần %d)", r.status_code, url, attempt + 1)
            attempt += 1
            time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as exc:
            if _is_network_error(exc):
                # Mất internet → chờ phục hồi rồi thử lại (không tính vào retries)
                log.warning("🔌 Mất kết nối mạng khi tải: %s", url)
                _wait_for_internet()
                # Reset attempt để thử lại URL này từ đầu
                attempt = 0
            else:
                log.warning("Lỗi request %s (lần %d): %s", url, attempt + 1, exc)
                attempt += 1
                time.sleep(2 ** attempt)

    return None




# ─── Parse helpers ─────────────────────────────────────────────────────────────
def _parse_price_to_vnd(text: str) -> float | None:
    """Chuyển chuỗi giá tiền Việt Nam → số đồng VND.

    Ví dụ:
        "3 tỷ 500 triệu" → 3_500_000_000
        "2.5 tỷ"         → 2_500_000_000
        "850 triệu"      → 850_000_000
    """
    if not text:
        return None
    t = text.strip().lower().replace("\xa0", " ")
    if t in ("thoả thuận", "thỏa thuận", "liên hệ", ""):
        return None
    try:
        def parse_part(part_str):
            if not part_str:
                return 0.0
            ps = part_str.replace(" ", "")
            if "," in ps and "." not in ps:
                ps = ps.replace(",", ".")
            elif "," in ps and "." in ps:
                ps = ps.replace(".", "").replace(",", ".")
            try:
                return float(ps)
            except ValueError:
                return 0.0

        if "tỷ" in t:
            ty_m = re.search(r"([\d,.]+)\s*tỷ", t)
            tr_m = re.search(r"tỷ.*?([\d,.]+)\s*triệu", t)
            ty_val = parse_part(ty_m.group(1)) if ty_m else 0.0
            tr_val = parse_part(tr_m.group(1)) if tr_m else 0.0
            return round(ty_val * 1_000_000_000 + tr_val * 1_000_000, 0)
        if any(x in t for x in ("triệu", "trieu")):
            m = re.search(r"([\d,.]+)", t)
            if m:
                return round(parse_part(m.group(1)) * 1_000_000, 0)
    except (ValueError, AttributeError):
        pass
    return None


def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def _first_number(text: str) -> str:
    m = re.search(r"\d+", text)
    return m.group() if m else ""


def _first_float(text: str) -> str:
    """Trich xuat so thuc dau tien (ho tro ca dang x.y)."""
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    return m.group().replace(",", ".") if m else ""


def _extract_fields_from_text(text: str) -> dict:
    """
    Trich xuat cac truong bo sung tu tieu de / mo ta bang regex.
    Hoat dong tren ca listing page lan detail page.
    """
    t = text.lower()
    result: dict = {}

    # --- So tang ---
    m = re.search(r"(\d+)\s*tang", t)
    if m:
        fl = int(m.group(1))
        if 1 <= fl <= 30:
            result["floors_n"] = str(fl)

    # --- Mat tien (m) ---
    m = re.search(r"mat\s*tien[^\d]*(\d+(?:[.,]\d+)?)\s*m", t)
    if not m:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*m\s*mat\s*tien", t)
    if m:
        fw = float(m.group(1).replace(",", "."))
        if 2.0 <= fw <= 50.0:
            result["facade_width"] = str(fw)

    # --- Nam xay dung (chi lay khi co context ro rang) ---
    m = re.search(r"(?:nam\s*xay|xay\s*dung|xay\s*nam|built|year)[^\d]{0,15}((?:19[7-9]|200|201|202)\d)", t)
    if m:
        result["year_built"] = m.group(1)
    # Khong dung fallback tim nam tuy tien vi se bat nham ngay dang tin

    # --- Phap ly ---
    legal_map = [
        (r"so\s*do",                "so do"),
        (r"so\s*hong",              "so hong"),
        (r"giay\s*tay",             "giay tay"),
        (r"hop\s*dong\s*mua\s*ban", "hop dong mua ban"),
        (r"chu\s*quyen",            "so hong"),
        (r"phap\s*ly.*day\s*du",    "so hong"),
    ]
    txt_no_accent = _strip_accents(t)
    for pat, label in legal_map:
        if re.search(pat, txt_no_accent):
            result["legal_status"] = label
            break

    return result


def _normalize_area_to_m2(text: str) -> str:
    """Chuẩn hoá diện tích về dạng 'x m2'."""
    if text is None:
        return ""

    t = str(text).strip().lower().replace("\xa0", " ")
    if not t:
        return ""

    m = re.search(r"(\d+(?:[\.,]\d+)?)", t)
    if not m:
        return ""

    raw = m.group(1).replace(",", ".")
    try:
        val = float(raw)
    except ValueError:
        return ""

    if val.is_integer():
        num = str(int(val))
    else:
        num = (f"{val:.2f}").rstrip("0").rstrip(".")

    return f"{num} m2"


def _strip_accents(text: str) -> str:
    """Loại bỏ dấu thanh, chuyển về ASCII thường."""
    s = str(text or "").replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _clean_location_segment(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# Import alias từ constants.py — Single Source of Truth (không định nghĩa lại)
_CITY_ALIAS_CRAWLER: dict[str, str] = _CITY_ALIAS_CRAWLER_IMPORT




def _normalize_city_name(text: str) -> str:
    """Chuẩn hóa tên tỉnh/thành phố từ chuỗi địa chỉ."""
    raw = _clean_location_segment(text)
    if not raw:
        return "Khác"

    key = _strip_accents(raw).lower()
    key = re.sub(r"[^a-z0-9\s]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()

    if key in _CITY_ALIAS_CRAWLER:
        return _CITY_ALIAS_CRAWLER[key]

    # Thử strip tiền tố "thanh pho", "tp.", "tinh"
    key2 = re.sub(r"^(?:thanh pho|tp\.?|tinh)\s+", "", key).strip()
    if key2 in _CITY_ALIAS_CRAWLER:
        return _CITY_ALIAS_CRAWLER[key2]

    # Chuẩn hóa viết tắt phổ biến trong raw string
    result = raw
    for src, dst in [
        ("TPHCM", "TP. Hồ Chí Minh"),
        ("TpHCM", "TP. Hồ Chí Minh"),
        ("TP HCM", "TP. Hồ Chí Minh"),
        ("TP.HCM", "TP. Hồ Chí Minh"),
        ("Biên Hoà", "Biên Hòa"),
    ]:
        result = result.replace(src, dst)
    if "Hồ Chí Minh" in result and "TP. Hồ Chí Minh" not in result:
        result = result.replace("Hồ Chí Minh", "TP. Hồ Chí Minh")
    return result


def _normalize_location(text: str) -> str:
    raw = _clean_location_segment(text)
    if not raw:
        return ""

    def _strip_city_mentions(seg: str, city_name: str) -> str:
        s = _clean_location_segment(seg)
        if not s:
            return ""

        patterns: list[str]
        if city_name == "TP. Hồ Chí Minh":
            patterns = [
                r"TP\.?\s*Hồ\s*Chí\s*Minh",
                r"TP\.?\s*HCM",
                r"TPHCM",
                r"Hồ\s*Chí\s*Minh",
                r"\bHCM\b",
            ]
        elif city_name == "Biên Hòa":
            patterns = [
                r"TP\.?\s*Biên\s*Hòa",
                r"Biên\s*Hoà",
                r"Biên\s*Hòa",
            ]
        elif city_name == "Hà Nội":
            patterns = [r"TP\.?\s*Hà\s*Nội", r"Hà\s*Nội"]
        elif city_name == "Đà Nẵng":
            patterns = [r"TP\.?\s*Đà\s*Nẵng", r"Đà\s*Nẵng"]
        elif city_name == "Cần Thơ":
            patterns = [r"TP\.?\s*Cần\s*Thơ", r"Cần\s*Thơ"]
        elif city_name == "Hải Phòng":
            patterns = [r"TP\.?\s*Hải\s*Phòng", r"Hải\s*Phòng"]
        else:
            patterns = [re.escape(city_name)]

        for pat in patterns:
            s = re.sub(pat, " ", s, flags=re.I)
        s = re.sub(r"(?:TP\.\s*){2,}", "TP. ", s)
        s = re.sub(r"\s+", " ", s).strip(" ,")
        return s

    def _canon_segment(seg: str) -> str:
        s = _clean_location_segment(seg)
        if not s:
            return ""
        s = s.replace("TPHCM", "TP. Hồ Chí Minh")
        s = s.replace("TpHCM", "TP. Hồ Chí Minh")
        s = s.replace("TP HCM", "TP. Hồ Chí Minh")
        s = s.replace("TP.HCM", "TP. Hồ Chí Minh")
        if "Hồ Chí Minh" in s and "TP. Hồ Chí Minh" not in s:
            s = s.replace("Hồ Chí Minh", "TP. Hồ Chí Minh")
        if "Biên Hoà" in s and "Biên Hòa" not in s:
            s = s.replace("Biên Hoà", "Biên Hòa")
        s = s.replace("Biên Hoà", "Biên Hòa")
        s = s.replace("Thành phố", "TP.")
        s = re.sub(r"^TP\s+(?=\S)", "TP. ", s)
        s = re.sub(r"^(?:TP\.\s*)+", "TP. ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    parts = [_canon_segment(p) for p in raw.split(",") if _canon_segment(p)]
    if not parts:
        return ""

    city = _normalize_city_name(parts[-1])
    middle = parts[:-1]

    # Remove duplicated city fragments from the middle section.
    cleaned_middle = []
    city_key = _strip_accents(city).lower()
    for part in middle:
        part = _canon_segment(part)
        pkey = _strip_accents(part).lower()
        if not pkey:
            continue
        if pkey == city_key or city_key in pkey:
            continue
        stripped = _strip_city_mentions(part, city)
        if not stripped:
            continue
        stripped_key = _strip_accents(stripped).lower()
        if stripped_key == city_key:
            continue
        cleaned_middle.append(stripped)

    normalized_parts = cleaned_middle + [city]
    normalized_parts = [p for p in normalized_parts if p]
    normalized = ", ".join(normalized_parts)
    normalized = re.sub(r"(?:TP\.\s*){2,}", "TP. ", normalized)
    normalized = re.sub(r"(?:,\s*){2,}", ", ", normalized)
    normalized = normalized.replace("Đường TP. Hồ Chí Minh", "Đường")
    normalized = normalized.replace("Duong TP. Ho Chi Minh", "Duong")
    normalized = normalized.replace("Đường TP. TP. Hồ Chí Minh", "Đường")
    normalized = normalized.replace("Duong TP. TP. Ho Chi Minh", "Duong")
    normalized = re.sub(r"\s+,", ",", normalized)
    normalized = re.sub(r",\s+", ", ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_district_from_location(location: str) -> str:
    """Trích xuất quận/huyện từ chuỗi địa chỉ đã chuẩn hóa.

    Địa chỉ có dạng: "<đường>, <phường>, <quận>, <tỉnh>"
    Hoặc: "<quận/huyện>, <tỉnh>"
    Phần thứ 2 từ cuối thường là quận/huyện.
    """
    if not location:
        return ""
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if len(parts) >= 3:
        return parts[-2]   # quận/huyện = phần áp chót
    if len(parts) == 2:
        return parts[0]    # chỉ có "quận, tỉnh" → lấy quận
    return ""


def _infer_house_type(title: str, category: str) -> str:
    t = title.lower()
    if "biệt thự" in t or "villa" in t:
        return "biệt thự"
    if "căn hộ" in t or "chung cư" in t or "apartment" in t:
        return "căn hộ"
    if "đất nền" in t or "lô đất" in t or "đất thổ cư" in t:
        return "đất nền"
    if "nhà phố" in t or "townhouse" in t:
        return "nhà phố"
    if "nhà hẻm" in t:
        return "nhà hẻm"
    # fallback từ category
    _map = {"can-ho": "căn hộ", "biet-thu": "biệt thự",
            "dat-nen": "đất nền", "nha-pho": "nhà phố"}
    return _map.get(category, "nhà ở")


def has_next_page_mogi(html: str) -> bool:
    """Kiểm tra còn trang tiếp theo không (mogi.vn)."""
    soup = BeautifulSoup(html, "html.parser")
    nxt = soup.select_one('div.paging a[gtm-act="next"]')
    return nxt is not None


def has_next_page_homedy(html: str) -> bool:
    """Kiểm tra còn trang tiếp theo không (homedy.com)."""
    soup = BeautifulSoup(html, "html.parser")
    nxt = soup.select_one(
        "a.next, li.next:not(.disabled) a, .pagination .next a, "
        "a[rel='next'], a[href*='/p2'], a[href*='/p3']"
    )
    return nxt is not None


def has_next_page_alonhadat(html: str) -> bool:
    """Kiểm tra còn trang tiếp theo không (alonhadat.com.vn).

    Phân trang alonhadat có dạng: .page a[href*='trang-']
    Trang cuối vẫn có link ".page a" (liên kết trang trước) nên
    chỉ kiểm tra `.page a` là điều kiện cần, chưa đủ.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Tìm link Next rõ ràng
    nxt = soup.select_one("a.next-page, a[rel='next'], .paging a.active + a[href]")
    if nxt:
        return True
    # Kiểm tra có link trang phân trang có href chứa 'trang-' không
    page_links = soup.select(".page a[href*='trang-'], .pagination a[href*='trang-']")
    if not page_links:
        return False
    # Lấy số trang lớn nhất trong các link phân trang
    max_pg = 0
    cur_pg = 1
    for a in page_links:
        href = a.get("href", "")
        m = re.search(r"trang-(\d+)", href)
        if m:
            max_pg = max(max_pg, int(m.group(1)))
    # Tìm trang hiện tại
    cur_el = soup.select_one(".page .current, .page .active, .page strong, .pagination .active")
    if cur_el:
        try:
            cur_pg = int(_first_number(cur_el.get_text()))
        except ValueError:
            pass
    return max_pg > cur_pg


def _parse_mogi_detail(html: str) -> dict:
    """
    Phan tich trang chi tiet mogi.vn, trich xuat cac truong tu div.info-attr.
    Tra ve dict voi cac key: floors_n, year_built, facade_width, legal_status.
    """
    soup = BeautifulSoup(html, "html.parser")
    result: dict = {}

    # Map tu label tieng Viet -> field name
    LABEL_MAP = {
        "so tang":        "floors_n",
        "tang":           "floors_n",
        "mat tien":       "facade_width",
        "chieu ngang":    "facade_width",
        "nam xay dung":   "year_built",
        "nam xay":        "year_built",
        "phap ly":        "legal_status",
        "phap li":        "legal_status",
        "huong nha":      "direction",
        "huong cua":      "direction",
    }

    for attr_div in soup.select("div.info-attr"):
        raw = attr_div.get_text(" ", strip=True)
        # raw vi du: "Phap ly So hong" hoac "So tang 4"
        raw_norm = _strip_accents(raw).lower()

        matched_field = None
        matched_value = None
        for label_key, field in LABEL_MAP.items():
            if raw_norm.startswith(label_key):
                # Value la phan con lai sau label
                matched_field = field
                matched_value = raw_norm[len(label_key):].strip()
                break

        if not matched_field or not matched_value:
            continue

        if matched_field == "floors_n":
            num = _first_number(matched_value)
            if num and 1 <= int(num) <= 30:
                result["floors_n"] = num

        elif matched_field == "facade_width":
            num = _first_float(matched_value)
            if num:
                fw = float(num)
                if 2.0 <= fw <= 50.0:
                    result["facade_width"] = str(fw)

        elif matched_field == "year_built":
            m = re.search(r"((?:19|20)\d{2})", matched_value)
            if m:
                yr = int(m.group(1))
                if 1950 <= yr <= 2026:
                    result["year_built"] = str(yr)

        elif matched_field == "legal_status":
            legal_map = {
                "so do":             "so do",
                "so hong":           "so hong",
                "giay tay":          "giay tay",
                "hop dong mua ban":  "hop dong mua ban",
                "chu quyen":         "so hong",
            }
            for k, v in legal_map.items():
                if k in matched_value:
                    result["legal_status"] = v
                    break
            else:
                # Giu nguyen text ngan
                result["legal_status"] = matched_value[:40].strip()

    # Chi goi fallback text neu khong lay duoc bat ky truong nao tu info-attr
    # (tranh false positive tu ngay dang tin)
    if not result:
        page_text = soup.get_text(" ", strip=True)
        extras = _extract_fields_from_text(page_text)
        # Chi lay floors_n va facade_width tu text (khong lay year_built)
        for k in ("floors_n", "facade_width", "legal_status"):
            if k in extras:
                result.setdefault(k, extras[k])

    return result




def parse_homedy_page(html: str, category: str) -> list[dict]:
    """
    Phan tich HTML trang ket qua homedy.com.
    Co bo sung trich xuat regex tu title de lay floors/legal.
    """
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict] = []

    cards = soup.select("div.product-item, article.product-item, div.realty-item")
    if not cards:
        cards = soup.select("div[class*='product'] article, ul.product-list > li")

    for card in cards:
        a_tag = card.find("a", href=True)
        if not a_tag:
            continue
        href = a_tag.get("href", "")
        url  = href if href.startswith("http") else (BASE_URL_HMDY + href)

        title_el = card.select_one("h2, h3, .product-title, .realty-title")
        title = _text(title_el) or a_tag.get("title", "")
        if not title:
            continue

        price_el = card.select_one(
            ".product-price, .realty-price, .price-value, "
            ".price-number, span.gia, strong.price, [data-price]"
        )
        if not price_el:
            price_el = card.select_one(".price, [class*='price']")
        price = _parse_price_to_vnd(_text(price_el)) if price_el else None
        if price is None:
            price = _parse_price_to_vnd(card.get_text(" "))

        loc_el = card.select_one(
            ".product-short-description, .product-location, "
            ".address, [class*='location'], [class*='address']"
        )
        location = _normalize_location(_text(loc_el))

        area_el = card.select_one("[class*='area'], [class*='dien-tich']")
        area_txt = _text(area_el)
        if not area_txt:
            m = re.search(r"[\d,.]+\s*m\s*[2²]", card.get_text(" "))
            area_txt = m.group(0) if m else ""
        area_txt = _normalize_area_to_m2(area_txt)

        bedrooms = bathrooms = ""
        bed_el  = card.select_one("[class*='bed'], [class*='pn'], [title*='ngủ']")
        bath_el = card.select_one("[class*='bath'], [class*='wc'], [title*='tắm']")
        if bed_el:  bedrooms  = _first_number(_text(bed_el))
        if bath_el: bathrooms = _first_number(_text(bath_el))

        house_type = _infer_house_type(title, category)
        district   = _extract_district_from_location(location)

        # Trich xuat bo sung tu tieu de
        extras = _extract_fields_from_text(title + " " + card.get_text(" "))

        if not _validate_record(price, bedrooms, bathrooms, extras.get("floors_n", "")):
            continue

        items.append({
            "price":        price,
            "area":         area_txt,
            "location":     location,
            "district":     district,
            "bedrooms_n":   bedrooms,
            "bathrooms_n":  bathrooms,
            "floors_n":     extras.get("floors_n", ""),
            "house_type":   house_type,
            "year_built":   extras.get("year_built", ""),
            "facade_width": extras.get("facade_width", ""),
            "legal_status": extras.get("legal_status", ""),
        })

    return items


def parse_alonhadat_page(html: str, category: str) -> list[dict]:
    """Phan tich HTML trang ket qua alonhadat.com.vn, co bo sung regex."""
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict] = []

    cards = soup.select("article.property-item")
    for card in cards:
        a_tag = card.select_one("a.link[href]")
        if not a_tag:
            continue

        title = _text(a_tag)
        if not title:
            continue

        href = a_tag.get("href", "")
        url = href if href.startswith("http") else (BASE_URL_ALN + href)

        price_txt = _text(card.select_one(".price"))
        price = _parse_price_to_vnd(price_txt)
        if price is None:
            price = _parse_price_to_vnd(card.get_text(" "))

        area = _normalize_area_to_m2(_text(card.select_one(".area")))
        location = _normalize_location(
            _text(card.select_one(".new-address")) or
            _text(card.select_one(".property-address"))
        )
        bedrooms  = _first_number(_text(card.select_one(".bedroom")))
        floors    = _first_number(_text(card.select_one(".floors")))
        house_type = _infer_house_type(title, category)
        district   = _extract_district_from_location(location)

        # Bo sung tu regex tren title + card text
        extras = _extract_fields_from_text(title + " " + card.get_text(" "))
        if not floors:
            floors = extras.get("floors_n", "")

        if not _validate_record(price, bedrooms, "", floors):
            continue

        items.append({
            "price":        price,
            "area":         area,
            "location":     location,
            "district":     district,
            "bedrooms_n":   bedrooms,
            "bathrooms_n":  "",
            "floors_n":     floors,
            "house_type":   house_type,
            "year_built":   extras.get("year_built", ""),
            "facade_width": extras.get("facade_width", ""),
            "legal_status": extras.get("legal_status", ""),
        })

    return items


# ─── Parser batdongsan.com.vn ────────────────────────────────────────────────────
def parse_batdongsan_page(html: str, category: str) -> list[dict]:
    """Phân tích HTML trang kết quả batdongsan.com.vn."""
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict] = []

    # batdongsan.com.vn dùng class `js__product-link-for-product-id` hoặc div.product-item
    cards = soup.select("div.re__pr-card, div.pr-card, div.js__card, div.product-item")
    if not cards:
        # fallback: tìm thẻ li chứa link bất động sản
        cards = soup.select("ul.product-list > li, div[id^='product_id_']")

    for card in cards:
        a_tag = card.select_one("a[href*='/mua-'], a[href*='/ban-'], a.js__product-link-for-product-id")
        if not a_tag:
            a_tag = card.find("a", href=re.compile(r"/[a-z0-9-]+-pr\d+\.html"))
        if not a_tag:
            a_tag = card.find("a", href=True)
        if not a_tag:
            continue

        href = a_tag.get("href", "")
        url  = href if href.startswith("http") else (BASE_URL_BDS + href)

        title_el = card.select_one("span.product-title, h3, h2, .js__card-title")
        title = _text(title_el) or a_tag.get("title", "")
        if not title:
            title = _text(a_tag)
        if not title:
            continue

        # Giá — batdongsan thường là "3 tỷ", "1.5 tỷ", "850 triệu"
        price_el = card.select_one(
            ".re__card-config-price, span.product-price, div.product-price, .re__product-price, "
            "span.gia-ban, strong[class*='price'], [data-price]"
        )
        price_txt = _text(price_el)
        if not price_txt:
            price_txt = card.get_text(" ")
        price = _parse_price_to_vnd(price_txt)

        # Diện tích
        area_el = card.select_one(
            ".re__card-config-area, span.product-area, .re__product-area, span[class*='area'], "
            "span[class*='dien-tich'], td.area"
        )
        area_txt = _text(area_el)
        if not area_txt:
            m = re.search(r"[\d]+(?:[.,][\d]+)?\s*m[2²]", card.get_text(" "))
            area_txt = m.group(0) if m else ""
        area_txt = _normalize_area_to_m2(area_txt)

        # Địa chỉ
        loc_el = card.select_one(
            ".re__card-location, div.product-location, span.product-location, .re__product-locality, "
            "div[class*='address'], span[class*='addr'], td.location"
        )
        loc_raw = _text(loc_el).strip().lstrip("·").strip()
        location = _normalize_location(loc_raw)

        # Phòng ngủ / phòng tắm
        bedrooms = bathrooms = ""
        bed_el  = card.select_one(
            ".re__card-config-bedroom, [class*='bed'], [class*='phong-ngu'], span[title*='ngủ']"
        )
        bath_el = card.select_one(
            ".re__card-config-toilet, [class*='bath'], [class*='wc'], span[title*='tắm']"
        )
        if bed_el:  bedrooms  = _first_number(_text(bed_el))
        if bath_el: bathrooms = _first_number(_text(bath_el))

        house_type = _infer_house_type(title, category)
        district   = _extract_district_from_location(location)
        extras     = _extract_fields_from_text(title + " " + card.get_text(" "))

        if not _validate_record(price, bedrooms, bathrooms, extras.get("floors_n", "")):
            continue

        items.append({
            "price":        price,
            "area":         area_txt,
            "location":     location,
            "district":     district,
            "bedrooms_n":   bedrooms,
            "bathrooms_n":  bathrooms,
            "floors_n":     extras.get("floors_n", ""),
            "house_type":   house_type,
            "year_built":   extras.get("year_built", ""),
            "facade_width": extras.get("facade_width", ""),
            "legal_status": extras.get("legal_status", ""),
            "_url":         url,
            "category":     category,
        })

    return items


def has_next_page_batdongsan(html: str, current_page: int = 1) -> bool:
    """Kiểm tra có trang tiếp theo trên batdongsan.com.vn không."""
    soup = BeautifulSoup(html, "html.parser")
    # Kiểm tra xem có liên kết dẫn tới trang tiếp theo không (ví dụ: /p2, /p3)
    next_page_suffix = f"/p{current_page + 1}"
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if href.endswith(next_page_suffix) or f"{next_page_suffix}?" in href or f"{next_page_suffix}/" in href:
            return True
            
    nxt = soup.select_one(
        "a.re__pagination-page-next:not(.disabled), "
        "a[rel='next'], li.next:not(.disabled) a, a.re__pagination-icon"
    )
    if nxt:
        return True
    # Fallback: tìm nút "Trang sau" hoặc số trang
    for a in soup.select("a.pagination-item, a[class*='page']"):
        txt = _text(a).strip()
        if txt.lower() in (">", "»", "trang sau", "next", "sau"):
            parent = a.find_parent("li")
            if parent and "disabled" not in parent.get("class", []):
                return True
    return False


# ─── Parser / Crawler nhatot.com (Chợ Tốt) qua JSON API ────────────────────
def parse_nhatot_api_page(data: dict, category: str) -> list[dict]:
    """Phân tích JSON trả về từ gateway.chotot.com và chuyển sang schema CSV chuẩn."""
    items: list[dict] = []
    ads = data.get("ads", [])

    for ad in ads:
        price = ad.get("price")
        if not price:
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        area = ad.get("area") or ad.get("size")
        if not area:
            continue
        try:
            area = float(area)
        except (TypeError, ValueError):
            continue

        region   = ad.get("region_name", "") or NHATOT_REGIONS.get(ad.get("region_v2", 0), "")
        district = ad.get("area_name", "")
        ward     = ad.get("ward_name", "")
        street   = ad.get("street_name", "")
        loc_parts = [p for p in [street, ward, district, region] if p]
        location = ", ".join(loc_parts) if loc_parts else region

        bedrooms = bathrooms = floors = facade = legal = ""
        for p in ad.get("params", []):
            pid = p.get("id", "")
            val = str(p.get("value", "")).strip()
            if pid in ("bedroom", "bedrooms"):
                bedrooms = _first_number(val)
            elif pid in ("bathroom", "bathrooms", "wc"):
                bathrooms = _first_number(val)
            elif pid in ("floors", "floor"):
                floors = _first_number(val)
            elif pid in ("width", "mat_tien", "facade"):
                facade = _first_float(val)
            elif pid in ("property_legal_document",):
                legal = val[:40]

        title      = ad.get("subject", "")
        house_type = _infer_house_type(title, category)

        if not _validate_record(price, bedrooms, bathrooms, floors):
            continue

        items.append({
            "price":        price,
            "area":         str(area),
            "location":     _normalize_location(location),
            "district":     district,
            "bedrooms_n":   bedrooms,
            "bathrooms_n":  bathrooms,
            "floors_n":     floors,
            "house_type":   house_type,
            "year_built":   "",
            "facade_width": facade,
            "legal_status": legal,
            "_url":         f"https://nhatot.com/{ad.get('list_id', '')}",
            "category":     category,
        })

    return items


def crawl_batdongsan_playwright(
    categories: list[str],
    max_pages: int,
    existing_keys: set,
    seen_urls: set,
    writer,
    fout,
    delay: float = 2.0,
    start_page: int = 1,
) -> int:
    """Thu nhập từ batdongsan.com.vn sử dụng Playwright để bypass Cloudflare."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Chưa cài đặt Playwright! Vui lòng chạy: pip install playwright && python -m playwright install chromium")
        return 0

    total_new = 0
    
    with sync_playwright() as p:
        log.info("[BATDONGSAN] Khởi động trình duyệt Playwright Chromium...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
            locale="vi-VN",
            timezone_id="Asia/Ho_Chi_Minh"
        )
        
        # Tiêm mã giả lập (Stealth) tránh bị phát hiện webdriver
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['vi-VN', 'vi', 'en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)
        
        page = context.new_page()

        for idx, cat in enumerate(categories):
            # Thêm thời gian chờ khi đổi category để tránh kích hoạt Cloudflare
            if idx > 0:
                sleep_between = random.uniform(6.0, 10.0)
                log.info("[BATDONGSAN] Chờ %.2fs trước khi chuyển sang danh mục tiếp theo...", sleep_between)
                time.sleep(sleep_between)

            base_url = CATEGORIES_BDS.get(cat, CATEGORIES_BDS.get("nha-dat"))
            if not base_url:
                continue
            log.info("=== [BATDONGSAN] Category: %s ===", cat)

            for pg in range(start_page, start_page + max_pages):
                url = base_url if pg == 1 else f"{base_url}/p{pg}"
                log.info("[BATDONGSAN] Trang %d/%d  %s", pg, start_page + max_pages - 1, url)

                html = None
                success = False
                
                for attempt in range(1, 4):
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        
                        # Chờ selector tin đăng xuất hiện
                        selector = "div.re__pr-card, div.pr-card, div.js__card, div.product-item"
                        try:
                            page.wait_for_selector(selector, timeout=12000)
                            html = page.content()
                            # Kiểm tra xem có còn hiển thị trang challenge không
                            title = page.title()
                            if "Just a moment" in title or "Chờ một chút" in title or "challenge-running" in html:
                                raise Exception("Cloudflare challenge page detected in page title/content")
                            success = True
                            break
                        except Exception:
                            # Nếu gặp Cloudflare hoặc timeout
                            title = page.title()
                            html = page.content()
                            if "Just a moment" in title or "Chờ một chút" in title or "challenge-running" in html or "cf-challenge" in html:
                                log.warning("[BATDONGSAN] Phát hiện trang xác minh Cloudflare (Lần thử %d/3). Đang chờ 10s để tự động giải quyết...", attempt)
                                time.sleep(10)
                                # Thử chờ selector lại sau khi ngủ
                                try:
                                    page.wait_for_selector(selector, timeout=10000)
                                    html = page.content()
                                    # Thêm cuộn trang nhẹ để kích hoạt tải dữ liệu
                                    page.evaluate("window.scrollBy(0, 300)")
                                    success = True
                                    break
                                except Exception:
                                    pass
                            
                            # Nếu vẫn không được, sleep và reload ở vòng lặp sau
                            log.warning("[BATDONGSAN] Không tìm thấy tin đăng ở trang %d (Lần thử %d/3). Đang tải lại...", pg, attempt)
                            time.sleep(random.uniform(4.0, 6.0))
                            
                    except Exception as e:
                        if _is_network_error(e):
                            log.warning("🔌 [BATDONGSAN] Mất mạng: %s. Chờ internet phục hồi...", e)
                            _wait_for_internet()
                        else:
                            log.warning("[BATDONGSAN] Lỗi tải trang %s (Lần thử %d/3): %s", url, attempt, e)
                            time.sleep(3)

                if not success or not html:
                    log.error("[BATDONGSAN] Không thể vượt qua Cloudflare hoặc tải trang %d. Dừng category.", pg)
                    break

                listings = parse_batdongsan_page(html, cat)
                if not listings:
                    # Hãy check xem có thực sự là hết trang hay do selector thay đổi
                    title = page.title()
                    if "Just a moment" in title or "Chờ một chút" in title or "challenge-running" in html or "cf-challenge" in html:
                        log.error("[BATDONGSAN] Phát hiện bị chặn bởi Cloudflare challenge. Dừng category.")
                    else:
                        log.info("[BATDONGSAN] Không tìm thấy tin đăng nào – kết thúc category")
                    break

                new = [
                    l for l in listings
                    if l.get("_url") not in seen_urls
                    and (l.get("location", ""), str(l.get("price", "")), l.get("area", ""))
                    not in existing_keys
                ]
                
                for l in new:
                    writer.writerow(l)
                    seen_urls.add(l.get("_url", ""))
                    existing_keys.add((l.get("location", ""), str(l.get("price", "")), l.get("area", "")))

                total_new += len(new)
                log.info("  [BATDONGSAN] +%d mới | tổng: %d", len(new), len(existing_keys))
                fout.flush()

                _has_next = has_next_page_batdongsan(html, pg)
                if not _has_next:
                    log.info("[BATDONGSAN] Không có trang tiếp theo -> chuyển category")
                    break

                # Sleep ngẫu nhiên để tránh bot detection
                time.sleep(delay + random.uniform(0.5, 1.5))

        browser.close()
        
    return total_new


def crawl_nhatot_api(
    categories: list[str],
    max_pages: int,
    session: requests.Session,
    existing_keys: set,
    seen_urls: set,
    writer,
    fout,
    delay: float = 1.5,
) -> int:
    """Thu thập từ Chợ Tốt qua JSON API — không bị chặn, không cần JS."""
    LIMIT = 20
    total_new = 0
    api_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
        "Accept":     "application/json",
        "Referer":    "https://nhatot.com/",
    }

    for cat_key in categories:
        cg = NHATOT_CATEGORIES.get(cat_key, 1020)
        log.info("=== [NHATOT-API] Category: %s (cg=%d) ===", cat_key, cg)

        for region_code, region_name in NHATOT_REGIONS.items():
            if region_code == 0:
                continue

            for page in range(max_pages):
                params = {
                    "cg": cg, "region_v2": region_code,
                    "o": page * LIMIT, "st": "s,h",
                    "limit": LIMIT, "key_param_included": "true",
                }
                while True:
                    try:
                        r = session.get(NHATOT_API_URL, params=params,
                                        headers=api_headers, timeout=15)
                        break
                    except requests.exceptions.RequestException as exc:
                        if _is_network_error(exc):
                            log.warning("🔌 [NHATOT-API] Mất mạng, chờ phục hồi...")
                            _wait_for_internet()
                        else:
                            r = None
                            break

                if r is None or r.status_code != 200:
                    break
                try:
                    data = r.json()
                except Exception:
                    break

                listings = parse_nhatot_api_page(data, cat_key)
                if not listings:
                    break

                new = [
                    l for l in listings
                    if l.get("_url") not in seen_urls
                    and (l.get("location", ""), str(l.get("price", "")), l.get("area", ""))
                    not in existing_keys
                ]
                for l in new:
                    writer.writerow(l)
                    seen_urls.add(l.get("_url", ""))
                    existing_keys.add((l.get("location", ""), str(l.get("price", "")), l.get("area", "")))

                total_new += len(new)
                if new:
                    log.info("  [%s][%s] trang %d: +%d mới | tổng: %d",
                             region_name, cat_key, page + 1, len(new), len(existing_keys))
                fout.flush()

                if len(listings) < LIMIT:
                    break
                time.sleep(delay + random.uniform(0, 0.5))

    return total_new


# ─── Parser chính cho trang mogi.vn ──────────────────────────────────────────
def parse_mogi_page(
    html: str,
    category: str,
    session: "requests.Session | None" = None,
    fetch_detail: bool = False,
    delay: float = 0.8,
) -> list[dict]:
    """
    Phan tich HTML trang ket qua mogi.vn.
    Neu fetch_detail=True va session duoc truyen, se lay them trang chi tiet
    de trich xuat: phap ly, so tang, mat tien, nam xay dung.
    """
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict] = []

    for card in soup.find_all("div", class_="prop-info"):
        # Tieu de & URL
        a = card.find("a", class_="link-overlay")
        title_el = card.find("h2", class_="prop-title")
        title = _text(title_el)
        if not title:
            continue
        href = a.get("href", "") if a else ""
        url = href if href.startswith("http") else (BASE_URL + href)

        # Gia
        price_text = _text(card.find("div", class_="price"))
        price = _parse_price_to_vnd(price_text)

        # Dia chi
        location = _normalize_location(_text(card.find("div", class_="prop-addr")))

        # Thuoc tinh: dien tich / phong ngu / WC
        attr_ul  = card.find("ul", class_="prop-attr")
        attr_lis = attr_ul.find_all("li") if attr_ul else []
        area          = _normalize_area_to_m2(_text(attr_lis[0])) if len(attr_lis) > 0 else ""
        bedrooms_raw  = _text(attr_lis[1]) if len(attr_lis) > 1 else ""
        bathrooms_raw = _text(attr_lis[2]) if len(attr_lis) > 2 else ""
        bedrooms  = _first_number(bedrooms_raw)
        bathrooms = _first_number(bathrooms_raw)

        house_type = _infer_house_type(title, category)
        district   = _extract_district_from_location(location)

        # Trich xuat so bo tu tieu de
        extras = _extract_fields_from_text(title)

        # Validate
        if not _validate_record(price, bedrooms, bathrooms, extras.get("floors_n", "")):
            continue

        record = {
            "price":        price,
            "area":         area,
            "location":     location,
            "district":     district,
            "bedrooms_n":   bedrooms,
            "bathrooms_n":  bathrooms,
            "floors_n":     extras.get("floors_n", ""),
            "house_type":   house_type,
            "year_built":   extras.get("year_built", ""),
            "facade_width": extras.get("facade_width", ""),
            "legal_status": extras.get("legal_status", ""),
            "_url":         url,
            "category":     category,
        }

        # Lay trang chi tiet de bo sung truong con thieu
        if fetch_detail and session and url:
            missing = not all([
                record["floors_n"] or house_type in ("dat nen", "dat nền", "căn hộ"),
                record["legal_status"],
            ])
            if missing:
                try:
                    detail_html = _get(session, url, retries=2)
                    if detail_html and not _is_robot_page(detail_html):
                        detail_extras = _parse_mogi_detail(detail_html)
                        for k, v in detail_extras.items():
                            if not record.get(k):
                                record[k] = v
                    time.sleep(delay)
                except Exception:
                    pass

        items.append(record)

    return items


# ─── Hàm crawl chính ─────────────────────────────────────────────────────────
def crawl(
    categories: list[str],
    max_pages: int,
    output: Path,
    delay: float,
    start_page: int,
    sources: list[str] | None = None,
    fetch_detail: bool = False,
    fresh: bool = False,
) -> int:
    """
    Crawl nhieu category, nhieu trang, luu ket qua vao CSV.
    Ho tro resume: neu file da ton tai, chi ghi them ban ghi moi.
    sources: danh sach nguon can crawl, mac dinh ['mogi', 'homedy', 'alonhadat']
    fetch_detail: voi mogi, vao trang chi tiet de lay them truong
    fresh: xoa file cu truoc khi crawl
    """
    import pandas as pd  # lazy import

    # Reset counter cho phien crawl moi
    for key in _filtered_count:
        _filtered_count[key] = 0

    if sources is None:
        sources = ["mogi", "homedy", "alonhadat"]

    # Fresh mode: xoa file cu
    if fresh and output.exists():
        output.unlink()
        log.info("[fresh] Da xoa file cu: %s", output)

    existing_keys: set[tuple] = set()  # (location, price_str, area) cho resume
    seen_urls: set[str] = set()         # dedup trong phiên hiện tại
    write_header = not (output.exists() and output.stat().st_size > 0)

    if not write_header:
        try:
            df_ex = pd.read_csv(output, encoding="utf-8-sig")
            existing_keys = set(
                zip(
                    df_ex["location"].fillna("").tolist(),
                    df_ex["price"].fillna("").astype(str).tolist(),
                    df_ex["area"].fillna("").tolist(),
                )
            )
            log.info("Resume – đã có %d bản ghi trong %s", len(df_ex), output)
        except Exception as e:
            log.warning("Không đọc được file cũ: %s", e)
            write_header = True

    session = requests.Session()
    total_new = 0

    with open(output, "a", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for src in sources:
            log.info("\n======== Nguồn: %s ========", src.upper())

            # ── nhatot: dùng JSON API riêng, không phải HTML scraper ──────
            if src == "nhatot":
                total_new += crawl_nhatot_api(
                    categories=categories,
                    max_pages=max_pages,
                    session=session,
                    existing_keys=existing_keys,
                    seen_urls=seen_urls,
                    writer=writer,
                    fout=fout,
                    delay=delay,
                )
                continue

            # ── batdongsan: Dùng Playwright bypass Cloudflare ────────
            if src == "batdongsan":
                total_new += crawl_batdongsan_playwright(
                    categories=categories,
                    max_pages=max_pages,
                    existing_keys=existing_keys,
                    seen_urls=seen_urls,
                    writer=writer,
                    fout=fout,
                    delay=delay,
                    start_page=start_page,
                )
                continue

            for cat in categories:
                if src == "mogi":
                    cat_url_map = CATEGORIES_MOGI
                elif src == "homedy":
                    cat_url_map = CATEGORIES_HMDY
                else:
                    cat_url_map = CATEGORIES_ALN
                base_url = cat_url_map.get(cat, cat_url_map.get("nha-dat", ""))
                if not base_url:
                    continue
                log.info("=== [%s] Category: %s ===", src.upper(), cat)

                for page in range(start_page, start_page + max_pages):
                    if src == "mogi":
                        url = base_url if page == 1 else f"{base_url}?cp={page}"
                    elif src == "homedy":
                        url = base_url if page == 1 else f"{base_url}/p{page}"
                    else:  # alonhadat
                        url = base_url if page == 1 else f"{base_url}/trang-{page}"
                    log.info("Trang %d/%d  %s", page, start_page + max_pages - 1, url)

                    html = _get(session, url)
                    if html is None:
                        log.error("Trang %d: không lấy được, dừng category", page)
                        break

                    if src == "mogi":
                        listings = parse_mogi_page(
                            html, cat,
                            session=session,
                            fetch_detail=fetch_detail,
                            delay=delay * 0.4,
                        )
                        _has_next = has_next_page_mogi(html)
                    elif src == "homedy":
                        listings = parse_homedy_page(html, cat)
                        _has_next = has_next_page_homedy(html)
                    else:
                        listings = parse_alonhadat_page(html, cat)
                        _has_next = has_next_page_alonhadat(html)


                    if not listings:
                        log.info("Trang %d: không có tin nào – hết dữ liệu", page)
                        break

                    new = [l for l in listings
                           if l.get("_url") not in seen_urls
                           and (l.get("location", ""), str(l.get("price", "")), l.get("area", "")) not in existing_keys]
                    for l in new:
                        writer.writerow(l)
                        seen_urls.add(l.get("_url", ""))
                        existing_keys.add((l.get("location", ""), str(l.get("price", "")), l.get("area", "")))

                    total_new += len(new)
                    log.info("  +%d mới | tổng: %d", len(new), len(existing_keys))
                    fout.flush()

                    if not _has_next:
                        log.info("Không có trang tiếp theo -> chuyển category")
                        break

                    sleep_t = delay + random.uniform(0.5, 1.5)
                    time.sleep(sleep_t)

                # Bổ sung crawl theo tỉnh/thành để tăng độ phủ dữ liệu cho HOMEDY/ALONHADAT
                if src in ("homedy", "alonhadat"):
                    seed_html = _get(session, base_url)
                    if seed_html:
                        if src == "homedy":
                            region_links = _extract_region_links_homedy(seed_html, base_url)
                        else:
                            region_links = _extract_region_links_alonhadat(seed_html, base_url)

                        if region_links:
                            log.info("[%s][%s] Tìm thấy %d link tỉnh/thành", src.upper(), cat, len(region_links))

                        for ridx, region_url in enumerate(region_links[:300], start=1):
                            html_r = _get(session, region_url)
                            if html_r is None:
                                continue

                            if src == "homedy":
                                listings_r = parse_homedy_page(html_r, cat)
                            else:
                                listings_r = parse_alonhadat_page(html_r, cat)

                            if not listings_r:
                                continue

                            new_r = [
                                l for l in listings_r
                                if l.get("_url") not in seen_urls
                                and (l.get("location", ""), str(l.get("price", "")), l.get("area", "")) not in existing_keys
                            ]

                            for l in new_r:
                                writer.writerow(l)
                                seen_urls.add(l.get("_url", ""))
                                existing_keys.add((l.get("location", ""), str(l.get("price", "")), l.get("area", "")))

                            if new_r:
                                total_new += len(new_r)
                                log.info("  +%d mới từ tỉnh/thành (%d/%d)", len(new_r), ridx, len(region_links[:150]))
                                fout.flush()

                            sleep_t = min(2.5, delay + random.uniform(0.2, 1.0))
                            time.sleep(sleep_t)

    _log_filter_stats()
    log.info("Hoàn thành. Tổng bản ghi mới: %d | File: %s", total_new, output)
    return total_new


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thu thập dữ liệu bất động sản Việt Nam từ mogi.vn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pages",      type=int,   default=200,
                        help="Số trang crawl mỗi category")
    parser.add_argument("--output",                 default="vietnam_house_raw.csv",
                        help="File CSV đầu ra")
    parser.add_argument("--delay",      type=float, default=2.5,
                        help="Thời gian chờ giữa request (giây)")
    parser.add_argument("--start-page", type=int,   default=1,
                        help="Bắt đầu từ trang số")
    parser.add_argument(
        "--categories", nargs="+",
        default=["nha-dat", "can-ho", "biet-thu", "dat-nen", "nha-pho"],
        choices=list(CATEGORIES_MOGI.keys()),
        help="Danh mục cần crawl",
    )
    parser.add_argument(
        "--sources", nargs="+",
        default=["mogi", "homedy", "alonhadat"],
        choices=["mogi", "homedy", "alonhadat", "batdongsan", "nhatot"],
        help="Nguon du lieu: mogi, homedy, alonhadat (bat dau chay duoc); batdongsan/nhatot bi 403",
    )
    parser.add_argument(
        "--detail", action="store_true", default=False,
        help="[mogi] Vao trang chi tiet de lay them: phap ly, so tang, mat tien, nam xay",
    )
    parser.add_argument(
        "--fresh", action="store_true", default=False,
        help="Xoa file cu truoc khi crawl (khong resume)",
    )
    args = parser.parse_args()

    crawl(
        categories=args.categories,
        max_pages=args.pages,
        output=Path(args.output),
        delay=args.delay,
        start_page=args.start_page,
        sources=args.sources,
        fetch_detail=args.detail,
        fresh=args.fresh,
    )
