#!/usr/bin/env python3
"""
Thu thập dữ liệu bất động sản Việt Nam từ mogi.vn.

Sử dụng:
    python crawler.py --pages 100 --output vietnam_house_raw.csv
    python crawler.py --pages 50 --categories nha-dat can-ho biet-thu --delay 2.5
"""

import argparse
import csv
import logging
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

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

# Cho phép dùng key chung
CATEGORIES = CATEGORIES_MOGI  # dùng mogi làm default cho backward compat

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

CSV_FIELDS = [
    "price", "area", "location",
    "bedrooms", "bathrooms", "house_type", "floors",
    "category",
]


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
        if path.startswith(base_path + "-") and "/" not in path and "es" not in path:
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


def _get(session: requests.Session, url: str, retries: int = 3) -> str | None:
    """GET với retry + exponential back-off."""
    for attempt in range(retries):
        try:
            r = session.get(url, headers=_headers(url), timeout=20, allow_redirects=True)
            if r.status_code == 200:
                if _is_robot_page(r.text):
                    wait = 5 * (attempt + 1)
                    log.warning("Bị chặn robot-check: %s | chờ %ds rồi thử lại", url, wait)
                    time.sleep(wait)
                    continue
                return r.text
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("Rate-limited (429). Chờ %ds...", wait)
                time.sleep(wait)
                continue
            log.warning("HTTP %s – %s (lần %d)", r.status_code, url, attempt + 1)
        except requests.exceptions.RequestException as exc:
            log.warning("Lỗi request %s (lần %d): %s", url, attempt + 1, exc)
        time.sleep(2 ** attempt)
    return None


# ─── Parse helpers ─────────────────────────────────────────────────────────────
def _parse_price_to_vnd(text: str) -> float | None:
    """Chuyển chuỗi giá tiền Việt Nam → số đồng VND."""
    if not text:
        return None
    t = text.strip().lower().replace("\xa0", " ")
    if t in ("thoả thuận", "thỏa thuận", "liên hệ", ""):
        return None
    try:
        if "tỷ" in t:
            ty_m = re.search(r"([\d,.]+)\s*tỷ", t)
            tr_m = re.search(r"tỷ.*?([\d,.]+)\s*triệu", t)
            ty_val = float(ty_m.group(1).replace(",", "")) if ty_m else 0
            tr_val = float(tr_m.group(1).replace(",", "")) if tr_m else 0
            return round((ty_val * 1_000 + tr_val) * 1_000_000, 0)
        if any(x in t for x in ("triệu", "trieu")):
            m = re.search(r"([\d,.]+)", t)
            if m:
                return round(float(m.group(1).replace(",", "")) * 1_000_000, 0)
    except (ValueError, AttributeError):
        pass
    return None


def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def _first_number(text: str) -> str:
    m = re.search(r"\d+", text)
    return m.group() if m else ""


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
    """Kiểm tra còn trang tiếp theo không (alonhadat.com.vn)."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.select_one(".page a") is not None


# Alias cũ để không vỡ code cũ
has_next_page = has_next_page_mogi


# ─── Parser cho homedy.com ───────────────────────────────────────────
def parse_homedy_page(html: str, category: str) -> list[dict]:
    """
    Phân tích HTML trang kết quả homedy.com.

    Cấu trúc:
      Container : div.product-item  hoặc  div.items  > article
      Giá       : .product-price, .price, [class*='price']
      Địa chỉ   : .product-short-description, .address, [class*='location']
      Diện tích : [class*='area'], text chứa m²
      PN / WC   : [class*='bed'], [class*='bath']
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

        # ─ Tiêu đề ─
        title_el = card.select_one("h2, h3, .product-title, .realty-title")
        title = _text(title_el) or a_tag.get("title", "")
        if not title:
            continue

        # ─ Giá ─
        price_el = card.select_one(
            ".product-price, .price, .realty-price, "
            "[class*='price'], [data-price], strong.price, "
            ".price-value, .price-number, span.gia"
        )
        price = _parse_price_to_vnd(_text(price_el)) if price_el else None
        # fallback: quét toàn bộ text tìm pattern "X tỷ" / "X triệu"
        if price is None:
            card_text = card.get_text(" ")
            price = _parse_price_to_vnd(card_text)

        # ─ Địa chỉ ─
        loc_el = card.select_one(
            ".product-short-description, .product-location, "
            ".address, [class*='location'], [class*='address']"
        )
        location = _text(loc_el)

        # ─ Diện tích ─
        area_el = card.select_one("[class*='area'], [class*='dien-tich']")
        area_txt = _text(area_el)
        if not area_txt:
            m = re.search(r"[\d,.]+\s*m\s*[2²]", card.get_text(" "))
            area_txt = m.group(0) if m else ""

        # ─ Phòng ngủ / Phòng tắm ─
        bedrooms = bathrooms = ""
        bed_el  = card.select_one("[class*='bed'], [class*='pn'], [title*='ngủ']")
        bath_el = card.select_one("[class*='bath'], [class*='wc'], [title*='tắm']")
        if bed_el:  bedrooms  = _first_number(_text(bed_el))
        if bath_el: bathrooms = _first_number(_text(bath_el))

        house_type = _infer_house_type(title, category)

        items.append({
            "price":      price,
            "area":       area_txt,
            "location":   location,
            "bedrooms":   bedrooms,
            "bathrooms":  bathrooms,
            "house_type": house_type,
            "floors":     "",
            "_url":       url,
            "category":   category,
        })

    return items


def parse_alonhadat_page(html: str, category: str) -> list[dict]:
    """Phân tích HTML trang kết quả alonhadat.com.vn."""
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

        area = _text(card.select_one(".area"))
        location = _text(card.select_one(".new-address")) or _text(card.select_one(".property-address"))
        bedrooms = _first_number(_text(card.select_one(".bedroom")))
        floors = _first_number(_text(card.select_one(".floors")))
        house_type = _infer_house_type(title, category)

        items.append({
            "price": price,
            "area": area,
            "location": location,
            "bedrooms": bedrooms,
            "bathrooms": "",
            "house_type": house_type,
            "floors": floors,
            "_url": url,
            "category": category,
        })

    return items


# ─── Parser chính cho trang mogi.vn ──────────────────────────────────────────
def parse_mogi_page(html: str, category: str) -> list[dict]:
    """
    Phân tích HTML trang kết quả mogi.vn.

    Cấu trúc đã xác minh qua kiểm tra HTML thực tế:
      Container : div.prop-info         (15 mục / trang)
      Tiêu đề   : h2.prop-title         bên trong a.link-overlay
      URL       : a.link-overlay[href]
      Địa chỉ   : div.prop-addr
      Thuộc tính: ul.prop-attr > li     [0]=diện tích  [1]=PN  [2]=WC
      Giá       : div.price
    """
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict] = []

    for card in soup.find_all("div", class_="prop-info"):
        # ── tiêu đề & URL ──
        a = card.find("a", class_="link-overlay")
        title_el = card.find("h2", class_="prop-title")
        title = _text(title_el)
        if not title:
            continue
        href = a.get("href", "") if a else ""
        url = href if href.startswith("http") else (BASE_URL + href)

        # ── giá ──
        price_text = _text(card.find("div", class_="price"))
        price = _parse_price_to_vnd(price_text)

        # ── địa chỉ ──
        location = _text(card.find("div", class_="prop-addr"))

        # ── thuộc tính: diện tích / phòng ngủ / WC ──
        attr_ul = card.find("ul", class_="prop-attr")
        attr_lis = attr_ul.find_all("li") if attr_ul else []
        area          = _text(attr_lis[0]) if len(attr_lis) > 0 else ""
        bedrooms_raw  = _text(attr_lis[1]) if len(attr_lis) > 1 else ""
        bathrooms_raw = _text(attr_lis[2]) if len(attr_lis) > 2 else ""
        bedrooms  = _first_number(bedrooms_raw)
        bathrooms = _first_number(bathrooms_raw)

        house_type = _infer_house_type(title, category)

        items.append({
            "price":      price,
            "area":       area,
            "location":   location,
            "bedrooms":   bedrooms,
            "bathrooms":  bathrooms,
            "house_type": house_type,
            "floors":     "",
            "_url":       url,
            "category":   category,
        })

    return items


# ─── Hàm crawl chính ─────────────────────────────────────────────────────────
def crawl(
    categories: list[str],
    max_pages: int,
    output: Path,
    delay: float,
    start_page: int,
    sources: list[str] | None = None,
) -> int:
    """
    Crawl nhiều category, nhiều trang, lưu kết quả vào CSV.
    Hỗ trợ resume: nếu file đã tồn tại, chỉ ghi thêm bản ghi mới.
    sources: danh sách nguồn cần crawl, mặc định ['mogi', 'homedy', 'alonhadat']
    """
    import pandas as pd  # lazy import

    if sources is None:
        sources = ["mogi", "homedy", "alonhadat"]

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
                    elif src == "homedy":  # homedy.com dùng dạng /p2, /p3...
                        url = base_url if page == 1 else f"{base_url}/p{page}"
                    else:  # alonhadat.com.vn dùng /trang-{page}
                        url = base_url if page == 1 else f"{base_url}/trang-{page}"
                    log.info("Trang %d/%d  %s", page, start_page + max_pages - 1, url)

                    html = _get(session, url)
                    if html is None:
                        log.error("Trang %d: không lấy được, dừng category", page)
                        break

                    if src == "mogi":
                        listings = parse_mogi_page(html, cat)
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

                        for ridx, region_url in enumerate(region_links[:150], start=1):
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

    log.info("Hoàn thành. Tổng bản ghi mới: %d | File: %s", total_new, output)
    return total_new


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thu thập dữ liệu bất động sản Việt Nam từ mogi.vn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pages",      type=int,   default=100,
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
        choices=["mogi", "homedy", "alonhadat"],
        help="Nguồn dữ liệu: mogi = mogi.vn, homedy = homedy.com, alonhadat = alonhadat.com.vn",
    )
    args = parser.parse_args()

    crawl(
        categories=args.categories,
        max_pages=args.pages,
        output=Path(args.output),
        delay=args.delay,
        start_page=args.start_page,
        sources=args.sources,
    )
