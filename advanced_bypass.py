import requests
import time
import random
from bs4 import BeautifulSoup

# Headers mô phỏng real browser tốt hơn
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

headers_template = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Cache-Control': 'max-age=0',
}

# Test URLs
urls = [
    ('batdongsan', 'https://batdongsan.com.vn/nha-dat-ban'),
    ('batdongsan', 'https://batdongsan.com.vn/nha-dat-ban-tphcm'),
    ('nhatot', 'https://nhatot.com/ban-nha-dat'),
]

session = requests.Session()

print("🔓 Thử bypass anti-scraping với kỹ thuật nâng cao:\n")

for site, url in urls:
    print(f"{'='*70}")
    print(f"🔗 {site.upper()}: {url[-40:]}")
    print(f"{'='*70}")
    
    for attempt in range(3):
        try:
            headers = headers_template.copy()
            headers['User-Agent'] = random.choice(user_agents)
            
            print(f"  Attempt {attempt + 1}/3...")
            
            # Delay ngẫu nhiên trước request
            time.sleep(random.uniform(0.5, 2))
            
            resp = session.get(
                url,
                headers=headers,
                timeout=15,
                allow_redirects=True,
                verify=True
            )
            
            print(f"    Status: {resp.status_code}")
            print(f"    Content-Type: {resp.headers.get('Content-Type', 'N/A')[:50]}")
            print(f"    Content-Length: {len(resp.content)} bytes")
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Get page title
                title = soup.select_one('title')
                if title:
                    print(f"    Title: {title.get_text()[:50]}")
                
                # Try to find listing items
                items = soup.select('[class*="item"]')[:1]
                if items:
                    print(f"    ✅ Found {len(soup.select('[class*=\"item\"]'))} items")
                    break
                else:
                    print(f"    ⚠️  No items detected")
                    
            elif resp.status_code == 403:
                print(f"    ❌ 403 Forbidden (still blocked)")
                
        except Exception as e:
            print(f"    Error: {str(e)[:60]}")
        
        if attempt < 2:
            # Exponential backoff
            wait = 2 ** (attempt + 1) + random.uniform(0, 1)
            print(f"    Chờ {wait:.1f}s trước attempt tiếp...")
            time.sleep(wait)

print("\n" + "="*70)
print("📊 KẾT LUẬN:")
print("="*70)
print("""
✅ Nếu bypass thành công: Tôi sẽ thêm parser cho 2 trang này vào crawler
❌ Nếu vẫn 403: Giải pháp thay thế:
   1. Cài Selenium + Chrome/Firefox headless
   2. Tăng crawl alonhadat.com.vn thêm pages
   3. Tìm các trang khác (free listing)
""")
