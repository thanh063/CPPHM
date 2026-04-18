import requests
from bs4 import BeautifulSoup
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Phân tích chi tiết 2 trang chính
sites_detail = {
    'batdongsan24h.com': 'https://batdongsan24h.com',
    'dautu.vn': 'https://dautu.vn/bat-dong-san',
}

print("📍 PHÂN TÍCH CHI TIẾT\n")

for site_name, base_url in sites_detail.items():
    print("="*70)
    print(f"🔍 {site_name.upper()}")
    print("="*70)
    
    try:
        resp = requests.get(base_url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # Get title
            title = soup.select_one('title')
            print(f"\n✅ Status: 200 OK")
            print(f"   Title: {title.get_text() if title else 'N/A'}")
            
            # Try pagination patterns
            print(f"\n📄 Pagination:")
            paginations = [
                soup.select('a[href*="page"]')[:2],
                soup.select('a[href*="p="]')[:2],
                soup.select('a[rel="next"]'),
            ]
            for i, pg in enumerate(paginations):
                if pg:
                    print(f"   Pattern {i+1}: {len(pg)} links found")
                    for link in pg:
                        href = link.get('href', '')
                        if href:
                            print(f"     - {href[:60]}")
            
            # Try to find property containers
            print(f"\n🏠 Property containers:")
            containers = [
                ('div[class*="item"]', soup.select('div[class*="item"]')),
                ('div[class*="product"]', soup.select('div[class*="product"]')),
                ('article', soup.select('article')),
                ('div[class*="listing"]', soup.select('div[class*="listing"]')),
                ('li[class*="item"]', soup.select('li[class*="item"]')),
            ]
            
            found = False
            for selector, items in containers:
                if items:
                    print(f"   ✅ {selector}: {len(items)} items")
                    
                    # Sample first item
                    first = items[0]
                    print(f"      Sample: {first.name}.{'.'.join(first.get('class', [])[:2])}")
                    
                    # Try to extract price/area
                    text = first.get_text(separator=' | ', strip=True)[:100]
                    print(f"      Text: {text}...")
                    
                    # Look for links
                    links = first.select('a[href]')
                    if links:
                        print(f"      Link pattern: {links[0].get('href', '')[:50]}...")
                    
                    found = True
                    break
            
            if not found:
                print(f"   ⚠️  No standard containers found")
                print(f"   Available divs: {len(soup.select('div'))}")
                print(f"   Available articles: {len(soup.select('article'))}")
                
        else:
            print(f"❌ Status: {resp.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")

print("\n" + "="*70)
print("💡 RECOMMENDATION:")
print("="*70)
print("""
✅ BEST OPTIONS:
  1. batdongsan24h.com - Dedicated real estate site, 8 items/page
  2. dautu.vn - Has 224 items, lists news/properties
  3. alonhadat.com.vn - Already working (expand to 100 pages)

❌ NOT SUITABLE:
  - sendo, shopee, lazada = E-commerce (not real estate focused)
  - Many news sites = Complex to parse, mixed content

🚀 NEXT STEP:
  If batdongsan24h & dautu work: Add parsers to crawler.py
  Else: Expand alonhadat to 100 pages instead
""")
