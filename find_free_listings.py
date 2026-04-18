import requests
from bs4 import BeautifulSoup
import time

# Các trang free listing bất động sản VN
free_listing_sites = {
    '1. dienmayxanh': 'https://realestate.dienmayxanh.com/nha-dat',
    '2. sendo.vn': 'https://sendo.vn/bat-dong-san-c-3102',
    '3. lazada': 'https://lazada.vn/properties',
    '4. tiki.vn': 'https://tiki.vn/bat-dong-san',
    '5. shopee': 'https://shopee.vn/search?keyword=nha-dat',
    '6. voz.vn': 'https://voz.vn/f/trao-doi-bat-dong-san.47/',
    '7. 24h.com.vn': 'https://24h.com.vn/bat-dong-san',
    '8. vietnamnet.vn': 'https://vietnamnet.vn/trang-chu-bat-dong-san',
    '9. cafeF.vn': 'https://cafef.vn/bat-dong-san',
    '10. fpt.vn': 'https://fpt.vn/bat-dong-san',
    '11. vccinews.com': 'https://vccinews.com/bat-dong-san',
    '12. kinhtevadubao.vn': 'https://kinhtevadubao.vn/bat-dong-san',
    '13. tukudo.vn': 'https://tukudo.vn',
    '14. batdongsan24h.com': 'https://batdongsan24h.com',
    '15. dautu.vn': 'https://dautu.vn/bat-dong-san',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

print("🔍 Tìm kiếm các trang FREE LISTING:\n")
print("="*70)

results = []

for site_name, url in free_listing_sites.items():
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        status = resp.status_code
        
        if status == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            title = soup.select_one('title')
            title_text = title.get_text()[:50] if title else "N/A"
            
            # Try to find item containers
            items = len(soup.select('[class*="item"], [class*="listing"], [class*="product"], article')) 
            
            print(f"✅ {site_name}")
            print(f"   Status: {status} | Items: {items}")
            print(f"   Title: {title_text}")
            results.append((site_name, url, 'OK'))
            
        elif status == 404:
            print(f"❌ {site_name:25} : 404 Not Found")
            
        else:
            print(f"⚠️  {site_name:25} : {status}")
            
    except requests.exceptions.Timeout:
        print(f"⏱️  {site_name:25} : Timeout")
    except requests.exceptions.ConnectionError:
        print(f"❌ {site_name:25} : Connection Error")
    except Exception as e:
        print(f"❌ {site_name:25} : {str(e)[:40]}")
    
    time.sleep(1)

print("\n" + "="*70)
print("📊 TÓM LẠI - POSSIBLE SOURCES:\n")

ok_sites = [r for r in results if r[2] == 'OK']

if ok_sites:
    print(f"✅ Tìm được {len(ok_sites)} trang OK:\n")
    for name, url, _ in ok_sites:
        print(f"  • {name:30} : {url}")
else:
    print("❌ Không tìm được trang free listing nào hoạt động")

print("\n" + "="*70)
print("💡 KHUYẾN NGHỊ:\n")
print("""
Nếu không tìm được free listing khác:
  1. 🔄 Tăng crawl alonhadat.com.vn: --pages 100
  2. 📰 Crawl news/portals kết hợp:
     - vnexpress.net (tin bất động sản)
     - vietnamplus.vn
     - tuoitre.vn/bat-dong-san
  3. ⚙️ Setup Selenium để bypass Batdongsan/Nhatot
  4. 🔗 Tìm RSS feeds hoặc API công khai
""")
