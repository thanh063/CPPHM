import requests
from bs4 import BeautifulSoup
import time

# More realistic headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
    'Referer': 'https://www.google.com/',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Try different URL patterns
test_configs = [
    {
        'name': 'batdongsan.com.vn',
        'urls': [
            'https://batdongsan.com.vn/nha-dat-ban-tphcm',
            'https://batdongsan.com.vn/nha-dat-ban',
            'https://batdongsan.com.vn/ban-nha-dat',
        ]
    },
    {
        'name': 'nhatot.com',
        'urls': [
            'https://nhatot.com/ban-nha-dat-tphcm',
            'https://nhatot.com/ban-nha-dat',
            'https://nhatot.com/nha-dat',
        ]
    }
]

session = requests.Session()

for config in test_configs:
    print(f"\n{'='*60}")
    print(f"🔍 Kiểm tra: {config['name']}")
    print(f"{'='*60}")
    
    for url in config['urls']:
        try:
            print(f"\n  Testing: {url}")
            resp = session.get(url, headers=headers, timeout=15, allow_redirects=True)
            print(f"  Status: {resp.status_code}")
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Print page title to confirm we got content
                title = soup.select_one('title')
                if title:
                    print(f"  Title: {title.get_text()[:60]}")
                
                # Find all divs and links
                items = soup.select('[class*="item"]')[:3]
                if items:
                    print(f"  Found {len(soup.select('[class*=\"item\"]'))} potential item elements")
                    for item in items:
                        classes = '.'.join(item.get('class', []))
                        print(f"    - {item.name}.{classes[:50]}")
                
                links = soup.select('a[href*="nha"], a[href*="dat"], a[href*="ban"]')[:3]
                if links:
                    print(f"  Found {len(links)} property links")
                    for link in links:
                        print(f"    - {link.get('href', '')[:70]}")
                
                break  # Success, break from URL loop
                
        except Exception as e:
            print(f"  Error: {str(e)[:80]}")
        
        time.sleep(2)

