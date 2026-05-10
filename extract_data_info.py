"""
Nature Scientific Data論文からデータリポジトリ情報を抽出

論文PDF: doc/2301_Fatigue database of complex metallic alloys.pdf
"""

import sys
from pathlib import Path

# PDFからテキスト抽出を試みる
try:
    import PyPDF2
    pdf_available = True
except ImportError:
    print("PyPDF2 not installed. Trying pdfplumber...")
    try:
        import pdfplumber
        pdf_available = True
    except ImportError:
        print("Neither PyPDF2 nor pdfplumber available.")
        pdf_available = False

if not pdf_available:
    print("\nInstalling PyPDF2...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

# PDFファイルのパス
pdf_path = Path("doc") / "2301_Fatigue database of complex metallic alloys.pdf"

print("=" * 70)
print("Extracting Data Repository Information from PDF")
print("=" * 70)
print(f"\nPDF: {pdf_path}")

# PDFを開いてテキスト抽出
with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    
    print(f"Total pages: {num_pages}")
    print("\n" + "-" * 70)
    print("Searching for data repository information...")
    print("-" * 70)
    
    # 最初の数ページと最後のページ（通常Data Availabilityセクションがある）を重点的に検索
    pages_to_check = list(range(min(5, num_pages))) + list(range(max(0, num_pages-3), num_pages))
    
    keywords = [
        'data availability',
        'data repository',
        'figshare',
        'zenodo',
        'github',
        'doi',
        'supplementary',
        'dataset',
        'download',
        'http',
        'www',
        '.csv',
        '.xlsx',
        '.zip'
    ]
    
    relevant_sections = []
    
    for page_num in pages_to_check:
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        
        # キーワードが含まれる段落を抽出
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords):
                # 前後の行も含めてコンテキストを取得
                start = max(0, i-2)
                end = min(len(lines), i+3)
                context = '\n'.join(lines[start:end])
                
                relevant_sections.append({
                    'page': page_num + 1,
                    'text': context
                })
    
    # 重複を除去して表示
    print("\nFound relevant sections:\n")
    seen = set()
    for section in relevant_sections:
        text = section['text'].strip()
        if text and text not in seen:
            seen.add(text)
            print(f"[Page {section['page']}]")
            print(text)
            print("-" * 70)
    
    # DOIやURLを特定
    print("\n" + "=" * 70)
    print("Extracted URLs and DOIs:")
    print("=" * 70)
    
    import re
    all_text = ""
    for page_num in range(num_pages):
        all_text += pdf_reader.pages[page_num].extract_text() + "\n"
    
    # URLパターン
    url_pattern = r'https?://[^\s\)"\']+'
    urls = re.findall(url_pattern, all_text)
    
    # DOIパターン
    doi_pattern = r'10\.\d{4,}/[^\s\)"\']+'
    dois = re.findall(doi_pattern, all_text)
    
    print("\nURLs found:")
    data_related_urls = []
    for url in set(urls):
        if any(keyword in url.lower() for keyword in ['figshare', 'zenodo', 'github', 'data', 'supplementary']):
            print(f"  ★ {url}")
            data_related_urls.append(url)
        else:
            print(f"    {url}")
    
    print("\nDOIs found:")
    for doi in set(dois):
        print(f"  {doi}")
    
    # 推奨されるデータ取得方法
    print("\n" + "=" * 70)
    print("Recommended Data Access Steps:")
    print("=" * 70)
    
    if data_related_urls:
        print(f"\n1. Direct download from repository:")
        for url in data_related_urls[:3]:  # 上位3つのみ表示
            print(f"   {url}")
    
    print(f"\n2. Search on Nature Scientific Data:")
    print(f"   https://www.nature.com/articles/s41597-023-02347-z")
    
    print(f"\n3. Search on Figshare:")
    print(f"   https://figshare.com/search?q=fatigue+complex+metallic+alloys")
    
    print(f"\n4. Search on Zenodo:")
    print(f"   https://zenodo.org/search?q=fatigue+complex+metallic+alloys")

print("\n" + "=" * 70)
print("Extraction Complete")
print("=" * 70)
