"""
FatigueData-CMA2022データベースのダウンロードと変換スクリプト

データソース: https://doi.org/10.6084/m9.figshare.23007362
Nature Scientific Data (2023): Fatigue database of complex metallic alloys
"""

import requests
import json
import pandas as pd
from pathlib import Path
import zipfile
import io

print("=" * 70)
print("FatigueData-CMA2022 Database Download")
print("=" * 70)

# Figshare APIを使用してデータセット情報を取得
figshare_article_id = "23007362"
api_url = f"https://api.figshare.com/v2/articles/{figshare_article_id}"

print(f"\nFetching metadata from Figshare...")
print(f"API URL: {api_url}")

response = requests.get(api_url)
if response.status_code == 200:
    metadata = response.json()
    
    print(f"\n✓ Successfully retrieved metadata")
    print(f"\nTitle: {metadata.get('title', 'N/A')}")
    print(f"DOI: {metadata.get('doi', 'N/A')}")
    print(f"Published: {metadata.get('published_date', 'N/A')}")
    print(f"Authors: {', '.join([author['full_name'] for author in metadata.get('authors', [])])}")
    
    # ファイルリスト
    files = metadata.get('files', [])
    print(f"\nAvailable files ({len(files)}):")
    
    data_dir = Path("data") / "figshare_download"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    excel_files = []
    json_files = []
    mat_files = []
    
    for i, file_info in enumerate(files, 1):
        name = file_info['name']
        size_mb = file_info['size'] / (1024 * 1024)
        download_url = file_info['download_url']
        
        print(f"\n  [{i}] {name}")
        print(f"      Size: {size_mb:.2f} MB")
        print(f"      URL: {download_url}")
        
        # ファイルタイプを分類
        if name.endswith('.xlsx') or name.endswith('.xls'):
            excel_files.append(file_info)
        elif name.endswith('.json'):
            json_files.append(file_info)
        elif name.endswith('.mat'):
            mat_files.append(file_info)
    
    # ダウンロード推奨
    print("\n" + "-" * 70)
    print("Download Recommendations:")
    print("-" * 70)
    
    if excel_files:
        print("\n[Recommended] Excel files for easy access:")
        for f in excel_files:
            print(f"  - {f['name']} ({f['size']/(1024*1024):.2f} MB)")
    
    if json_files:
        print("\n[Alternative] JSON files for programmatic access:")
        for f in json_files:
            print(f"  - {f['name']} ({f['size']/(1024*1024):.2f} MB)")
    
    # ユーザーに選択を求める
    print("\n" + "=" * 70)
    print("Download Options:")
    print("=" * 70)
    print("\nWhich file format would you like to download?")
    print("  1. Excel (.xlsx) - Recommended for this MVP")
    print("  2. JSON (.json) - For structured data processing")
    print("  3. All files")
    print("  4. Skip download (manual download required)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    files_to_download = []
    if choice == "1":
        files_to_download = excel_files
    elif choice == "2":
        files_to_download = json_files
    elif choice == "3":
        files_to_download = files
    else:
        print("\n⚠ Manual download required.")
        print(f"\nPlease visit: https://doi.org/10.6084/m9.figshare.23007362")
        print(f"And download files to: {data_dir}")
        exit(0)
    
    # ダウンロード実行
    if files_to_download:
        print(f"\n✓ Downloading {len(files_to_download)} file(s)...")
        
        for file_info in files_to_download:
            name = file_info['name']
            download_url = file_info['download_url']
            file_path = data_dir / name
            
            print(f"\n  Downloading: {name}...")
            
            response = requests.get(download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r    Progress: {percent:.1f}%", end='', flush=True)
                
                print(f"\n    ✓ Saved to: {file_path}")
            else:
                print(f"\n    ✗ Failed to download: {response.status_code}")
        
        print("\n" + "=" * 70)
        print("Download Complete!")
        print("=" * 70)
        print(f"\nFiles saved to: {data_dir}")
        
        # 次のステップ
        print("\nNext Steps:")
        print("  1. Examine the downloaded files")
        print("  2. Convert data to MVP-compatible format (CSV)")
        print("  3. Run: python convert_figshare_to_csv.py")

else:
    print(f"\n✗ Failed to retrieve metadata: {response.status_code}")
    print(f"\nPlease manually download from:")
    print(f"https://doi.org/10.6084/m9.figshare.23007362")
