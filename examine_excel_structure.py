"""
FatigueData-CMA2022 Excelファイルの構造調査

ダウンロードしたExcelファイルの構造を調査し、
MVP v0.1.2で使用できるデータを特定する
"""

import pandas as pd
from pathlib import Path
import openpyxl

print("=" * 70)
print("FatigueData-CMA2022 Excel Structure Examination")
print("=" * 70)

excel_path = Path("data") / "figshare_download" / "FatigueData-CMA2022.xlsx"

if not excel_path.exists():
    print(f"\n✗ File not found: {excel_path}")
    print("Please run: python download_figshare_data.py")
    exit(1)

print(f"\nExcel file: {excel_path}")
print(f"File size: {excel_path.stat().st_size / (1024*1024):.2f} MB")

# Excelファイルを開いてシート名を取得
print("\n" + "-" * 70)
print("Sheet Names:")
print("-" * 70)

wb = openpyxl.load_workbook(excel_path, read_only=True)
sheet_names = wb.sheetnames
print(f"\nTotal sheets: {len(sheet_names)}")
for i, name in enumerate(sheet_names, 1):
    print(f"  [{i}] {name}")

# 各シートの最初の数行を確認
print("\n" + "=" * 70)
print("Sheet Contents Preview:")
print("=" * 70)

for sheet_name in sheet_names[:5]:  # 最初の5シートのみ表示
    print(f"\n{'-' * 70}")
    print(f"Sheet: {sheet_name}")
    print(f"{'-' * 70}")
    
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=10)
        print(f"\nShape: {df.shape} (rows x columns)")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # データ型を表示
        print(f"\nData types:")
        print(df.dtypes)
        
    except Exception as e:
        print(f"\n✗ Error reading sheet: {e}")

# 特定のシートを探す（materials, composition, testing conditions など）
print("\n" + "=" * 70)
print("Searching for Relevant Sheets:")
print("=" * 70)

keywords = ['material', 'composition', 'element', 'test', 'fatigue', 
            'stress', 'cycle', 'mpea', 'alloy', 'dataset']

relevant_sheets = []
for sheet_name in sheet_names:
    for keyword in keywords:
        if keyword.lower() in sheet_name.lower():
            relevant_sheets.append(sheet_name)
            break

print(f"\nFound {len(relevant_sheets)} potentially relevant sheet(s):")
for sheet in relevant_sheets:
    print(f"  - {sheet}")

# より詳細な調査
if relevant_sheets:
    print("\n" + "=" * 70)
    print("Detailed Examination of Relevant Sheets:")
    print("=" * 70)
    
    for sheet_name in relevant_sheets[:3]:  # 最初の3つの関連シートを詳細に調査
        print(f"\n{'-' * 70}")
        print(f"Sheet: {sheet_name}")
        print(f"{'-' * 70}")
        
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"\nFull shape: {df.shape}")
            print(f"\nColumns ({len(df.columns)}):")
            for i, col in enumerate(df.columns):
                print(f"  [{i+1}] {col}")
            
            print(f"\nSample data (first 3 rows):")
            print(df.head(3))
            
            print(f"\nData statistics:")
            print(df.describe())
            
        except Exception as e:
            print(f"\n✗ Error: {e}")

print("\n" + "=" * 70)
print("Examination Complete")
print("=" * 70)

print("\nNext Steps:")
print("  1. Identify sheets containing MPEA composition data")
print("  2. Identify sheets containing S-N fatigue data")
print("  3. Identify sheets containing testing conditions")
print("  4. Create conversion script to merge data into MVP format")
