"""
FatigueData-CMA2022 Parameterシートの詳細調査

重要なメタデータ（組成、試験条件など）を含むparameterシートを詳細に調査
"""

import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 70)
print("Parameter Sheet Detailed Examination")
print("=" * 70)

excel_path = Path("data") / "figshare_download" / "FatigueData-CMA2022.xlsx"

# 最初に生データを読んで構造を理解する
print("\n[1] Reading raw parameter sheet structure...")
df_raw = pd.read_excel(excel_path, sheet_name='parameter', header=None, nrows=5)
print(f"\nFirst 5 rows (no header):")
print(df_raw)

# 実際のヘッダー行を探す
print("\n[2] Identifying header rows...")
print("\nRow 0 (potential header 1):")
print(df_raw.iloc[0].values)
print("\nRow 1 (potential header 2):")
print(df_raw.iloc[1].values)

# ヘッダーを正しく読み込む（header=0が正しい行）
print("\n[3] Reading with proper header...")
df = pd.read_excel(excel_path, sheet_name='parameter', header=0)
print(f"\nShape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns):
    print(f"  [{i+1}] {col}")

# 最初の数行を表示
print("\n[4] First 3 data rows:")
print(df.head(3))

# カラムを調査して意味を推測
print("\n" + "=" * 70)
print("Column Content Analysis:")
print("=" * 70)

# 各カラムのユニークな値の数と例を表示
for col in df.columns:
    n_unique = df[col].nunique()
    sample_values = df[col].dropna().unique()[:5]
    
    print(f"\n[{col}]")
    print(f"  Unique values: {n_unique}")
    print(f"  Sample: {sample_values}")
    if n_unique <= 10 and n_unique > 0:
        print(f"  All values: {df[col].dropna().unique()}")

# データセットIDが含まれる列を探す
print("\n" + "=" * 70)
print("Looking for Key Columns:")
print("=" * 70)

# dataset id 列を探す
dataset_id_cols = [col for col in df.columns if 'dataset' in col.lower() or 'id' in col.lower()]
print(f"\nDataset ID columns: {dataset_id_cols}")
if dataset_id_cols:
    print(f"Sample values: {df[dataset_id_cols[0]].head(10).tolist()}")

# composition / element / alloy 関連の列を探す
composition_keywords = ['element', 'composition', 'alloy', 'Fe', 'Ni', 'Cr', 'Co', 'Al', 
                        'Cu', 'Ti', 'Mn', 'Zr', 'material']
comp_cols = [col for col in df.columns 
             if any(keyword.lower() in str(col).lower() for keyword in composition_keywords)]
print(f"\nComposition-related columns: {comp_cols}")

# testing condition 関連の列を探す
test_keywords = ['stress', 'temperature', 'frequency', 'ratio', 'environment', 
                 'load', 'test', 'R', 'temp', 'freq']
test_cols = [col for col in df.columns 
             if any(keyword.lower() in str(col).lower() for keyword in test_keywords)]
print(f"\nTesting condition columns: {test_cols}")

# fatigue type 関連の列を探す
fatigue_keywords = ['fatigue', 'S-N', 'type', 'curve']
fatigue_cols = [col for col in df.columns 
                if any(keyword.lower() in str(col).lower() for keyword in fatigue_keywords)]
print(f"\nFatigue type columns: {fatigue_cols}")

# S-Nシートとparameterシートをdataset idでマージしてみる
print("\n" + "=" * 70)
print("Merging S-N Data with Parameters:")
print("=" * 70)

df_sn = pd.read_excel(excel_path, sheet_name='S-N')
print(f"\nS-N sheet shape: {df_sn.shape}")
print(f"S-N columns: {list(df_sn.columns)}")
print(f"Unique dataset IDs in S-N: {df_sn['dataset id'].nunique()}")
print(f"Sample dataset IDs: {sorted(df_sn['dataset id'].unique())[:10]}")

# マージを試みる
if 'dataset id' in df.columns:
    print(f"\nParameter sheet unique dataset IDs: {df['dataset id'].nunique()}")
    print(f"Sample: {sorted(df['dataset id'].unique())[:10]}")
    
    # 簡単なマージを試みる
    merged = pd.merge(df_sn.head(20), df, on='dataset id', how='left')
    print(f"\nMerged shape: {merged.shape}")
    print(f"\nMerged columns: {list(merged.columns)}")
    print(f"\nSample merged data:")
    print(merged.head(3))

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
