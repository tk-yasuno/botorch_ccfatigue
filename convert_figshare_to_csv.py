"""
FatigueData-CMA2022をMVP v0.1.2形式に変換

1. MPEAsのS-Nデータのみを抽出
2. 組成データをパース（Fe, Ni, Cr, Co, Alなど）
3. 試験条件を抽出
4. S-Nシートとマージしてサイクル数を取得
5. MVP形式のCSVに保存
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

print("=" * 70)
print("Converting FatigueData-CMA2022 to MVP v0.1.2 Format")
print("=" * 70)

excel_path = Path("data") / "figshare_download" / "FatigueData-CMA2022.xlsx"
output_path = Path("data") / "complex_alloy_fatigue_real.csv"

# ===================================================================
# Step 1: Load parameter sheet with correct header
# ===================================================================
print("\n[1] Loading parameter sheet...")
df_param = pd.read_excel(excel_path, sheet_name='parameter', header=1)
print(f"  Shape: {df_param.shape}")
print(f"  Columns: {list(df_param.columns)}")

# 重要な列の名前を確認して保存
key_columns = {
    'dataset_id': 'dataset id',
    'material_type': 'type of the material',
    'fatigue_type': 'types of fatigue data',
    'composition': 'composition\n(at%)',
    'temperature': 'fatigue temperature \n(°C)',
    'environment': 'fatigue environment',
    'load_ratio': 'load ratio',
    'frequency': 'frequency \n(Hz)',
    'test_type': 'types of fatigue tests'
}

print(f"\n  Key columns identified:")
for key, col in key_columns.items():
    if col in df_param.columns:
        print(f"    ✓ {key}: '{col}'")
    else:
        print(f"    ✗ {key}: '{col}' NOT FOUND")

# ===================================================================
# Step 2: Filter for MPEAs with S-N data
# ===================================================================
print("\n[2] Filtering for MPEAs with S-N data...")

# Material type distribution
print(f"\n  Material type distribution:")
print(df_param[key_columns['material_type']].value_counts())

# Fatigue data type distribution
print(f"\n  Fatigue data type distribution:")
print(df_param[key_columns['fatigue_type']].value_counts())

# Filter for MPEAs
df_mpea = df_param[df_param[key_columns['material_type']] == 'multi-principal elements alloy'].copy()
print(f"\n  ✓ MPEAs: {len(df_mpea)} datasets")

# Filter for S-N data
df_mpea_sn = df_mpea[df_mpea[key_columns['fatigue_type']] == 'sn'].copy()
print(f"  ✓ MPEAs with S-N data: {len(df_mpea_sn)} datasets")

# ===================================================================
# Step 3: Parse composition data
# ===================================================================
print("\n[3] Parsing composition data...")

def parse_composition(comp_str):
    """
    Parse composition string like "Fe50.0-Ni25.0-Cr15.0-Co5.0-Al5.0"
    Returns dict like {'Fe': 50.0, 'Ni': 25.0, 'Cr': 15.0, 'Co': 5.0, 'Al': 5.0}
    """
    if pd.isna(comp_str):
        return {}
    
    comp_dict = {}
    # Split by hyphen
    parts = str(comp_str).split('-')
    
    for part in parts:
        # Match pattern: Element + number (e.g., "Fe50.0")
        match = re.match(r'([A-Z][a-z]?)(\d+\.?\d*)', part)
        if match:
            element = match.group(1)
            value = float(match.group(2))
            comp_dict[element] = value
    
    return comp_dict

# Test parsing on first few rows
print(f"\n  Testing composition parser:")
for i, comp in enumerate(df_mpea_sn[key_columns['composition']].head(3)):
    parsed = parse_composition(comp)
    print(f"    [{i+1}] '{comp}' → {parsed}")

# Parse all compositions
df_mpea_sn['composition_dict'] = df_mpea_sn[key_columns['composition']].apply(parse_composition)

# Extract individual elements
target_elements = ['Fe', 'Ni', 'Cr', 'Co', 'Al', 'Ti', 'Cu', 'Mn', 'V', 'Nb', 'Mo', 'W']
for element in target_elements:
    df_mpea_sn[element] = df_mpea_sn['composition_dict'].apply(
        lambda x: x.get(element, 0.0)
    )

print(f"\n  Element distribution (non-zero counts):")
for element in target_elements:
    count = (df_mpea_sn[element] > 0).sum()
    if count > 0:
        print(f"    {element}: {count} datasets")

# ===================================================================
# Step 4: Load and merge S-N data
# ===================================================================
print("\n[4] Loading and merging S-N fatigue data...")

df_sn = pd.read_excel(excel_path, sheet_name='S-N')
print(f"  S-N sheet shape: {df_sn.shape}")
print(f"  S-N columns: {list(df_sn.columns)}")

# Merge parameter data with S-N data
df_merged = pd.merge(
    df_sn,
    df_mpea_sn,
    left_on='dataset id',
    right_on=key_columns['dataset_id'],
    how='inner'
)
print(f"\n  ✓ Merged shape: {df_merged.shape}")
print(f"  ✓ Total data points: {len(df_merged)}")

# ===================================================================
# Step 5: Prepare MVP format
# ===================================================================
print("\n[5] Preparing MVP v0.1.2 format...")

# Helper function to clean numeric fields
def clean_numeric_field(value):
    """
    Clean numeric fields that may contain multiple values or non-numeric strings
    Examples: '0.25, 0.75' -> 0.25, '10' -> 10.0, 'NaN' -> np.nan
    """
    if pd.isna(value):
        return np.nan
    
    # Convert to string and handle comma-separated values
    value_str = str(value).strip()
    
    # If comma-separated, take the first value
    if ',' in value_str:
        value_str = value_str.split(',')[0].strip()
    
    # Try to convert to float
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return np.nan

# Rename columns to MVP format
df_mvp = pd.DataFrame()

# Composition columns (use target_elements)
for element in target_elements:
    if element in df_merged.columns:
        df_mvp[element] = df_merged[element]

# Stress amplitude (convert to stress_max assuming stress_ratio)
# stress_max = stress_amplitude / (1 - load_ratio) for tension-tension
# For tension-compression (R=-1): stress_max = stress_amplitude
df_mvp['stress_amplitude'] = df_merged['stress amplitude\nσa (MPa)']

# Clean and convert stress_ratio
df_mvp['stress_ratio'] = df_merged[key_columns['load_ratio']].apply(clean_numeric_field)

# Calculate stress_max based on load ratio
# σa = (σmax - σmin) / 2
# R = σmin / σmax
# → σa = σmax * (1 - R) / 2
# → σmax = 2 * σa / (1 - R)
def calculate_stress_max(sigma_a, R):
    if pd.isna(R):
        return sigma_a  # Assume tension-compression if R is missing
    
    try:
        R = float(R)
        if R == -1:
            return sigma_a  # For R=-1, σmax = σa
        else:
            return 2 * sigma_a / (1 - R)
    except:
        return sigma_a

df_mvp['stress_max'] = df_merged.apply(
    lambda row: calculate_stress_max(
        row['stress amplitude\nσa (MPa)'],
        row[key_columns['load_ratio']]
    ),
    axis=1
)

# Other test conditions
df_mvp['frequency'] = df_merged[key_columns['frequency']].apply(clean_numeric_field)
df_mvp['temperature'] = df_merged[key_columns['temperature']].apply(clean_numeric_field)
df_mvp['environment'] = df_merged[key_columns['environment']]
df_mvp['loading_mode'] = df_merged[key_columns['test_type']]

# Fatigue life
df_mvp['cycles_to_failure'] = df_merged['life\nN (cycle)']
df_mvp['log10_cycles'] = np.log10(df_merged['life\nN (cycle)'])

# Dataset ID for reference
df_mvp['dataset_id'] = df_merged['dataset id']

print(f"\n  MVP columns: {list(df_mvp.columns)}")
print(f"  MVP shape: {df_mvp.shape}")

# ===================================================================
# Step 6: Data quality checks
# ===================================================================
print("\n[6] Data quality checks...")

# Check for missing values
print(f"\n  Missing values:")
missing = df_mvp.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print(missing)
else:
    print("    None")

# Check data ranges
print(f"\n  Data ranges:")
print(df_mvp[['stress_max', 'stress_ratio', 'frequency', 'temperature', 'cycles_to_failure']].describe())

# Environment distribution
print(f"\n  Environment distribution:")
print(df_mvp['environment'].value_counts())

# Loading mode distribution
print(f"\n  Loading mode distribution:")
print(df_mvp['loading_mode'].value_counts())

# Temperature distribution
print(f"\n  Temperature distribution:")
print(df_mvp['temperature'].value_counts())

# ===================================================================
# Step 7: Save to CSV
# ===================================================================
print("\n[7] Saving to CSV...")

output_path.parent.mkdir(parents=True, exist_ok=True)
df_mvp.to_csv(output_path, index=False)
print(f"\n  ✓ Saved to: {output_path}")
print(f"  ✓ Total samples: {len(df_mvp)}")

print("\n" + "=" * 70)
print("Conversion Complete!")
print("=" * 70)

print(f"\nSummary:")
print(f"  - Source: FatigueData-CMA2022 (Nature Scientific Data 2023)")
print(f"  - Material: Multi-Principal Element Alloys (MPEAs)")
print(f"  - Fatigue type: S-N (stress-life)")
print(f"  - Total samples: {len(df_mvp)}")
print(f"  - Output: {output_path}")

print(f"\nNext steps:")
print(f"  1. Filter data (room temperature, air, tension-compression)")
print(f"  2. Test with: python test_complex_alloy.py")
print(f"  3. Run optimization: python main_demo.py --dataset-name complex_alloy")
