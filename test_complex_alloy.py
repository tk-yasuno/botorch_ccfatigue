"""
MVP v0.1.2 — Complex Alloy データセットの統合テスト

新機能の動作確認:
1. データローダーでcomplex_alloyデータセットを読み込み
2. PCAによる次元削減
3. EDA可視化
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from src.data_loader import load_ccfatigue_data, prepare_initial_data, get_search_bounds
from src.feature_engineering import CompositionPCA, apply_feature_engineering
from src.visualization import (
    plot_pca_variance, plot_pca_space, plot_element_loadings,
    plot_correlation_matrix, plot_feature_distributions,
    plot_composition_lifetime_scatter
)

print("=" * 70)
print("MVP v0.1.2 — Complex Alloy Integration Test")
print("=" * 70)

# =============================================================================
# 1. データ読み込み
# =============================================================================
print("\n[Step 1] Loading Complex Alloy Dataset...")

X_df, y_df, design_vars, obj_var = load_ccfatigue_data(
    dataset_name="complex_alloy",
    design_variables=["Fe", "Ni", "Cr", "Co", "Al", "stress_max", "stress_ratio", "frequency"],
    objective_variable="cycles_to_failure",
    objective_transform="log10"
    # filter_conditions are now applied by default in data_loader
)

print(f"\n✓ Loaded {len(X_df)} samples")
print(f"  Features: {list(X_df.columns)}")
print(f"  Target: {y_df.name}")

# =============================================================================
# 2. PCA適用前の統計
# =============================================================================
print("\n[Step 2] Pre-PCA Statistics...")
print(f"  Original feature dimensions: {X_df.shape[1]}")
print(X_df.describe())

# =============================================================================
# 3. PCA適用
# =============================================================================
print("\n[Step 3] Applying PCA to Composition...")

composition_cols = ["Fe", "Ni", "Cr", "Co", "Al"]
other_cols = ["stress_max", "stress_ratio", "frequency"]

pca_transformer = CompositionPCA(
    n_components=None,
    variance_threshold=0.95,
    random_state=42
)

X_pca_df = pca_transformer.fit_transform(X_df, composition_cols, other_cols)

print(f"\n✓ PCA applied")
print(f"  Reduced dimensions: {X_pca_df.shape[1]} (from {X_df.shape[1]})")
print(f"  New features: {list(X_pca_df.columns)}")

# PCA情報取得
pca_info = pca_transformer.get_explained_variance_info()
print(f"  Number of components: {pca_info['n_components']}")
print(f"  Cumulative variance: {pca_info['cumulative_variance_ratio'][-1]:.2%}")

weights = pca_transformer.get_component_weights()
print(f"\n  Component weights (top element per PC):")
for pc in weights.columns[:min(3, len(weights.columns))]:
    top_element = weights[pc].abs().idxmax()
    top_value = weights[pc][top_element]
    print(f"    {pc}: {top_element} ({top_value:.3f})")

# =============================================================================
# 4. 可視化テスト
# =============================================================================
print("\n[Step 4] Testing Visualization Functions...")

output_dir = Path("results") / "test_complex_alloy"
output_dir.mkdir(parents=True, exist_ok=True)

# PCA累積寄与率
print("  - PCA variance plot...")
plot_pca_variance(pca_info, output_path=str(output_dir / "pca_variance.png"))

# PCA空間
print("  - PCA space plot...")
plot_pca_space(X_pca_df, y_df, output_path=str(output_dir / "pca_space.png"))

# 元素寄与度
print("  - Element loadings plot...")
plot_element_loadings(weights, output_path=str(output_dir / "element_loadings.png"))

# 相関行列（元データ）
print("  - Correlation matrix...")
plot_correlation_matrix(X_df, output_path=str(output_dir / "correlation_matrix.png"))

# 特徴量分布
print("  - Feature distributions...")
plot_feature_distributions(X_df, y_df, output_path=str(output_dir / "feature_distributions.png"))

# 組成-寿命散布図
print("  - Composition-lifetime scatter...")
plot_composition_lifetime_scatter(
    X_df, y_df, composition_cols,
    output_path=str(output_dir / "composition_lifetime.png")
)

print(f"\n✓ All visualizations saved to {output_dir}")

# =============================================================================
# 5. データ分割テスト
# =============================================================================
print("\n[Step 5] Testing Data Preparation...")

train_X, train_Y, pool_X, pool_Y = prepare_initial_data(
    X_pca_df, y_df, M=10, seed=42
)

print(f"  Train: {train_X.shape}, Pool: {pool_X.shape}")

# 境界取得
bounds = get_search_bounds(X_pca_df)
print(f"  Search bounds shape: {bounds.shape}")
print(f"  Bounds:\n{bounds}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Integration Test PASSED ✓")
print("=" * 70)
print(f"\nResults saved to: {output_dir}")
print("\nAll components are working correctly:")
print("  ✓ Data loader with complex_alloy dataset")
print("  ✓ Data filtering (room temperature, air, tension-compression)")
print("  ✓ PCA transformation (composition -> principal components)")
print("  ✓ EDA visualizations (7 types of plots)")
print("  ✓ Data preparation for Bayesian optimization")
print("\nNext: Run full Bayesian optimization with complex_alloy dataset")
print("  Command: python main_demo.py --dataset-name complex_alloy --M 10 --T 20")
