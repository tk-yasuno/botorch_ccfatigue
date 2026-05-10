# Material Usability Maximizer MVP — Version History

## v0.1.2 (2026-05-10)

### 新機能

1. **複雑金属合金データセットのサポート**
   - Nature Scientific Data (2023) の複雑金属合金疲労データベースに対応
   - データローダー拡張: `dataset_name="complex_alloy"` オプション追加
   - データフィルタリング機能: 室温・空気中・一定振幅疲労に絞り込み
   - 材料組成（Fe, Ni, Cr, Co, Al など）と試験条件を統合的に扱う

2. **PCA（主成分分析）による次元削減**
   - 新モジュール `src/feature_engineering.py` 追加
   - `CompositionPCA` クラス: 組成ベクトルの次元を自動削減
   - 累積寄与率95%を基準にコンポーネント数を決定
   - PCA適用後も試験条件（応力、周波数など）は保持

3. **探索的データ分析（EDA）機能**
   - `src/visualization.py` に7つの新しいプロット関数を追加:
     - `plot_pca_variance()`: PCA累積寄与率プロット
     - `plot_pca_space()`: PCA空間での分布可視化（PC1 vs PC2）
     - `plot_element_loadings()`: 各主成分における元素の寄与度
     - `plot_correlation_matrix()`: 特徴量相関行列ヒートマップ
     - `plot_feature_distributions()`: 特徴量分布ヒストグラム
     - `plot_composition_lifetime_scatter()`: 組成-寿命散布図

4. **設定ファイルの拡張**
   - `config.yaml` に `complex_alloy` セクション追加
   - PCA設定: `use_pca`, `pca_variance_threshold`, `pca_n_components`
   - データフィルタ条件の設定
   - カテゴリ変数（荷重モードなど）の指定

5. **ドキュメントの充実**
   - `doc/ComplexAlloy_DataSource.md`: データ取得ガイド
   - `data/generate_mock_complex_alloy.py`: モックデータ生成スクリプト
   - `test_complex_alloy.py`: 統合テストスクリプト

### 改善

- `src/data_loader.py` の `load_ccfatigue_data()` 関数に `filter_conditions` 引数追加
- データセット別のデフォルト設計変数を自動設定
- エラーメッセージとログ出力の改善

### テスト

- 統合テスト成功: 174サンプル（フィルタ後）で動作確認
- PCA次元削減: 8次元 → 7次元（4主成分 + 3試験条件）
- すべてのEDA可視化が正常に動作

### 既知の制限

- 実際のNature Scientific Data (2023) からのデータ取得は手動
- MVP v0.1.2では室温・空気中・引張圧縮のデータのみ使用
- カテゴリ変数のOne-hotエンコーディングは部分実装
- main_demo.py の完全な統合は次のステップ

### 次のステップ

- main_demo.py を拡張してcomplex_alloyデータセットに対応
- Jupyter Notebook にPCAとEDAのセクション追加
- 実データを使った検証
- 熱処理条件の統合（将来のバージョン）

---

## v0.1.1 (2026-04-XX)

### 初回リリース

- BoTorchベースのベイズ最適化エンジン
- CCFatigue S-N曲線データのサポート
- 3つの計算モード（no_parallel, cpu_16cores, gpu_16gb）
- 基本的な可視化機能
- Jupyter Notebookデモ
- 包括的なドキュメント
