# Material Usability Maximizer MVP

ベイズ最適化（BoTorch + Gaussian Process）を用いた材料寿命最大化の適応的実験計画システム

[![Version](https://img.shields.io/badge/version-v0.1.2-blue.svg)](https://github.com/tk-yasuno/botorch_ccfatigue/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![BoTorch](https://img.shields.io/badge/BoTorch-0.17.2-orange.svg)](https://botorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 概要

このプロジェクトは、材料の疲労寿命を最大化する製造条件を効率的に探索するためのMVP（Minimum Viable Product）です。v0.1.2では、複雑金属合金（多主元素合金、メタリックガラス）の疲労データベースに対応し、材料組成と試験条件を統合的に最適化できるようになりました。

### 主な特徴

- **BoTorchベースのベイズ最適化**: SingleTaskGPモデルとExpected Improvementによる高度な探索
- **複数のデータセット対応** (v0.1.2 NEW):
  - CCFatigue S-N曲線データ（従来）
  - 複雑金属合金疲労データベース（Nature Scientific Data 2023）
- **PCA次元削減** (v0.1.2 NEW): 材料組成の高次元データを効率的に扱う
- **探索的データ分析（EDA）** (v0.1.2 NEW): 相関行列、PCA空間、特徴量分布の可視化
- **3つの計算モード**: 
  - `no_parallel`: 単一スレッド（再現性重視）
  - `cpu_16cores`: CPU 16コア並列処理
  - `gpu_16gb`: GPU加速（16GBメモリ制約対応）
- **包括的な可視化**: 収束曲線、探索空間、GP予測分布、計算パフォーマンス
- **柔軟な設定**: `config.yaml`とコマンドライン引数による制御
- **インタラクティブデモ**: Jupyter Notebookによるステップバイステップ実行

### v0.1.2の新機能

1. **複雑金属合金データセットのサポート**
   - 材料組成（Fe, Ni, Cr, Co, Al など）を特徴量として使用
   - 試験条件（応力、応力比、周波数）との統合最適化
   - データフィルタリング（室温・空気中・一定振幅疲労）

2. **PCA（主成分分析）による次元削減**
   - 組成ベクトルの次元を自動削減（累積寄与率95%基準）
   - 計算効率の向上と解釈性の向上
   - PCA空間の可視化と元素寄与度の分析

3. **探索的データ分析（EDA）機能**
   - 特徴量相関行列ヒートマップ
   - 特徴量分布ヒストグラム
   - 組成-寿命散布図
   - PCA累積寄与率プロット
   - 主成分空間の可視化

## インストール

### 前提条件

- Python 3.8以上
- (オプション) CUDA対応GPU（gpu_16gbモード使用時）

### セットアップ

```bash
# リポジトリのクローン（将来的にGitHub登録予定）
cd I:\ACT2025.5.26-2030\MVP\SURVIV\BayesOpt\botorch_ccfatigue

# 仮想環境の作成（推奨）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 基本実行

デフォルト設定（CCFatigue S-N曲線データ、自動デバイス検出）で実行:

```bash
python main_demo.py
```

### 複雑金属合金データセットの使用 (v0.1.2 NEW)

#### 実データのダウンロード

Nature Scientific Data (2023) から公開されている実データを使用できます：

```bash
# 1. FatigueData-CMA2022 データベースをダウンロード
python download_figshare_data.py

# 2. MVP形式のCSVに変換
python convert_figshare_to_csv.py

# データソース: https://doi.org/10.6084/m9.figshare.23007362
# 論文: Zhang et al., Scientific Data 10:447 (2023)
```

変換後、`data/complex_alloy_fatigue_real.csv`に以下の形式で保存されます：
- **材料組成**: Fe, Ni, Cr, Co, Al, Mn, Ti, Cu など（at%）
- **試験条件**: stress_max, stress_ratio, frequency, temperature, environment, loading_mode
- **疲労寿命**: cycles_to_failure, log10_cycles

#### 統合テストの実行

```bash
# PCA、EDA、BoTorch準備の動作確認
python test_complex_alloy.py

# 結果は results/test_complex_alloy/ に保存されます：
#   - pca_variance.png: PCA累積寄与率
#   - pca_space.png: 主成分空間（PC1 vs PC2）
#   - element_loadings.png: 元素寄与度
#   - correlation_matrix.png: 特徴量相関行列
#   - feature_distributions.png: 特徴量分布
#   - composition_lifetime.png: 組成-寿命関係
```

#### 最適化の実行

```bash
# 実データで最適化を実行（初期点10、反復20回）
python main_demo.py --dataset-name complex_alloy --M 10 --T 20

# データフィルタリング条件（デフォルト）:
#   - 温度: 室温（20-30°C）
#   - 環境: 空気中
#   - 荷重: 単軸引張圧縮（uniaxial）
```

#### データセットの統計

- **総データ**: 503サンプル（MPEAs with S-N data）
- **フィルタ後**: 230サンプル（室温・空気中・単軸）
- **主要元素**: Fe (20-47%), Ni (0-25%), Cr (10-25%), Co (10-26%)
- **疲労寿命範囲**: 10^2.8 ~ 10^8.7 サイクル

#### モックデータでの試行

実データダウンロード前に、合成データで動作確認：

```bash
# モックデータを生成（300サンプル）
cd data
python generate_mock_complex_alloy.py

# 統合テストを実行
cd ..
python test_complex_alloy.py
```

### 計算モードの指定

```bash
# CPU単一スレッド（再現性重視）
python main_demo.py --compute-mode no_parallel

# CPU 16コア並列処理
python main_demo.py --compute-mode cpu_16cores

# GPU加速
python main_demo.py --compute-mode gpu_16gb
```

### パラメータのカスタマイズ

```bash
# 初期点数と反復回数の変更
python main_demo.py --M 15 --T 30

# 乱数シードの指定（再現性）
python main_demo.py --seed 12345

# 出力ディレクトリの指定
python main_demo.py --output results_custom
```

### 設定ファイルの使用

`config.yaml`を編集してデフォルト設定を変更:

```yaml
compute_mode: cpu_16cores
optimization:
  M: 20
  T: 50
  acq_function: LogEI
```

### Jupyter Notebookデモ

インタラクティブなデモを実行:

```bash
jupyter notebook demo_notebook.ipynb
```

## プロジェクト構成

```
botorch_ccfatigue/
├── src/                          # ソースコード
│   ├── __init__.py
│   ├── compute_config.py         # 計算環境管理
│   ├── data_loader.py            # データ読み込みと前処理
│   ├── bayesian_optimizer.py     # ベイズ最適化エンジン
│   └── visualization.py          # 可視化機能
├── doc/                          # ドキュメント
│   ├── Concept_20260510.pdf      # コンセプト図
│   └── Implementation_Notes.md   # 実装詳細
├── results/                      # 結果保存（.gitignore対象）
├── main_demo.py                  # スタンドアロン実行スクリプト
├── demo_notebook.ipynb           # Jupyterデモ
├── config.yaml                   # 設定ファイル
├── requirements.txt              # 依存パッケージ
└── README.md                     # このファイル
```

## コンセプト

このシステムは以下の適応的実験計画サイクルを実装しています:

1. **t: iteration** - 反復的に実験条件を探索（t=1から開始）
2. **ym: deterioration index** - 劣化指標（疲労寿命）の測定
3. **Bayes Optimize fm(x,t)** - ベイズ最適化による次候補点の提案
4. **Bayes update → t+1** - 観測データによるGPモデルの更新と次反復へ

詳細は`doc/Concept_20260510.pdf`を参照してください。

## 技術スタック

- **最適化**: BoTorch 0.9+, GPyTorch 1.11+
- **機械学習**: PyTorch 2.0+
- **データ処理**: pandas, NumPy, scikit-learn
- **可視化**: matplotlib, seaborn
- **材料データ**: CCFatigue

## 出力

実行後、`results/`ディレクトリに以下が保存されます:

- `convergence_curve.png`: 最適化の収束曲線
- `explored_space.png`: 探索された条件空間の可視化
- `gp_predictions.png`: GP予測分布（1Dスライス）
- `acquisition_function.png`: 獲得関数の可視化
- `compute_performance.png`: 計算パフォーマンス（実行時間、メモリ使用量）
- `optimization_summary.csv`: 最適化履歴の詳細
- `best_conditions.json`: 発見された最良条件

## トラブルシューティング

### GPU非搭載環境で`gpu_16gb`モードを指定した場合

自動的にCPUモードにフォールバックします。警告メッセージが表示されます。

### CCFatigueのインストールエラー

CCFatigueパッケージが見つからない場合は、公式リポジトリを確認してください:
```bash
pip install ccfatigue
```

### メモリ不足エラー

GPUメモリが不足する場合は、`config.yaml`で以下を調整:
```yaml
gpu_16gb:
  num_restarts: 5   # デフォルト10から削減
  raw_samples: 64   # デフォルト128から削減
```

## 将来の拡張

- 実際の実験装置との統合
- 多目的最適化（寿命 + コスト + 製造容易性）
- 制約条件の明示的な扱い
- バッチ実験計画（qEI）
- 時系列劣化の動的モデリング
- 分散計算対応

## ライセンス

Apache License 2.0 - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## お問い合わせ

質問、バグレポート、機能リクエストは、GitHubのIssueをご利用ください：

**GitHub Issues**: https://github.com/tk-yasuno/botorch_ccfatigue/issues

プロジェクト管理者: [@tk-yasuno](https://github.com/tk-yasuno)

GitHubリポジトリ: https://github.com/tk-yasuno/botorch_ccfatigue

## 謝辞

- BoTorchチーム: 高品質なベイズ最適化フレームワーク
- CCFatigueプロジェクト: 疲労データセット提供

---

**Version**: v0.1.1

**Built with**: BoTorch 0.17.2 | PyTorch 2.11.0 | Python 3.12.10

**最終更新**: 2026年5月10日
