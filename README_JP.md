# Material Usability Maximizer MVP

ベイズ最適化（BoTorch + Gaussian Process）を用いた材料寿命最大化の適応的実験計画システム

[![Version](https://img.shields.io/badge/version-v0.1.1-blue.svg)](https://github.com/tk-yasuno/botorch_ccfatigue/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![BoTorch](https://img.shields.io/badge/BoTorch-0.17.2-orange.svg)](https://botorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 概要

このプロジェクトは、材料の疲労寿命を最大化する製造条件を効率的に探索するためのMVP（Minimum Viable Product）です。CCFatigueデータセットを用いて、ベイズ最適化による適応的実験計画の実行可能性を検証します。

### 主な特徴

- **BoTorchベースのベイズ最適化**: SingleTaskGPモデルとExpected Improvementによる高度な探索
- **3つの計算モード**: 
  - `no_parallel`: 単一スレッド（再現性重視）
  - `cpu_16cores`: CPU 16コア並列処理
  - `gpu_16gb`: GPU加速（16GBメモリ制約対応）
- **包括的な可視化**: 収束曲線、探索空間、GP予測分布、計算パフォーマンス
- **柔軟な設定**: `config.yaml`とコマンドライン引数による制御
- **インタラクティブデモ**: Jupyter Notebookによるステップバイステップ実行

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

デフォルト設定（自動デバイス検出）で実行:

```bash
python main_demo.py
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
