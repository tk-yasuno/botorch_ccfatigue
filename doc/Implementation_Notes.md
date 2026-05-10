# Material Usability Maximizer MVP - 実装詳細

**作成日**: 2026年5月10日  
**バージョン**: 0.1.0

## 目次

1. [コンセプト図との対応](#コンセプト図との対応)
2. [アーキテクチャ概要](#アーキテクチャ概要)
3. [主要モジュールの詳細](#主要モジュールの詳細)
4. [計算環境管理](#計算環境管理)
5. [ベイズ最適化のフロー](#ベイズ最適化のフロー)
6. [技術的詳細](#技術的詳細)
7. [仮想実験の実装](#仮想実験の実装)
8. [可視化戦略](#可視化戦略)
9. [パフォーマンス最適化](#パフォーマンス最適化)
10. [制限事項と将来の拡張](#制限事項と将来の拡張)

---

## コンセプト図との対応

### コンセプト図の要素

[doc/Concept_20260510.pdf](Concept_20260510.pdf) に示されている主要な要素:

1. **t: iteration, start t=1**
   - 反復的に実験条件を探索
   - 実装: `BayesianOptimizer.optimize_loop()` のメインループ

2. **ym: deterioration index**
   - 劣化指標 = 疲労寿命 (`cycles_to_failure`)
   - 実装: `y_df` として `log10(cycles_to_failure)` で管理

3. **Bayes Optimize fm(x,t)**
   - ベイズ最適化による次候補点の提案
   - 実装: 
     - GPモデル: `SingleTaskGP`（BoTorch）
     - 獲得関数: `ExpectedImprovement` または `LogExpectedImprovement`
     - 最適化: `optimize_acqf`

4. **Predict M+1 ... M+H**
   - GPによる予測分布
   - 実装: `model.posterior(X_test)` による平均と分散の予測

5. **Bayes update → t+1**
   - 観測データによるモデル更新
   - 実装: `torch.cat()` でデータを追加し、次反復でGPを再学習

### 実装マッピング

| コンセプト | 実装クラス/関数 | ファイル |
|-----------|----------------|---------|
| データ読み込み | `load_ccfatigue_data()` | `data_loader.py` |
| 初期データ分割 | `prepare_initial_data()` | `data_loader.py` |
| 仮想実験 | `DatabaseEvaluator.evaluate()` | `data_loader.py` |
| GPモデル学習 | `BayesianOptimizer.fit_model()` | `bayesian_optimizer.py` |
| 候補点提案 | `BayesianOptimizer.propose_next_point()` | `bayesian_optimizer.py` |
| 最適化ループ | `BayesianOptimizer.optimize_loop()` | `bayesian_optimizer.py` |
| 収束曲線可視化 | `plot_convergence()` | `visualization.py` |
| GP予測可視化 | `plot_gp_predictions()` | `visualization.py` |

---

## アーキテクチャ概要

### システム構成

```
┌─────────────────────────────────────────────────────────────┐
│                     main_demo.py                            │
│                 (統合実行スクリプト)                          │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ComputeConfig │  │ DataLoader   │  │Visualization │
│              │  │              │  │              │
│・デバイス管理 │  │・CCFatigue   │  │・収束曲線     │
│・スレッド制御 │  │・前処理       │  │・探索空間     │
│・メモリ監視   │  │・仮想実験     │  │・GP予測      │
└──────────────┘  └──────────────┘  └──────────────┘
                          │
                          ↓
                  ┌──────────────┐
                  │BayesianOpt   │
                  │              │
                  │・GPモデル     │
                  │・獲得関数     │
                  │・最適化ループ │
                  └──────────────┘
```

### データフロー

1. **初期化フェーズ**
   ```
   config.yaml → ComputeConfig → デバイス設定
   CCFatigue → load_data → X_df, y_df
   X_df, y_df → prepare_initial_data → train_X, train_Y, pool_X, pool_Y
   ```

2. **最適化フェーズ**（t = 1, 2, ..., T）
   ```
   train_X, train_Y → fit_model → SingleTaskGP
   SingleTaskGP → propose_next_point → x_candidate
   x_candidate → DatabaseEvaluator → x_real, y_real
   x_real, y_real → torch.cat → train_X, train_Y (更新)
   ```

3. **可視化フェーズ**
   ```
   results → plot_convergence → 収束曲線
   results → plot_explored_space → 散布図
   results → plot_gp_predictions → GP予測
   results → save_results_summary → CSV/JSON
   ```

---

## 主要モジュールの詳細

### 1. compute_config.py

**役割**: 計算環境の統一的管理

**主要クラス**: `ComputeConfig`

**機能**:
- `config.yaml` の読み込み
- 計算モードの決定（auto/no_parallel/cpu_16cores/gpu_16gb）
- `torch.device` の設定
- `torch.set_num_threads()` によるCPU並列制御
- GPU自動検出とフォールバック
- GPUメモリ使用量監視

**設計決定**:
- config.yaml + コマンドライン引数（引数優先）
- autoモード: `torch.cuda.is_available()` で自動選択
- GPU非搭載環境での安全なフォールバック

### 2. data_loader.py

**役割**: データ管理と仮想実験

**主要関数**:
- `load_ccfatigue_data()`: CCFatigueデータの読み込みと前処理
- `prepare_initial_data()`: ランダム分割によるM点の初期データ作成
- `get_search_bounds()`: 探索空間の境界抽出

**主要クラス**: `DatabaseEvaluator`

**仮想実験の実装**:
- BoTorchが提案した連続値 `x_candidate` に対して
- `NearestNeighbors` で最も近い実験データ点を探索
- その点の `cycles_to_failure` を「実験結果」として返す
- MVP版の簡易実装（将来は実験装置APIに置き換え可能）

**設計決定**:
- log10変換: 寿命の桁が広範囲なため対数スケールで扱う
- 最近傍探索: scikit-learnの `NearestNeighbors` を使用（高速）
- デバイス管理: CPU上で探索、結果をdeviceに転送

### 3. bayesian_optimizer.py

**役割**: ベイズ最適化のコアロジック

**主要クラス**: `BayesianOptimizer`

**主要メソッド**:

#### `fit_model(train_X, train_Y)`
- `SingleTaskGP` の構築
- 入力正規化: `Normalize(d)`
- 出力標準化: `Standardize(m=1)`
- 学習: `ExactMarginalLogLikelihood` + `fit_gpytorch_mll`

#### `propose_next_point(model, train_Y)`
- 現在の最良値 `best_f = train_Y.max()`
- 獲得関数: `ExpectedImprovement(model, best_f)`
- 最適化: `optimize_acqf(acq_func, bounds, q=1, ...)`
- GPUメモリに応じて `num_restarts`, `raw_samples` を調整

#### `optimize_loop(train_X, train_Y, T)`
- t = 1 から T までの反復
- 各反復:
  1. GPモデル学習
  2. 候補点提案
  3. 仮想実験評価
  4. データ追加（ベイズ更新）
  5. 履歴記録（best_y, 実行時間, メモリ）

**設計決定**:
- SingleTaskGP: 単一タスク回帰に最適
- EI vs LogEI: LogEIは数値的に安定（小さいEI値でも扱える）
- q=1: 順次最適化（バッチ実験は将来の拡張）

### 4. visualization.py

**役割**: 包括的な可視化機能

**主要関数**:

#### `plot_convergence(history)`
- 最良値の累積推移（青線）
- 各反復での観測値（赤点）
- トレンドの確認

#### `plot_explored_space(train_X, train_Y, feature_names)`
- 2D散布図の組み合わせ
- 色: log10(cycles_to_failure)
- すべての特徴量ペアを可視化

#### `plot_gp_predictions(model, bounds, feature_names, ...)`
- 1Dスライスによる予測分布
- 各設計変数について、他を中央値に固定
- 平均 ± 2σ の信頼区間
- 訓練データ点の重ね合わせ

#### `plot_compute_performance(history, compute_mode)`
- 各反復の実行時間（棒グラフ）
- GPUメモリ使用量推移（折れ線グラフ）
- モード間のパフォーマンス比較に有用

#### `save_results_summary(results, feature_names, output_dir)`
- CSVで最適化履歴
- JSONで最良条件
- テキストでサマリー

---

## 計算環境管理

### 3つの計算モード

#### 1. no_parallel
- **用途**: デバッグ、再現性確保
- **設定**: device='cpu', num_threads=1
- **特徴**: 最も遅いが、結果が完全に再現可能

#### 2. cpu_16cores
- **用途**: CPU並列処理の活用
- **設定**: device='cpu', num_threads=16
- **特徴**: `torch.set_num_threads(16)` で線形代数演算を並列化
- **効果**: 行列演算（GP学習、獲得関数最適化）が高速化

#### 3. gpu_16gb
- **用途**: GPU加速、16GBメモリ制約対応
- **設定**: device='cuda', batch_size=auto, num_restarts=10, raw_samples=128
- **特徴**: 
  - テンソルをGPUに配置
  - メモリに応じて `optimize_acqf` のパラメータ調整
  - メモリ監視機能
- **フォールバック**: GPU非搭載時は自動的にCPUへ

### デバイス管理の実装

```python
# ComputeConfig で統一的に管理
device = torch.device("cuda" if torch.cuda.is_available() and mode == "gpu_16gb" else "cpu")

# 全テンソルをdeviceに配置
train_X = train_X.to(device)
train_Y = train_Y.to(device)
model = model.to(device)

# 最適化パラメータの動的調整
if mode == "gpu_16gb":
    num_restarts = 10
    raw_samples = 128
else:
    num_restarts = 5
    raw_samples = 64
```

---

## ベイズ最適化のフロー

### 詳細ステップ（t = 1 の例）

```
[初期状態] t=0
  train_X: (10, 3)  # M=10の初期データ
  train_Y: (10, 1)
  best_y: 6.2341

[Iteration t=1]
  1. fit_model(train_X, train_Y)
     → SingleTaskGP学習
     → ハイパーパラメータ最適化（lengthscale, outputscale, noise）
  
  2. propose_next_point(model, train_Y)
     → best_f = 6.2341
     → EI(x) = E[max(0, f(x) - 6.2341)]
     → optimize_acqf: x_candidate = [320.5, -0.3, 25.1]
  
  3. evaluator.evaluate(x_candidate)
     → 最近傍探索: x_real = [321.0, -0.29, 25.0]
     → y_real = 6.4521
  
  4. Bayes update
     → train_X = torch.cat([train_X, x_real])  # (11, 3)
     → train_Y = torch.cat([train_Y, y_real])  # (11, 1)
     → best_y = 6.4521 (更新!)
  
  5. 履歴記録
     → history["iteration"].append(1)
     → history["best_y"].append(6.4521)
     → history["execution_time"].append(2.34)

[Iteration t=2]
  （同様の処理を繰り返し...）
```

### 獲得関数の役割

**Expected Improvement (EI)**:
- 定義: `EI(x) = E[max(0, f(x) - f_best)]`
- 意味: 「現在の最良値を上回る期待改善量」
- 探索と活用のバランス:
  - 高い平均予測 → 活用（exploitation）
  - 高い不確実性 → 探索（exploration）

**LogEI**:
- 数値的に安定したEIの変種
- 小さいEI値でも扱いやすい
- 推奨設定（特にGPU環境）

---

## 技術的詳細

### Gaussian Process (GP)

**モデル**: `SingleTaskGP`（BoTorch）

**カーネル**: RBF (Radial Basis Function) カーネル（デフォルト）
```
k(x, x') = σ² * exp(-||x - x'||² / (2l²))
```
- `σ²`: outputscale（信号分散）
- `l`: lengthscale（特徴スケール）

**学習**:
- 尤度: `ExactMarginalLogLikelihood`
- 最適化: L-BFGS-B（GPyTorchのデフォルト）
- ハイパーパラメータ: lengthscale, outputscale, noise_variance

**予測**:
```python
posterior = model.posterior(X_test)
mean = posterior.mean  # 予測平均
variance = posterior.variance  # 予測分散
```

### 入出力変換

**入力正規化** (`Normalize`):
- 各特徴量を [0, 1] にスケーリング
- 理由: カーネルの数値安定性向上、lengthscale学習の容易化

**出力標準化** (`Standardize`):
- 平均0、標準偏差1に正規化
- 理由: GPの仮定（平均0）に適合、数値安定性

### 獲得関数最適化

**`optimize_acqf` の動作**:
1. `raw_samples` 個のランダム点を生成
2. 獲得関数値を評価
3. 上位 `num_restarts` 個から多スタート最適化
4. 最良の候補点を返す

**パラメータ調整**:
- GPU: `num_restarts=10, raw_samples=128` （探索を強化）
- CPU: `num_restarts=5, raw_samples=64` （計算コスト削減）

---

## 仮想実験の実装

### MVP版の設計

**制約**: 実際の実験装置が利用できない

**解決策**: CCFatigueの既存データを「真の世界」とみなす

**実装**:
```python
class DatabaseEvaluator:
    def __init__(self, X_all, y_all, device):
        # 最近傍探索器の準備
        self.nn = NearestNeighbors(n_neighbors=1)
        self.nn.fit(X_all.cpu().numpy())
    
    def evaluate(self, x_candidate):
        # 最も近いデータ点を探索
        dist, idx = self.nn.kneighbors(x_candidate)
        # その点の寿命を返す
        return X_all[idx], y_all[idx]
```

**利点**:
- 高速評価（実験待ち時間なし）
- MVP検証に十分
- コードの他部分は実験装置APIと交換可能

**制限**:
- 連続空間での真の評価ではない
- データベース外の条件は評価できない

### 将来の実験装置統合

**インターフェース設計**:
```python
class ExperimentDevice:
    def evaluate(self, x_candidate):
        # 1. 実験条件を装置に送信
        # 2. 実験実行（時間がかかる）
        # 3. 測定結果を取得
        return x_measured, y_measured
```

**統合手順**:
1. `DatabaseEvaluator` → `ExperimentDevice` に置き換え
2. 他のコードは変更不要（インターフェース互換）
3. 非同期実験への拡張も容易

---

## 可視化戦略

### 1. 収束曲線 (`plot_convergence`)

**目的**: 最適化の進捗確認

**要素**:
- 青線: 累積最良値（単調非減少）
- 赤点: 各反復での観測値

**判断基準**:
- 上昇トレンド: 最適化が進んでいる
- 平坦: 局所最適または収束
- 急上昇: 良い条件の発見

### 2. 探索空間 (`plot_explored_space`)

**目的**: どの領域を探索したか確認

**要素**:
- 2D散布図（特徴量ペア）
- 色: 寿命（viridisカラーマップ）

**判断基準**:
- 均一な分布: 探索的
- 偏った分布: 活用的（良い領域に集中）
- 未探索領域: さらなる最適化の余地

### 3. GP予測分布 (`plot_gp_predictions`)

**目的**: モデルの理解

**要素**:
- 1Dスライス（他の変数は中央値固定）
- 青線: 予測平均
- 青帯: ±2σ信頼区間
- 赤点: 訓練データ

**判断基準**:
- 狭い信頼区間: 確信度が高い
- 広い信頼区間: 不確実性が高い（探索余地）
- データが少ない領域: 予測が不安定

### 4. 計算パフォーマンス (`plot_compute_performance`)

**目的**: モード間の比較

**要素**:
- 実行時間の棒グラフ
- GPUメモリ使用量の推移

**判断基準**:
- cpu_16cores vs no_parallel: 並列化の効果
- gpu_16gb: GPU加速の効果
- メモリ推移: 制約内に収まっているか

---

## パフォーマンス最適化

### CPU並列化

**実装**: `torch.set_num_threads(16)`

**効果**:
- BLAS/LAPACK演算の並列化
- 行列分解（GP学習）の高速化
- 期待速度向上: 2-4倍（線形代数演算の割合に依存）

**注意**:
- I/O待ちには効果なし
- ハイパースレッディングは効果限定的

### GPU加速

**効果**:
- テンソル演算の大幅高速化
- 期待速度向上: 5-10倍（問題サイズに依存）

**制約**:
- 16GBメモリ制限
- データ転送のオーバーヘッド
- 小規模問題ではCPUと同等の場合も

**最適化テクニック**:
- バッチサイズの動的調整
- `num_restarts`, `raw_samples` の調整
- 不要なCPU-GPU転送の削減

### メモリ管理

**GPU 16GB制約への対応**:
1. `optimize_acqf` のパラメータ調整
2. バッチサイズの制限
3. 勾配チェックポイント（将来の拡張）

**モニタリング**:
```python
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
```

---

## 制限事項と将来の拡張

### 現在の制限

1. **仮想実験**: 最近傍探索による近似
2. **単一目的**: 寿命のみを最大化
3. **制約なし**: 物理的制約の明示的扱いなし
4. **順次最適化**: q=1（1点ずつ）
5. **静的モデル**: 時間依存性なし

### 将来の拡張

#### 1. 実験装置統合
```python
class RealExperimentDevice:
    def evaluate(self, x_candidate):
        # 実機での実験
        return x_measured, y_measured
```

#### 2. 多目的最適化
- 寿命 + コスト + 製造容易性
- BoTorchの `qExpectedHypervolumeImprovement`

#### 3. 制約条件
- 物理的実現可能性
- BoTorchの制約付き獲得関数

#### 4. バッチ実験
- `q > 1` による複数条件の同時提案
- `qExpectedImprovement`

#### 5. 時系列モデル
- 時刻 t を明示的に組み込む
- 動的GP、時系列カーネル

#### 6. 分散計算
- 複数GPU対応
- クラスタ環境での並列実行

#### 7. アクティブラーニング
- 不確実性重視の探索
- `qUpperConfidenceBound`

---

## まとめ

このMVPは、コンセプト図に示された適応的実験計画を忠実に実装し、以下を実現しています:

1. **BoTorchベースの高度な最適化**: SingleTaskGP + EI
2. **柔軟な計算環境**: 3モード対応（no_parallel, cpu_16cores, gpu_16gb）
3. **包括的な可視化**: 収束、探索空間、GP予測、パフォーマンス
4. **実用的な設計**: モジュール化、拡張可能性、再現性

CCFatigueデータでの小規模ケーススタディを通じて、材料開発チームに革新的な方法論を提案する基盤が整いました。

---

**Material Usability Maximizer MVP**  
**Version 0.1.0**  
**2026年5月10日**
