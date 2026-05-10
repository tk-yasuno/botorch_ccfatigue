# CCFatigue データ取得ガイド

**作成日**: 2026年5月10日  
**目的**: EPFLのCCFatigue Platformから材料疲労データを取得し、BoTorch MVPで活用する方法を記録

---

## 1. CCFatigue Platformとは

**CCFatigue** は、EPFL（スイス連邦工科大学ローザンヌ校）が提供する材料疲労データプラットフォームです。

- **公式サイト**: https://ccfatigue.epfl.ch
- **GitHubリポジトリ**: https://github.com/EPFL-ENAC/CCFatiguePlatform
- **目的**: 複合材料の疲労試験データの収集・管理・共有
- **データ形式**: CSV, JSON
- **ライセンス**: オープンソース

---

## 2. データセットの種類

CCFatigue Platformには以下のデータタイプが存在します：

### 2.1 S-N曲線データ (SNC: S-N Curve)
- **ファイル**: `Data/samples/SNC_sample_2022-09.csv`
- **サンプル数**: 762行
- **カラム**:
  - `stress_ratio`: 応力比 (R値)
  - `stress_max`: 最大応力 [MPa]
  - `cycles_to_failure`: 破壊までのサイクル数
  - `stress_lowerbound`: 応力下限
  - `stress_upperbound`: 応力上限

### 2.2 疲労試験データ (FA: Fatigue)
- **ディレクトリ**: `Data/preprocessed/TST_Nehme_2025-05_FA/`
- **ファイル**: `tests.csv`, `measure_XXX.csv`, `experiment.json`
- **内容**: 実験条件、荷重履歴、ヒステリシスループ

### 2.3 その他のデータ
- **AGG**: Aggregate data（集計データ）
- **CYC**: Cyclic data（サイクリックデータ）
- **LDS**: Load sequence data（荷重シーケンス）

---

## 3. データ取得方法

### 3.1 GitHubから直接ダウンロード

**PowerShellコマンド**:
```powershell
# dataディレクトリを作成
New-Item -ItemType Directory -Force -Path "data"

# S-N曲線データをダウンロード
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/EPFL-ENAC/CCFatiguePlatform/develop/Data/samples/SNC_sample_2022-09.csv" `
  -OutFile "data\ccfatigue_snc.csv"

# テストデータをダウンロード（オプション）
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/EPFL-ENAC/CCFatiguePlatform/develop/Data/preprocessed/TST_Nehme_2025-05_FA/tests.csv" `
  -OutFile "data\ccfatigue_tests.csv"
```

**Bashコマンド** (Linux/Mac):
```bash
mkdir -p data
curl -o data/ccfatigue_snc.csv \
  https://raw.githubusercontent.com/EPFL-ENAC/CCFatiguePlatform/develop/Data/samples/SNC_sample_2022-09.csv
```

### 3.2 GitHub APIで利用可能なファイルを検索

```powershell
$response = Invoke-RestMethod `
  -Uri "https://api.github.com/repos/EPFL-ENAC/CCFatiguePlatform/contents/Data/samples" `
  -Headers @{"Accept"="application/vnd.github.v3+json"}
$response | Select-Object name, size, download_url
```

---

## 4. データ構造の詳細

### 4.1 SNC_sample_2022-09.csv
```csv
stress_ratio,cycles_to_failure,stress_max,stress_lowerbound,stress_upperbound
0.1000000,1.000000,32.90580,30.40634,36.37630
0.1000000,51.00000,25.07248,23.79429,26.74287
0.1000000,101.0000,23.91540,22.78930,25.36434
...
```

**特徴**:
- 応力比（R値）が主に0.1で固定
- 最大応力範囲: 5.53～32.91 MPa
- サイクル数範囲: 1～10^9 サイクル（対数スケール）
- 信頼区間付き（lower/upper bound）

### 4.2 tests.csv（FA実験データ）
```csv
sequential number,specimen name,number of cycles,maximum load,run out,width,thickness
1,T1-FC-8-B5,1500,1064.925258,False,13.17,3.83
2,T1-FC-9-B5,14958,908.358634,False,13.21,3.92
...
```

**特徴**:
- 試験片ごとの詳細情報
- 荷重データから応力計算が必要（stress = load / (width × thickness)）
- ランアウト情報（run out: 破壊前に試験終了）

---

## 5. MVP実装への統合

### 5.1 src/data_loader.py の修正

**修正前**（ccfatigueパッケージ依存）:
```python
try:
    from ccfatigue.datasets import load_dataset
    CCFATIGUE_AVAILABLE = True
except ImportError:
    CCFATIGUE_AVAILABLE = False
```

**修正後**（ローカルCSVファイルから読み込み）:
```python
from pathlib import Path

CCFATIGUE_DATA_DIR = Path(__file__).parent.parent / "data"
CCFATIGUE_SNC_FILE = CCFATIGUE_DATA_DIR / "ccfatigue_snc.csv"
CCFATIGUE_TESTS_FILE = CCFATIGUE_DATA_DIR / "ccfatigue_tests.csv"
```

### 5.2 データ読み込み関数

```python
def load_ccfatigue_data(
    dataset_name: str = "snc",
    design_variables: Optional[List[str]] = None,
    objective_variable: str = "cycles_to_failure",
    objective_transform: str = "log10"
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    if design_variables is None:
        design_variables = ["stress_max", "stress_ratio"]
    
    if dataset_name == "snc" and CCFATIGUE_SNC_FILE.exists():
        df = pd.read_csv(CCFATIGUE_SNC_FILE)
        # ... 処理 ...
```

### 5.3 config.yaml の設定

```yaml
optimization:
  dataset_name: snc          # "snc" または "tests"
  design_variables:
    - stress_max
    - stress_ratio
  objective_variable: cycles_to_failure
  objective_transform: log10
```

---

## 6. データの前処理

### 6.1 対数変換

疲労寿命は対数正規分布に従うため、**log10変換**を適用：

```python
y_transformed = np.log10(cycles_to_failure)
```

**理由**:
- S-N曲線は両対数プロットで直線的
- ガウス過程モデルの適合性向上
- 桁違いの寿命差を適切に扱える

### 6.2 応力比の範囲調整

元データには異常値（stress_ratio = 10.0など）が含まれる場合があります：

```python
# 妥当な範囲でフィルタリング
df = df[(df['stress_ratio'] >= -1.0) & (df['stress_ratio'] <= 1.0)]
```

### 6.3 欠損値処理

```python
df_clean = df[design_variables + [objective_variable]].dropna()
```

---

## 7. 動作確認

### 7.1 データ読み込みテスト

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from data_loader import load_ccfatigue_data

X_df, y_df, design_vars, obj_var = load_ccfatigue_data(dataset_name='snc')
print(f'✓ Data loaded: X={X_df.shape}, y={y_df.shape}')
print(f'X range: stress_max [{X_df.stress_max.min():.2f}, {X_df.stress_max.max():.2f}]')
print(f'y range (log10): [{y_df.min():.2f}, {y_df.max():.2f}]')
"
```

**期待される出力**:
```
[DataLoader] Loading CCFatigue S-N curve data from: .../data/ccfatigue_snc.csv
[DataLoader] Dataset shape: (762, 5)
[DataLoader] ✓ Using real CCFatigue data with 762 samples
✓ Data loaded: X=(762, 2), y=(762,)
X range: stress_max [5.53, 32.91]
y range (log10): [0.00, 9.12]
```

### 7.2 MVP実行テスト

```bash
python main_demo.py --M 10 --T 20 --verbose
```

---

## 8. データの特徴と注意点

### 8.1 利点
- ✓ **実験データ**: 実際の疲労試験結果
- ✓ **サンプル数**: 762点（S-N曲線）で十分な学習データ
- ✓ **信頼区間付き**: 不確実性情報が利用可能
- ✓ **オープンデータ**: 自由に利用可能

### 8.2 注意点
- ⚠ **固定パラメータ**: stress_ratio = 0.1 が大半（変動少ない）
- ⚠ **単一材料**: データセット内の材料特性は限定的
- ⚠ **スケール**: MPa単位、実際の荷重ではない
- ⚠ **ランアウト**: 一部データは破壊前に試験終了（右打ち切り）

### 8.3 最適化への影響
- **2次元探索空間**: stress_max と stress_ratio のみ
- **目的**: 寿命（cycles_to_failure）の最大化
- **制約**: 実用的な応力範囲内（5～35 MPa程度）

---

## 9. 今後の拡張

### 9.1 追加データソース
- TST_Nehme_2025-05_FA の詳細測定データを統合
- 他の材料種類（AGG, CYCデータ）の活用
- 荷重履歴（Loops_measure）の時系列解析

### 9.2 特徴量エンジニアリング
- 応力振幅: `stress_amplitude = stress_max * (1 - stress_ratio) / 2`
- 平均応力: `stress_mean = stress_max * (1 + stress_ratio) / 2`
- Goodman線図やHaigh線図との照合

### 9.3 モデル改善
- 物理制約の組み込み（Basquin式、Coffin-Manson則）
- 複数材料の同時モデリング
- 打ち切りデータの生存時間解析

---

## 10. 参考資料

### 10.1 CCFatigue関連
- CCFatigue Platform: https://ccfatigue.epfl.ch
- GitHub Repository: https://github.com/EPFL-ENAC/CCFatiguePlatform
- Data Convention: `Data/SNC_Data_Convention.md`

### 10.2 疲労解析
- S-N曲線理論: Basquin's law, Goodman diagram
- 信頼性工学: Weibull distribution, Kaplan-Meier estimator
- 実験計画法: DOE for fatigue testing

### 10.3 ベイズ最適化
- Expected Improvement under Physical Constraints
- Multi-task GP for different materials
- Safe Bayesian Optimization for fatigue-critical applications

---

## まとめ

### ✅ 完了した作業
1. CCFatigue GitHubリポジトリの特定
2. S-N曲線データ（762サンプル）のダウンロード
3. `src/data_loader.py` の実データ対応修正
4. `config.yaml` の設定更新（snc、2変数設計空間）
5. データ読み込みの動作確認

### 📊 取得データ統計
- **データソース**: EPFL CCFatigue Platform
- **データセット**: SNC_sample_2022-09.csv
- **サンプル数**: 762
- **設計変数**: stress_max [5.53, 32.91 MPa], stress_ratio [-1.0, 10.0]
- **目的変数**: cycles_to_failure [1, 10^9] (log10スケール)

### 🎯 MVPでの利用
- `main_demo.py` と `demo_notebook.ipynb` で自動的に実データを使用
- モックデータへのフォールバック機能を維持
- ベイズ最適化による材料寿命最大化の実証が可能

---

**作成者**: GitHub Copilot (Claude Sonnet 4.5)  
**プロジェクト**: Material Usability Maximizer MVP - BoTorch Implementation
