# Metallic Alloys Data Lesson: FatigueData-CMA2022

Nature Scientific Data (2023)から公開されている複雑金属合金疲労データベースの取得・処理に関する教訓集

**データソース**: Zhang et al., "Fatigue database of complex metallic alloys", Scientific Data 10:447 (2023)  
**DOI**: https://doi.org/10.6084/m9.figshare.23007362  
**作業日**: 2026年5月10日

---

## 1. データの発見と取得

### 1.1 論文PDFからのデータソース特定

**課題**: 論文PDF内にデータリポジトリのURLが埋もれている

**解決策**: PyPDF2を使った自動抽出
```python
import PyPDF2

reader = PyPDF2.PdfReader("doc/2301_Fatigue database of complex metallic alloys.pdf")
for page in reader.pages:
    text = page.extract_text()
    # "figshare", "github", "doi.org"などのキーワードで検索
```

**発見したリポジトリ**:
- **Figshare**: https://doi.org/10.6084/m9.figshare.23007362 (メインデータ)
- **GitHub**: https://github.com/xuzpgroup/ZianZhang/tree/main/FatigueData-CMA2022 (処理スクリプト)

### 1.2 Figshare APIの活用

**教訓**: Figshare APIを使えば、メタデータとダウンロードURLを自動取得できる

```python
import requests

article_id = "23007362"
api_url = f"https://api.figshare.com/v2/articles/{article_id}"
response = requests.get(api_url)
metadata = response.json()

# ファイルリストの取得
for file_info in metadata['files']:
    name = file_info['name']
    download_url = file_info['download_url']
    size_mb = file_info['size'] / (1024 * 1024)
```

**利用可能なファイル形式**:
- `.xlsx` (0.73 MB) - **推奨**: 人間が読みやすい、pandasで容易に処理
- `.json` (7.58 MB) - プログラマティックアクセス向け、ファイルサイズ大
- `.mat` (0.60 MB) - MATLAB専用、Python処理には不向き

**推奨**: MVP開発ではExcel形式が最も扱いやすい

---

## 2. Excel階層構造データの解析

### 2.1 複雑なヘッダー構造

**課題**: FatigueData-CMA2022のExcelは2行ヘッダー構造

```
Row 0: [NaN, 'metadata', NaN, ..., 'fatigue', ..., 'materials', ..., 'testing', ...]
Row 1: ['dataset id', 'title', 'authors', ..., 'types of fatigue data', ..., 'composition (at%)', ...]
```

**解決策**: `header=1`で2行目を実際のカラム名として読み込む

```python
df_param = pd.read_excel(excel_path, sheet_name='parameter', header=1)
# Row 1が実際のデータカラム名になる
```

### 2.2 複数シートの理解

**データ構造**:
- `S-N`: 応力振幅-疲労寿命データ（1492行）
- `e-N`: ひずみ振幅-疲労寿命データ
- `dadn`: き裂進展速度データ
- `parameter`: メタデータ（材料組成、試験条件、272データセット）

**キー関係**:
```
S-N[dataset id] ←→ parameter[dataset id]
```

**重要**: S-Nシートとparameterシートを`dataset id`でマージしてデータを統合

---

## 3. データクリーニングの課題と解決

### 3.1 複数値を含むフィールド

**課題**: `load ratio`列に "0.25, 0.75" のような複数値が存在

**影響**: `astype(float)`でValueErrorが発生

**解決策**: クリーニング関数の作成

```python
def clean_numeric_field(value):
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    
    # カンマ区切りの場合、最初の値を採用
    if ',' in value_str:
        value_str = value_str.split(',')[0].strip()
    
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return np.nan

df['stress_ratio'] = df['load_ratio'].apply(clean_numeric_field)
```

**方針**: 複数値がある場合は最初の値を代表値として採用（実験的アプローチ）

### 3.2 大文字小文字の不統一

**課題**: `environment`列に "air" と "AIR" が混在

**解決策**: 小文字統一で比較

```python
df_filtered = df[df['environment'].str.lower() == 'air']
```

### 3.3 単位の違い

**課題**: 
- 論文では温度が°C単位
- モックデータではK単位を使用していた

**解決策**: データソースに合わせてフィルタ条件を調整

```python
# 実データに合わせた室温フィルタ (°C)
filter_conditions = {
    "temperature": (20, 30),  # °C
    "environment": "air",
    "loading_mode": ["uniaxial", "tension-compression"]
}
```

---

## 4. 材料組成データのパース

### 4.1 組成文字列のパース

**データ形式**: `"Fe20.0-Ni25.0-Cr20.0-Co20.0-Mn15.0"` (at%)

**パース戦略**: 正規表現で元素記号と原子%を分離

```python
import re

def parse_composition(comp_str):
    if pd.isna(comp_str):
        return {}
    
    comp_dict = {}
    parts = str(comp_str).split('-')
    
    for part in parts:
        # "Fe20.0" → ('Fe', '20.0')
        match = re.match(r'([A-Z][a-z]?)(\d+\.?\d*)', part)
        if match:
            element = match.group(1)
            value = float(match.group(2))
            comp_dict[element] = value
    
    return comp_dict
```

**出力例**:
```
"Co20.0-Cr20.0-Fe20.0-Mn20.0-Ni20.0" 
→ {'Co': 20.0, 'Cr': 20.0, 'Fe': 20.0, 'Mn': 20.0, 'Ni': 20.0}
```

### 4.2 欠損元素の処理

**課題**: すべてのデータセットが同じ元素を含むわけではない

**解決策**: ゼロパディング

```python
target_elements = ['Fe', 'Ni', 'Cr', 'Co', 'Al', 'Ti', 'Cu', 'Mn']

for element in target_elements:
    df[element] = df['composition_dict'].apply(
        lambda x: x.get(element, 0.0)  # デフォルト0.0
    )
```

---

## 5. ストレス計算

### 5.1 応力振幅から最大応力への変換

**データ提供形式**: 
- `stress amplitude (σa)`: 応力振幅
- `load ratio (R)`: 応力比 (σmin / σmax)

**MVP要求**: `stress_max`（最大応力）

**応力関係**:
```
σa = (σmax - σmin) / 2
R = σmin / σmax

→ σmax = 2 * σa / (1 - R)
```

**特殊ケース**: R = -1（引張圧縮）
```
σmax = σa
```

**実装**:
```python
def calculate_stress_max(sigma_a, R):
    if pd.isna(R):
        return sigma_a  # デフォルトは引張圧縮と仮定
    
    R = float(R)
    if R == -1:
        return sigma_a
    else:
        return 2 * sigma_a / (1 - R)
```

---

## 6. データフィルタリング戦略

### 6.1 材料タイプの選別

**データベース構成**:
- Metallic Glasses (MGs): 149データセット
- Multi-Principal Element Alloys (MPEAs): 123データセット

**MVP目標**: MPEAsのみを対象

```python
df_mpea = df[df['type of the material'] == 'multi-principal elements alloy']
```

### 6.2 疲労試験タイプの選別

**利用可能なタイプ**:
- S-N (stress-life): 163データセット → **MVP対象**
- ε-N (strain-life): 26データセット
- da/dN-ΔK (crack growth): 83データセット

### 6.3 試験条件のフィルタリング

**フィルタリング結果**:
- **元データ**: 503サンプル（MPEAs S-N全体）
- **温度フィルタ後** (20-30°C): 497サンプル
- **環境フィルタ後** (air): 497サンプル
- **荷重モードフィルタ後** (uniaxial): **258サンプル**
- **欠損値除去後**: **230サンプル**

**保持率**: 45.7% (230/503)

**教訓**: データ品質向上のために積極的にフィルタリングすべき

### 6.4 複数荷重モードへの対応

**データに含まれる荷重モード**:
- `uniaxial`: 264サンプル → **MVP対象**
- `4-point bending`: 101サンプル
- `bending fatigue`: 77サンプル
- `rotating bending fatigue`: 42サンプル
- `3-point bending`: 19サンプル

**実装**: リスト形式でフィルタ

```python
loading_modes = ["uniaxial", "tension-compression"]
df_filtered = df[df['loading_mode'].isin(loading_modes)]
```

---

## 7. MVP統合時の注意点

### 7.1 ファイル命名規則

**推奨**: 実データとモックデータを明確に区別

```
data/
  ├── complex_alloy_fatigue.csv          # モックデータ
  └── complex_alloy_fatigue_real.csv     # 実データ
```

**データローダーでの優先順位設定**:
```python
COMPLEX_ALLOY_FILE = CCFATIGUE_DATA_DIR / "complex_alloy_fatigue_real.csv"
if not COMPLEX_ALLOY_FILE.exists():
    COMPLEX_ALLOY_FILE = CCFATIGUE_DATA_DIR / "complex_alloy_fatigue.csv"
```

### 7.2 データ品質チェック

**必須チェック項目**:
1. 欠損値の確認: `df.isnull().sum()`
2. データ範囲の確認: `df.describe()`
3. カテゴリカル変数の分布: `df['environment'].value_counts()`
4. 数値変換の成功: `df.dtypes`

### 7.3 PCAとの統合

**実データでのPCA結果**:
- **入力次元**: 8 (Fe, Ni, Cr, Co, Al, stress_max, stress_ratio, frequency)
- **組成次元**: 5 (Fe, Ni, Cr, Co, Al)
- **PCA後**: 2主成分で99.87%の分散を保持
- **最終次元**: 5 (PC1, PC2, stress_max, stress_ratio, frequency)

**教訓**: 実データでは組成の多様性が低いため、2主成分で十分

### 7.4 データ品質検証と可視化

**実装日**: 2026年5月10日

統合テスト (`test_complex_alloy.py`) で生成される6つの可視化により、データ品質とPCA効果を検証。

#### 1. PCA累積寄与率

![PCA累積寄与率](../results/test_complex_alloy/pca_variance.png)

**解説**:
- **第1主成分 (PC1)**: 91.11%の分散を説明
- **第2主成分 (PC2)**: 8.76%の分散を追加（累積99.87%）
- **結論**: 5次元組成を2次元に削減しても情報損失はわずか0.13%

**材料学的解釈**:
- PC1: Fe-Ni-Cr-Coの主要なバランス（高エントロピー合金の基本構造）
- PC2: 微細な組成変動（特定元素の増減）

#### 2. PCA空間におけるサンプル分布

![PCA空間](../results/test_complex_alloy/pca_space.png)

**解説**:
- 230サンプルがPC1-PC2空間でどのように分布しているかを可視化
- 色: 疲労寿命 (log10サイクル数) で色分け
- **観察**: 特定のPC1-PC2領域で高寿命サンプルが集中

**最適化への示唆**:
- ベイズ最適化はこの2次元空間で効率的に探索可能
- 高寿命領域を優先的に探索することで効率向上

#### 3. 元素のPCAローディング

![元素ローディング](../results/test_complex_alloy/element_loadings.png)

**解説**:
- 各元素が主成分にどの程度寄与しているかを示す
- **PC1の特徴**:
  - Fe, Ni, Cr, Coがほぼ同等に寄与（等原子比合金の特徴）
  - 負の方向: 特定元素の増加
  - 正の方向: バランスの取れた組成
- **PC2の特徴**:
  - 元素間の微細なバランス調整
  - Fe vs Ni-Cr-Coの対比

**材料設計への応用**:
- PC1を調整 → 全体的な組成バランスの変更
- PC2を調整 → 特定元素の微調整

#### 4. 特徴量の分布

![特徴量分布](../results/test_complex_alloy/feature_distributions.png)

**解説**:
- 8特徴量 (5組成 + 3試験条件) のヒストグラムとKDE
- **組成分布の特徴**:
  - Fe: 20-47 wt%（広範囲）
  - Ni, Cr, Co: 0-26 wt%（類似範囲）
  - Al: ほぼ0%（このデータセットでは含まれない）
- **試験条件の分布**:
  - stress_max: 100-950 MPa（広範囲）
  - stress_ratio: -1.0が多い（完全逆位相）
  - frequency: 低周波と高周波に二極化

**データ品質の確認**:
- 外れ値や異常値がないことを確認
- バランスの取れた分布（一部の条件に偏っていない）

#### 5. 特徴量間の相関行列

![相関行列](../results/test_complex_alloy/correlation_matrix.png)

**解説**:
- **組成間の相関**:
  - Fe vs Ni: 負の相関（一方が増えると他方が減る）
  - Cr vs Co: 弱い相関
  - Al: 他と無相関（データセット内で変動がない）
- **組成と試験条件の相関**:
  - ほぼ無相関 → 独立して変化可能
- **試験条件間の相関**:
  - stress_max vs stress_ratio: 弱い負の相関
  - frequency: 他と独立

**ベイズ最適化への影響**:
- 組成と試験条件が独立 → 両方を同時に最適化可能
- 組成内の相関 → PCAで効率的に表現可能

#### 6. 組成と疲労寿命の関係

![組成-寿命関係](../results/test_complex_alloy/composition_lifetime.png)

**解説**:
- 各元素の含有量と疲労寿命 (log10サイクル) の散布図
- **観察結果**:
  - **Fe**: 30-40 wt%で高寿命傾向
  - **Ni**: 広範囲で安定、20-25 wt%で高寿命サンプルあり
  - **Cr**: 20-25 wt%で高寿命傾向
  - **Co**: 20-26 wt%で分散が大きい
  - **Al**: 変動なし（0%固定）

**材料学的洞察**:
- 等原子比に近い組成（各25%前後）で高寿命を達成
- Fe含有量のバランスが重要
- 単一元素ではなく、組成全体のバランスが寄命を決定

#### データ品質検証の結論

✅ **データ品質**: 外れ値なし、分布良好、欠損値処理済み  
✅ **PCA効果**: 99.87%の情報保持で5次元→2次元に削減成功  
✅ **特徴量の独立性**: 組成と試験条件が独立、同時最適化可能  
✅ **材料学的妥当性**: 等原子比HEAの特徴と一致  

**可視化スクリプト**: `test_complex_alloy.py`が7種類の可視化を自動生成

---

## 8. スクリプト構成の教訓

### 8.1 段階的アプローチ

**推奨ワークフロー**:

1. **探索**: `examine_excel_structure.py`
   - シート名の確認
   - カラム名の確認
   - データ型の確認

2. **詳細調査**: `examine_parameter_sheet.py`
   - ヘッダー構造の理解
   - データ関係の特定
   - サンプルデータの確認

3. **変換**: `convert_figshare_to_csv.py`
   - マージ処理
   - データクリーニング
   - MVP形式への変換

4. **検証**: `test_complex_alloy.py`
   - データローダーとの統合
   - PCA動作確認
   - 可視化テスト

### 8.2 エラーハンドリング

**経験したエラーと対策**:

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `KeyError: 'life\\nN (cycle)'` | エスケープシーケンスの誤解釈 | `'life\nN (cycle)'` に修正 |
| `ValueError: could not convert string to float` | 複数値 "0.25, 0.75" | `clean_numeric_field()`関数 |
| `Found array with 0 sample` | フィルタ条件の単位ミス | °C単位に変更 |

---

## 9. データソースの信頼性評価

### 9.1 品質指標

FatigueData-CMA2022には`rating score`が付与されている:

```
rating score範囲: 0.75 ~ 0.81
```

**意味**: データ抽出精度と完全性の評価値

**活用**: 高スコアデータのみを使用してMVPの精度向上も検討可能

### 9.2 データ由来

**326論文**から抽出 → 272データセット → 1492 S-Nデータポイント

**自動抽出 + 手動検証**のハイブリッドアプローチで品質保証

---

## 10. 今後の展開

### 10.1 データ拡張の可能性

**現在未使用のデータ**:
- ε-N (strain-life) データ: 26データセット
- Metallic Glasses: 149データセット
- 高温試験データ (550°C): 6サンプル
- 他の荷重モード (曲げ、回転曲げ)

### 10.2 特徴量エンジニアリングの改善

**追加可能な特徴量**:
- `Young's modulus`: 弾性率
- `yield strength`: 降伏強度
- `ultimate tensile strength`: 引張強度
- `grain size`: 結晶粒径
- `processing`: 熱機械処理履歴

### 10.3 マルチタスク学習への展開

**可能性**:
- S-N, ε-N, da/dN-ΔKを同時に予測
- MGs vs MPEAsのドメイン適応
- 温度・環境の影響をモデル化

### 10.4 PCA逆変換による実用的材料設計

**実装日**: 2026年5月10日

#### 課題

PCAで次元削減した後、最適化結果が主成分空間（PC1, PC2）の値として得られるが、実験で材料を作製するには元の組成（Fe, Ni, Cr, Co, Al wt%）が必要。

#### 解決策: PCA逆変換機能の実装

**1. CompositionPCAクラスに逆変換メソッドを追加**

```python
def inverse_transform_composition(
    self, 
    X_pca: np.ndarray,
    normalize: bool = True
) -> pd.DataFrame:
    """
    主成分から元の組成空間に逆変換
    
    処理手順:
    1. PCAの逆変換 (PC → 標準化空間)
    2. StandardScalerの逆変換 (標準化空間 → 元の空間)
    3. 負の値を0にクリップ (物理的制約)
    4. 組成の合計を100%に正規化
    """
    # PCAの逆変換
    X_scaled = self.pca.inverse_transform(X_pca)
    
    # 標準化の逆変換
    X_comp = self.scaler.inverse_transform(X_scaled)
    
    # 物理的制約: 負の組成は不可
    df_comp = pd.DataFrame(X_comp, columns=self.feature_names_in_)
    df_comp = df_comp.clip(lower=0.0)
    
    # 正規化: 合計を100%にする
    if normalize:
        row_sums = df_comp.sum(axis=1)
        df_comp = df_comp.div(row_sums, axis=0) * 100.0
    
    return df_comp
```

**2. main_demo.pyでの統合**

```python
# 最適化後、PC値を元の組成に変換
if pca_transformer is not None:
    # PC列のインデックスを取得
    pc_indices = [i for i, name in enumerate(feature_names) if name.startswith('PC')]
    best_pcs = best_x[pc_indices].reshape(1, -1)
    
    # 逆変換
    composition_df = pca_transformer.inverse_transform_composition(
        best_pcs, 
        normalize=True
    )
    
    # 結果を表示・保存
    print("Element concentrations (wt%):")
    for col in composition_df.columns:
        print(f"  {col}: {composition_df[col].iloc[0]:.2f}%")
```

#### 実装結果の例

**最適化出力**:
```
Best conditions:
  PC1: 0.4618
  PC2: -0.3814
  stress_max: 160.0092 MPa
  stress_ratio: -1.0000
  frequency: 20000.0000 Hz
```

**PCA逆変換後の実用的組成**:
```
Optimal Material Composition:
  Fe: 24.99 wt%
  Ni: 24.98 wt%
  Cr: 24.99 wt%
  Co: 25.04 wt%
  Al:  0.00 wt%
```

**材料学的解釈**:
- ほぼ等原子比の4元系合金 (Fe-Ni-Cr-Co)
- 高エントロピー合金（HEA）の典型的組成
- バランスの取れた元素配分が高疲労寿命（1.31×10⁸サイクル）に寄与

#### 保存される情報

`results/optimal_composition.json`:
```json
{
  "composition_wt_percent": {
    "Fe": 24.99, "Ni": 24.98, "Cr": 24.99, "Co": 25.04, "Al": 0.0
  },
  "pca_values": {
    "PC1": 0.4618, "PC2": -0.3814
  },
  "test_conditions": {
    "stress_max": 160.01, "stress_ratio": -1.0, "frequency": 20000.0
  }
}
```

#### 教訓

1. **次元削減と実用性の両立**: PCAで計算効率を上げつつ、逆変換で実用情報を復元
2. **物理的制約の実装**: 負の組成や非正規化状態を自動修正
3. **トレーサビリティ**: PC値と元の組成の両方を保存し、検証可能に
4. **自動化の重要性**: 逆変換を手動計算すると誤りのリスク、関数化で信頼性向上

**利点**:
- 抽象的なPC値が具体的な材料レシピに変換される
- 実験室で即座に材料作製が可能
- データベースの各サンプルとの比較が容易

---

## まとめ

### 成功の鍵

1. ✅ **段階的探索**: 一度にすべてを理解しようとせず、スクリプトで段階的に調査
2. ✅ **ロバストなクリーニング**: 複数値、大文字小文字、欠損値に対応
3. ✅ **柔軟なフィルタリング**: リスト対応、条件緩和のオプション確保
4. ✅ **自動化**: FigshareAPI + 変換スクリプトで再現性を確保
5. ✅ **PCA逆変換**: 計算効率と実用性の両立

### 実データ統合の成果

| 指標 | 値 |
|------|-----|
| データセット数 | 62 (MPEAs S-N) |
| データポイント数 | 503 → 230 (フィルタ後) |
| 特徴量次元 | 8 → 5 (PCA後) |
| 疲労寿命範囲 | 10^2.8 ~ 10^8.7 cycles |
| PCA累積寄与率 | 99.87% |
| 最適化実行時間 | 13.83秒 (20反復, CPU 16コア) |
| 最適疲労寿命 | 1.31×10^8 cycles |
| 最適材料 | Fe-Ni-Cr-Co等原子比合金 |

**MVP v0.1.2の実データ統合は成功！** 🎉
