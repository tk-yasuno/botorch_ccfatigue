# Complex Metallic Alloys Fatigue Database — Data Source Guide

## 論文情報

**Title**: Fatigue database of complex metallic alloys  
**Authors**: Zian Zhang, Haoxuan Tang, Zhiping Xu  
**Journal**: Scientific Data, Volume 10, Article 447 (2023)  
**Publisher**: Nature Publishing Group  
**DOI**: [10.1038/s41597-023-02347-z](https://doi.org/10.1038/s41597-023-02347-z)  
**Publication Date**: July 2023

## データセット概要

このデータベースは、複雑金属材料（メタリックガラス、多主元素合金など）の疲労特性を包括的にまとめたものです。

### 含まれるデータ種別

1. **S–N曲線データ** (応力–寿命)
   - 応力振幅 vs 疲労寿命
   - 破断までの繰返し数 (Nf)

2. **ε–N曲線データ** (ひずみ–寿命)
   - ひずみ振幅 vs 疲労寿命

3. **da/dN–ΔK曲線データ** (き裂進展)
   - き裂進展速度 vs 応力拡大係数範囲

### データセットの特徴

- **収録期間**: 2022年末までの文献から整理
- **材料種別**: 
  - 多主元素合金 (High-Entropy Alloys)
  - メタリックガラス (Metallic Glass)
  - その他複雑合金系
- **試験条件の多様性**:
  - 応力レベル（最大応力、応力比R）
  - 試験周波数
  - 環境条件（温度、雰囲気）
  - 荷重モード（引張圧縮、曲げ、ねじり）

## データ取得方法

### オプション1: Nature Scientific Data公式サイト

1. 論文ページにアクセス:
   ```
   https://www.nature.com/articles/s41597-023-02347-z
   ```

2. "Supplementary Information" または "Data availability" セクションからデータファイルをダウンロード

3. データファイル形式: CSV, Excel, またはデータリポジトリへのリンク

### オプション2: Figshare / Zenodo リポジトリ

Nature Scientific Dataの論文は通常、データを以下のリポジトリで公開しています:

- **Figshare**: https://figshare.com/ で論文タイトルまたはDOIで検索
- **Zenodo**: https://zenodo.org/ で検索

### オプション3: 著者への直接連絡

データが上記で見つからない場合:
- 論文に記載されている著者の連絡先にメールで問い合わせ
- データ共有ポリシーに従ってアクセスをリクエスト

## MVP v0.1.2 での使用方法

### 1. データファイルのダウンロード

取得したデータファイルを以下のディレクトリに配置:
```
data/complex_alloy_fatigue.csv
```

### 2. データフォーマット確認

期待される列（最低限必要なもの）:
- **組成情報**: 元素濃度（例: Fe, Ni, Cr, Co, Al など）
- **試験条件**: 
  - `stress_max`: 最大応力 (MPa)
  - `stress_ratio` または `R`: 応力比
  - `frequency`: 試験周波数 (Hz)
  - `temperature`: 試験温度 (K または °C)
  - `environment`: 雰囲気（空気、真空など）
  - `loading_mode`: 荷重モード（引張圧縮、曲げなど）
- **出力変数**: 
  - `cycles_to_failure` または `Nf`: 破断までの繰返し数

### 3. データフィルタリング（MVP v0.1.2）

初期実装では以下の条件に絞ることを推奨:
- 室温試験のみ (`temperature` ≈ 293K or 20°C)
- 空気中試験のみ (`environment` = "air" or "ambient")
- 一定振幅疲労のみ（変動荷重を除外）

### 4. 設定ファイルで指定

`config.yaml` で以下を設定:
```yaml
optimization:
  dataset_name: complex_alloy
  design_variables:
    - Fe  # 元素濃度
    - Ni
    - Cr
    - Co
    - stress_max
    - stress_ratio
    - frequency
  objective_variable: cycles_to_failure
  objective_transform: log10
```

## モックデータ（実装開発用）

実際のデータが入手できない場合、以下のモックデータで開発を進めることができます:

```python
# 生成方法は data/generate_mock_complex_alloy.py に記載
```

モックデータには以下が含まれます:
- 5元素系合金（Fe, Ni, Cr, Co, Al）の組成
- 応力レベル、応力比、周波数
- 簡易的なS-Nモデルから生成した疲労寿命

## 参考文献

Zhang, Z., Tang, H. & Xu, Z. Fatigue database of complex metallic alloys. Sci Data 10, 447 (2023). https://doi.org/10.1038/s41597-023-02347-z

## 注意事項

- データの使用にあたっては、論文の引用とデータ使用許諾条件を確認してください
- 研究目的での使用を前提としています
- 商用利用の場合は著者またはNature Publishing Groupに確認が必要な場合があります
