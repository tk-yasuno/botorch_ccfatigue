"""
特徴量エンジニアリングモジュール

PCAによる次元削減、カテゴリ変数エンコーディング、
特徴量スケーリングなどの前処理機能を提供します。

MVP v0.1.2で追加。
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, Dict, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path


class CompositionPCA:
    """
    組成ベクトルに対するPCA（主成分分析）クラス
    
    材料組成の高次元データを低次元に削減し、
    ベイズ最適化の計算効率を向上させます。
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        random_state: int = 42
    ):
        """
        初期化
        
        Args:
            n_components: 主成分の数（Noneの場合は自動決定）
            variance_threshold: 累積寄与率の閾値（n_componentsがNoneの場合に使用）
            random_state: 乱数シード
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, composition_columns: list) -> 'CompositionPCA':
        """
        組成データにPCAをフィット
        
        Args:
            X: 全特徴量のDataFrame
            composition_columns: 組成を表すカラムのリスト（例: ["Fe", "Ni", "Cr"]）
        
        Returns:
            self
        """
        # 組成データを抽出
        X_comp = X[composition_columns].values
        self.feature_names_in_ = composition_columns
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X_comp)
        
        # PCAの実行
        if self.n_components is None:
            # 累積寄与率で自動決定
            pca_temp = PCA(random_state=self.random_state)
            pca_temp.fit(X_scaled)
            
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_comp = np.argmax(cumsum_var >= self.variance_threshold) + 1
            
            print(f"[CompositionPCA] Auto-selected {n_comp} components "
                  f"(cumulative variance: {cumsum_var[n_comp-1]:.2%})")
            
            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        else:
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        
        self.pca.fit(X_scaled)
        
        # 主成分の名前
        self.feature_names_out_ = [f"PC{i+1}" for i in range(self.pca.n_components_)]
        
        self.is_fitted = True
        
        print(f"[CompositionPCA] Fitted PCA with {self.pca.n_components_} components")
        print(f"[CompositionPCA] Explained variance ratio: "
              f"{self.pca.explained_variance_ratio_[:5]}")
        print(f"[CompositionPCA] Cumulative variance: "
              f"{np.sum(self.pca.explained_variance_ratio_):.2%}")
        
        return self
    
    def transform(
        self,
        X: pd.DataFrame,
        composition_columns: list,
        other_columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        組成データをPCAで変換し、他の特徴量と結合
        
        Args:
            X: 全特徴量のDataFrame
            composition_columns: 組成を表すカラムのリスト
            other_columns: 組成以外の特徴量カラムのリスト（Noneの場合は自動検出）
        
        Returns:
            変換後のDataFrame（主成分 + 他の特徴量）
        """
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet. Call fit() first.")
        
        # 組成データを抽出して変換
        X_comp = X[composition_columns].values
        X_scaled = self.scaler.transform(X_comp)
        X_pca = self.pca.transform(X_scaled)
        
        # 主成分のDataFrame作成
        df_pca = pd.DataFrame(X_pca, columns=self.feature_names_out_, index=X.index)
        
        # 他の特徴量を結合
        if other_columns is None:
            # 組成以外の列を自動検出
            other_columns = [col for col in X.columns if col not in composition_columns]
        
        if other_columns:
            df_other = X[other_columns]
            df_result = pd.concat([df_pca, df_other], axis=1)
        else:
            df_result = df_pca
        
        return df_result
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        composition_columns: list,
        other_columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        fitとtransformを連続実行
        
        Args:
            X: 全特徴量のDataFrame
            composition_columns: 組成を表すカラムのリスト
            other_columns: 組成以外の特徴量カラムのリスト
        
        Returns:
            変換後のDataFrame
        """
        self.fit(X, composition_columns)
        return self.transform(X, composition_columns, other_columns)
    
    def get_component_weights(self) -> pd.DataFrame:
        """
        各主成分における元素の寄与度を取得
        
        Returns:
            元素 × 主成分の寄与度行列
        """
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet. Call fit() first.")
        
        weights = pd.DataFrame(
            self.pca.components_.T,
            columns=self.feature_names_out_,
            index=self.feature_names_in_
        )
        return weights
    
    def get_explained_variance_info(self) -> Dict[str, Any]:
        """
        説明分散の情報を取得
        
        Returns:
            説明分散比、累積説明分散比などの辞書
        """
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet. Call fit() first.")
        
        return {
            "n_components": self.pca.n_components_,
            "explained_variance": self.pca.explained_variance_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(self.pca.explained_variance_ratio_),
            "singular_values": self.pca.singular_values_
        }
    
    def inverse_transform_composition(
        self, 
        X_pca: np.ndarray,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        主成分から元の組成空間に逆変換
        
        Args:
            X_pca: 主成分の値（numpy配列またはDataFrame）
            normalize: 組成の合計を100%に正規化するか
        
        Returns:
            元の組成空間のDataFrame（Fe, Ni, Cr, etc.）
        """
        if not self.is_fitted:
            raise ValueError("PCA is not fitted yet. Call fit() first.")
        
        # DataFrameの場合はnumpy配列に変換
        if isinstance(X_pca, pd.DataFrame):
            X_pca = X_pca.values
        
        # PCAの逆変換
        X_scaled = self.pca.inverse_transform(X_pca)
        
        # 標準化の逆変換
        X_comp = self.scaler.inverse_transform(X_scaled)
        
        # DataFrameに変換
        df_comp = pd.DataFrame(
            X_comp, 
            columns=self.feature_names_in_
        )
        
        # 負の値を0にクリップ（物理的に意味がない）
        df_comp = df_comp.clip(lower=0.0)
        
        # 正規化（組成の合計を100%にする）
        if normalize:
            row_sums = df_comp.sum(axis=1)
            # ゼロ除算を避ける
            row_sums = row_sums.replace(0, 1.0)
            df_comp = df_comp.div(row_sums, axis=0) * 100.0
        
        return df_comp


class CategoricalEncoder:
    """
    カテゴリ変数のエンコーディングクラス
    
    One-hotエンコーディングまたはラベルエンコーディングを提供します。
    """
    
    def __init__(self, method: str = "onehot", handle_unknown: str = "ignore"):
        """
        初期化
        
        Args:
            method: エンコーディング方法（"onehot" または "label"）
            handle_unknown: 未知のカテゴリの扱い（"ignore" or "error"）
        """
        self.method = method
        self.handle_unknown = handle_unknown
        self.encoder = None
        self.categorical_columns = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, categorical_columns: list) -> 'CategoricalEncoder':
        """
        カテゴリ変数にエンコーダをフィット
        
        Args:
            X: 全特徴量のDataFrame
            categorical_columns: カテゴリ変数のカラムリスト
        
        Returns:
            self
        """
        self.categorical_columns = categorical_columns
        
        if self.method == "onehot":
            self.encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown=self.handle_unknown
            )
            X_cat = X[categorical_columns]
            self.encoder.fit(X_cat)
            
            print(f"[CategoricalEncoder] Fitted one-hot encoder for {len(categorical_columns)} columns")
            print(f"[CategoricalEncoder] Generated {len(self.encoder.get_feature_names_out())} features")
            
        elif self.method == "label":
            # ラベルエンコーディング（将来の拡張用）
            raise NotImplementedError("Label encoding not implemented yet")
        
        self.is_fitted = True
        return self
    
    def transform(
        self,
        X: pd.DataFrame,
        other_columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        カテゴリ変数を変換
        
        Args:
            X: 全特徴量のDataFrame
            other_columns: カテゴリ以外の列（Noneの場合は自動検出）
        
        Returns:
            変換後のDataFrame
        """
        if not self.is_fitted:
            raise ValueError("Encoder is not fitted yet. Call fit() first.")
        
        if self.method == "onehot":
            X_cat = X[self.categorical_columns]
            X_encoded = self.encoder.transform(X_cat)
            
            # エンコード後のカラム名取得
            encoded_columns = self.encoder.get_feature_names_out(self.categorical_columns)
            df_encoded = pd.DataFrame(X_encoded, columns=encoded_columns, index=X.index)
            
            # 他の特徴量と結合
            if other_columns is None:
                other_columns = [col for col in X.columns if col not in self.categorical_columns]
            
            if other_columns:
                df_other = X[other_columns]
                df_result = pd.concat([df_other, df_encoded], axis=1)
            else:
                df_result = df_encoded
            
            return df_result
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        categorical_columns: list,
        other_columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        fitとtransformを連続実行
        
        Args:
            X: 全特徴量のDataFrame
            categorical_columns: カテゴリ変数のカラムリスト
            other_columns: カテゴリ以外の列
        
        Returns:
            変換後のDataFrame
        """
        self.fit(X, categorical_columns)
        return self.transform(X, other_columns)


def apply_feature_engineering(
    X_df: pd.DataFrame,
    composition_columns: Optional[list] = None,
    categorical_columns: Optional[list] = None,
    use_pca: bool = True,
    pca_variance_threshold: float = 0.95,
    pca_n_components: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    特徴量エンジニアリングを一括適用
    
    Args:
        X_df: 元の特徴量DataFrame
        composition_columns: 組成カラムのリスト
        categorical_columns: カテゴリ変数カラムのリスト
        use_pca: PCAを使用するか
        pca_variance_threshold: PCAの累積寄与率閾値
        pca_n_components: PCAの成分数（Noneで自動）
    
    Returns:
        変換後のDataFrame, 変換情報の辞書
    """
    X_transformed = X_df.copy()
    transform_info = {}
    
    # 1. カテゴリ変数のエンコーディング
    if categorical_columns:
        cat_encoder = CategoricalEncoder(method="onehot")
        X_transformed = cat_encoder.fit_transform(X_transformed, categorical_columns)
        transform_info["categorical_encoder"] = cat_encoder
        print(f"[FeatureEngineering] Encoded {len(categorical_columns)} categorical columns")
    
    # 2. 組成のPCA
    if use_pca and composition_columns:
        pca_transformer = CompositionPCA(
            n_components=pca_n_components,
            variance_threshold=pca_variance_threshold
        )
        X_transformed = pca_transformer.fit_transform(X_transformed, composition_columns)
        transform_info["pca"] = pca_transformer
        print(f"[FeatureEngineering] Applied PCA to {len(composition_columns)} composition columns")
    
    print(f"[FeatureEngineering] Final feature dimensions: {X_transformed.shape[1]}")
    
    return X_transformed, transform_info
