"""
データ読み込みと前処理モジュール

CCFatigueデータセットの読み込み、設計変数と劣化指標の定義、
初期データ分割、最近傍探索による仮想実験を提供します。
"""

import torch
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from pathlib import Path

# CCFatigueデータファイルのパス
CCFATIGUE_DATA_DIR = Path(__file__).parent.parent / "data"
CCFATIGUE_SNC_FILE = CCFATIGUE_DATA_DIR / "ccfatigue_snc.csv"
CCFATIGUE_TESTS_FILE = CCFATIGUE_DATA_DIR / "ccfatigue_tests.csv"
# 実データを優先、存在しない場合はモックデータにフォールバック
COMPLEX_ALLOY_FILE = CCFATIGUE_DATA_DIR / "complex_alloy_fatigue_real.csv"
if not COMPLEX_ALLOY_FILE.exists():
    COMPLEX_ALLOY_FILE = CCFATIGUE_DATA_DIR / "complex_alloy_fatigue.csv"


def load_ccfatigue_data(
    dataset_name: str = "snc",
    design_variables: Optional[List[str]] = None,
    objective_variable: str = "cycles_to_failure",
    objective_transform: str = "log10",
    filter_conditions: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    """
    CCFatigueデータセットを読み込み、前処理を行う
    
    Args:
        dataset_name: データセット名（"snc", "tests", "complex_alloy"）
        design_variables: 設計変数のリスト（Noneの場合はデフォルト）
        objective_variable: 目的変数のカラム名
        objective_transform: 目的変数の変換（"log10", "log", "none"）
        filter_conditions: データフィルタ条件の辞書（例: {"temperature": 293, "environment": "air"}）
    
    Returns:
        X_df: 設計変数のDataFrame
        y_df: 目的変数のDataFrame
        design_var_names: 設計変数名のリスト
        objective_var_name: 目的変数名
    """
    if design_variables is None:
        # デフォルト設計変数をデータセットごとに設定
        if dataset_name == "complex_alloy":
            # 複雑合金データセットのデフォルト
            design_variables = ["Fe", "Ni", "Cr", "Co", "Al", "stress_max", "stress_ratio", "frequency"]
        else:
            design_variables = ["stress_max", "stress_ratio"]
    
    # 実データファイルが存在する場合は読み込む
    if dataset_name == "snc" and CCFATIGUE_SNC_FILE.exists():
        print(f"[DataLoader] Loading CCFatigue S-N curve data from: {CCFATIGUE_SNC_FILE}")
        df = pd.read_csv(CCFATIGUE_SNC_FILE)
        print(f"[DataLoader] Dataset shape: {df.shape}")
        print(f"[DataLoader] Available columns: {list(df.columns)}")
        
        # 必要なカラムの存在確認
        missing_cols = set(design_variables + [objective_variable]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # 欠損値の処理
        df_clean = df[design_variables + [objective_variable]].dropna()
        print(f"[DataLoader] After dropping NaN: {df_clean.shape}")
        
        X_df = df_clean[design_variables].astype(float).reset_index(drop=True)
        y_raw = df_clean[objective_variable].astype(float).reset_index(drop=True)
        
        print(f"[DataLoader] ✓ Using real CCFatigue data with {len(df_clean)} samples")
        
    elif dataset_name == "tests" and CCFATIGUE_TESTS_FILE.exists():
        print(f"[DataLoader] Loading CCFatigue test data from: {CCFATIGUE_TESTS_FILE}")
        df = pd.read_csv(CCFATIGUE_TESTS_FILE)
        # testsデータセットの場合、カラムマッピングが必要
        # maximum load -> stress_max, number of cycles -> cycles_to_failure
        df = df.rename(columns={
            'maximum load': 'stress_max',
            'number of cycles': 'cycles_to_failure'
        })
        
        # stress_ratioがない場合は固定値を使用
        if 'stress_ratio' not in df.columns:
            df['stress_ratio'] = 0.1  # デフォルト値
        
        df_clean = df[design_variables + [objective_variable]].dropna()
        X_df = df_clean[design_variables].astype(float).reset_index(drop=True)
        y_raw = df_clean[objective_variable].astype(float).reset_index(drop=True)
        
        print(f"[DataLoader] ✓ Using real CCFatigue test data with {len(df_clean)} samples")
    
    elif dataset_name == "complex_alloy" and COMPLEX_ALLOY_FILE.exists():
        print(f"[DataLoader] Loading Complex Alloy fatigue data from: {COMPLEX_ALLOY_FILE}")
        df = pd.read_csv(COMPLEX_ALLOY_FILE)
        print(f"[DataLoader] Dataset shape: {df.shape}")
        print(f"[DataLoader] Available columns: {list(df.columns)}")
        
        # フィルタ条件を適用（MVP v0.1.2: 室温・空気中・一定振幅のみ）
        if filter_conditions is None:
            filter_conditions = {
                "temperature": (20, 30),     # 室温範囲 (°C) - 実データに合わせて調整
                "environment": "air",        # 空気中 (小文字)
                "loading_mode": ["uniaxial", "tension-compression"]  # 軸荷重
            }
        
        print(f"[DataLoader] Applying filter conditions: {filter_conditions}")
        df_filtered = df.copy()
        
        # 温度フィルタ
        if "temperature" in filter_conditions:
            temp_range = filter_conditions["temperature"]
            if isinstance(temp_range, tuple):
                df_filtered = df_filtered[
                    (df_filtered["temperature"] >= temp_range[0]) &
                    (df_filtered["temperature"] <= temp_range[1])
                ]
                print(f"[DataLoader] Filtered by temperature: {len(df_filtered)} samples remaining")
        
        # 環境フィルタ (大文字小文字を無視)
        if "environment" in filter_conditions:
            env = filter_conditions["environment"]
            df_filtered = df_filtered[df_filtered["environment"].str.lower() == env.lower()]
            print(f"[DataLoader] Filtered by environment={env}: {len(df_filtered)} samples remaining")
        
        # 荷重モードフィルタ (複数モードに対応)
        if "loading_mode" in filter_conditions:
            mode = filter_conditions["loading_mode"]
            if isinstance(mode, list):
                df_filtered = df_filtered[df_filtered["loading_mode"].isin(mode)]
                print(f"[DataLoader] Filtered by loading_mode in {mode}: {len(df_filtered)} samples remaining")
            else:
                df_filtered = df_filtered[df_filtered["loading_mode"] == mode]
                print(f"[DataLoader] Filtered by loading_mode={mode}: {len(df_filtered)} samples remaining")
        
        # 必要なカラムの存在確認
        missing_cols = set(design_variables + [objective_variable]) - set(df_filtered.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        
        # 欠損値の処理
        df_clean = df_filtered[design_variables + [objective_variable]].dropna()
        print(f"[DataLoader] After dropping NaN: {df_clean.shape}")
        
        # データ型変換
        X_df = df_clean[design_variables].astype(float).reset_index(drop=True)
        y_raw = df_clean[objective_variable].astype(float).reset_index(drop=True)
        
        print(f"[DataLoader] ✓ Using Complex Alloy data with {len(df_clean)} samples")
        print(f"[DataLoader] ✓ Composition columns: {[col for col in design_variables if col in ['Fe', 'Ni', 'Cr', 'Co', 'Al', 'Ti', 'Mn', 'Cu', 'Zr']]}")
        print(f"[DataLoader] ✓ Test condition columns: {[col for col in design_variables if col in ['stress_max', 'stress_ratio', 'frequency']]}")
        
    else:
        # 実データファイルが見つからない場合、モックデータを生成
        print(f"[DataLoader] ⚠ CCFatigue data file not found: {CCFATIGUE_SNC_FILE}")
        print("[DataLoader] Generating mock data for demonstration")
        np.random.seed(42)
        n_samples = 200
        
        # 仮想的な疲労データを生成（SNCデータに類似）
        stress_max = np.random.uniform(20, 35, n_samples)
        stress_ratio = np.random.uniform(0.05, 0.15, n_samples)
        
        # 寿命を簡単なモデルで生成（応力が高いほど寿命が短い）
        # S-N曲線: log(N) = A - B*log(S)
        cycles = 10 ** (12 - 0.3 * stress_max + 2 * stress_ratio)
        cycles += np.random.lognormal(0, 0.3, n_samples) * cycles * 0.1  # ノイズ追加
        cycles = np.maximum(cycles, 1)  # 最小値の制限
        
        X_df = pd.DataFrame({
            "stress_max": stress_max,
            "stress_ratio": stress_ratio
        })
        y_raw = pd.Series(cycles, name=objective_variable)
        
        print(f"[DataLoader] Generated mock data: {X_df.shape}")
    
    # 目的変数の変換
    if objective_transform == "log10":
        y_transformed = np.log10(y_raw)
        print(f"[DataLoader] Applied log10 transform to objective variable")
    elif objective_transform == "log":
        y_transformed = np.log(y_raw)
        print(f"[DataLoader] Applied log transform to objective variable")
    else:
        y_transformed = y_raw
    
    y_df = pd.Series(y_transformed, name=f"{objective_variable}_{objective_transform}")
    
    print(f"[DataLoader] X range:")
    for col in design_variables:
        print(f"  {col}: [{X_df[col].min():.2f}, {X_df[col].max():.2f}]")
    print(f"[DataLoader] y range: [{y_df.min():.2f}, {y_df.max():.2f}]")
    
    return X_df, y_df, design_variables, objective_variable


def prepare_initial_data(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    M: int = 10,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    初期データとプールデータに分割
    
    Args:
        X_df: 設計変数のDataFrame
        y_df: 目的変数のDataFrame
        M: 初期データ点数
        seed: 乱数シード
    
    Returns:
        train_X: 初期データの設計変数（M × d）
        train_Y: 初期データの目的変数（M × 1）
        pool_X: プールデータの設計変数
        pool_Y: プールデータの目的変数
    """
    # データ点数の確認
    n_total = len(X_df)
    if M >= n_total:
        raise ValueError(f"M ({M}) must be less than total data size ({n_total})")
    
    # ランダム分割
    train_X_df, pool_X_df, train_y_df, pool_y_df = train_test_split(
        X_df, y_df, train_size=M, random_state=seed
    )
    
    # PyTorchテンソルに変換
    train_X = torch.tensor(train_X_df.values, dtype=torch.double)
    train_Y = torch.tensor(train_y_df.values, dtype=torch.double).reshape(-1, 1)
    pool_X = torch.tensor(pool_X_df.values, dtype=torch.double)
    pool_Y = torch.tensor(pool_y_df.values, dtype=torch.double).reshape(-1, 1)
    
    print(f"[DataLoader] Prepared initial data: train={train_X.shape}, pool={pool_X.shape}")
    
    return train_X, train_Y, pool_X, pool_Y


class DatabaseEvaluator:
    """
    最近傍探索による仮想実験評価クラス
    
    BoTorchが提案した連続値の候補点に対して、
    データベース内の最も近い実験結果を返すことで実験をシミュレートします。
    """
    
    def __init__(
        self,
        X_all: torch.Tensor,
        y_all: torch.Tensor,
        device: torch.device
    ):
        """
        初期化
        
        Args:
            X_all: 全データの設計変数
            y_all: 全データの目的変数
            device: PyTorchデバイス
        """
        self.X_all = X_all.to(device)
        self.y_all = y_all.to(device)
        self.device = device
        
        # 最近傍探索器の準備（CPU上で）
        self.nn = NearestNeighbors(n_neighbors=1)
        self.nn.fit(X_all.cpu().numpy())
        
        print(f"[DatabaseEvaluator] Initialized with {len(X_all)} data points on {device}")
    
    def evaluate(self, x_candidate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        候補点を評価（最近傍のデータ点を返す）
        
        Args:
            x_candidate: 候補点（d,）または（1, d）
        
        Returns:
            x_real: 最も近い実データ点（d,）
            y_real: 対応する目的変数値（1,）
        """
        # 形状の調整
        if x_candidate.dim() == 1:
            x_candidate = x_candidate.unsqueeze(0)
        
        # CPU上で最近傍探索
        x_np = x_candidate.detach().cpu().numpy()
        dist, idx = self.nn.kneighbors(x_np)
        idx = idx[0, 0]
        
        # 該当するデータ点を取得
        x_real = self.X_all[idx]
        y_real = self.y_all[idx]
        
        return x_real, y_real
    
    def batch_evaluate(
        self, 
        x_candidates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        複数の候補点を一度に評価
        
        Args:
            x_candidates: 候補点（n × d）
        
        Returns:
            x_reals: 最も近い実データ点（n × d）
            y_reals: 対応する目的変数値（n × 1）
        """
        x_np = x_candidates.detach().cpu().numpy()
        dist, indices = self.nn.kneighbors(x_np)
        indices = indices[:, 0]
        
        x_reals = self.X_all[indices]
        y_reals = self.y_all[indices]
        
        return x_reals, y_reals


def get_search_bounds(X_df: pd.DataFrame) -> torch.Tensor:
    """
    探索空間の境界を取得
    
    Args:
        X_df: 設計変数のDataFrame
    
    Returns:
        bounds: 境界テンソル（2 × d）[下限, 上限]
    """
    mins = torch.tensor(X_df.min().values, dtype=torch.double)
    maxs = torch.tensor(X_df.max().values, dtype=torch.double)
    bounds = torch.stack([mins, maxs])
    
    return bounds


def describe_data(X_df: pd.DataFrame, y_df: pd.DataFrame) -> Dict[str, Any]:
    """
    データの統計情報を返す
    
    Args:
        X_df: 設計変数のDataFrame
        y_df: 目的変数のDataFrame
    
    Returns:
        統計情報の辞書
    """
    info = {
        "n_samples": len(X_df),
        "n_features": len(X_df.columns),
        "feature_names": list(X_df.columns),
        "X_stats": X_df.describe().to_dict(),
        "y_stats": y_df.describe().to_dict(),
    }
    
    return info
