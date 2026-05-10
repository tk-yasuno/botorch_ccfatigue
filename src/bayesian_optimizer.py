"""
ベイズ最適化エンジンモジュール

BoTorchを用いた適応的実験計画を実装します。
SingleTaskGP、Expected Improvement、optimize_acqfを活用し、
反復的に最良条件を探索します。
"""

import torch
import time
from typing import Dict, List, Any, Optional, Tuple
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .compute_config import ComputeConfig
from .data_loader import DatabaseEvaluator


class BayesianOptimizer:
    """
    ベイズ最適化クラス
    
    t: iteration（反復回数、t=1から開始）
    ym: deterioration index（劣化指標 = cycles_to_failure）
    Bayes Optimize fm(x,t)（適応的実験計画による最適化）
    """
    
    def __init__(
        self,
        bounds: torch.Tensor,
        evaluator: DatabaseEvaluator,
        compute_config: ComputeConfig,
        acq_function: str = "EI"
    ):
        """
        初期化
        
        Args:
            bounds: 探索空間の境界（2 × d）
            evaluator: 仮想実験評価器
            compute_config: 計算環境設定
            acq_function: 獲得関数（"EI" or "LogEI"）
        """
        self.bounds = bounds.to(compute_config.device)
        self.evaluator = evaluator
        self.compute_config = compute_config
        self.device = compute_config.device
        self.acq_function_name = acq_function
        
        # 獲得関数最適化パラメータ
        self.acqf_params = compute_config.get_acqf_params()
        
        # 最適化履歴
        self.history = {
            "iteration": [],
            "best_y": [],
            "current_y": [],
            "execution_time": [],
            "gpu_memory_gb": []
        }
        
        print(f"[BayesianOptimizer] Initialized on device: {self.device}")
        print(f"[BayesianOptimizer] Acquisition function: {acq_function}")
        print(f"[BayesianOptimizer] Acqf params: {self.acqf_params}")
    
    def fit_model(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor
    ) -> SingleTaskGP:
        """
        GPモデルを学習
        
        Args:
            train_X: 訓練データの設計変数（n × d）
            train_Y: 訓練データの目的変数（n × 1）
        
        Returns:
            学習済みGPモデル
        """
        # データをdeviceに移動
        train_X = train_X.to(self.device)
        train_Y = train_Y.to(self.device)
        
        # モデルの構築（入力正規化、出力標準化）
        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(d=train_X.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        
        # モデルをdeviceに移動
        model = model.to(self.device)
        
        # 学習
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        return model
    
    def propose_next_point(
        self,
        model: SingleTaskGP,
        train_Y: torch.Tensor
    ) -> torch.Tensor:
        """
        次の候補点を提案
        
        Args:
            model: 学習済みGPモデル
            train_Y: 現在の訓練データの目的変数
        
        Returns:
            候補点（d,）
        """
        # 現在の最良値
        best_f = train_Y.max()
        
        # 獲得関数の構築
        if self.acq_function_name == "LogEI":
            acq_func = LogExpectedImprovement(model=model, best_f=best_f)
        else:  # "EI"
            acq_func = ExpectedImprovement(model=model, best_f=best_f)
        
        # 獲得関数の最適化
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=self.acqf_params["num_restarts"],
            raw_samples=self.acqf_params["raw_samples"],
        )
        
        # (1, d) → (d,)
        candidate = candidate.squeeze(0)
        
        return candidate
    
    def optimize_loop(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        T: int,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        適応的実験計画のメインループ
        
        t: iteration, start t=1
        ym: deterioration index
        Bayes Optimize fm(x,t)
        Bayes update → t+1
        
        Args:
            train_X: 初期データの設計変数（M × d）
            train_Y: 初期データの目的変数（M × 1）
            T: 反復回数
            verbose: 詳細ログの表示
        
        Returns:
            最適化結果の辞書
        """
        # データをdeviceに移動
        train_X = train_X.to(self.device)
        train_Y = train_Y.to(self.device)
        
        M = len(train_X)
        
        if verbose:
            print("\n" + "=" * 70)
            print("Starting Bayesian Optimization Loop")
            print("=" * 70)
            print(f"Initial data size: M = {M}")
            print(f"Number of iterations: T = {T}")
            print(f"Initial best y: {train_Y.max().item():.4f}")
            print("=" * 70 + "\n")
        
        # 初期状態を履歴に記録
        self.history["iteration"].append(0)
        self.history["best_y"].append(train_Y.max().item())
        self.history["current_y"].append(train_Y.max().item())
        self.history["execution_time"].append(0.0)
        
        mem_info = self.compute_config.monitor_gpu_memory()
        self.history["gpu_memory_gb"].append(mem_info[0] if mem_info else 0.0)
        
        # ベイズ最適化ループ（t = 1, 2, ..., T）
        for t in range(1, T + 1):
            iter_start_time = time.time()
            
            if verbose:
                print(f"\n--- Iteration t = {M + t} ---")
            
            # 1) モデルをフィット
            model = self.fit_model(train_X, train_Y)
            
            # 2) 次の候補点を提案
            x_cand = self.propose_next_point(model, train_Y)
            
            if verbose:
                print(f"Proposed x: {x_cand.cpu().numpy()}")
            
            # 3) データベースで「実験」を行う（最近傍）
            x_real, y_real = self.evaluator.evaluate(x_cand)
            
            if verbose:
                print(f"Nearest real x: {x_real.cpu().numpy()}")
                print(f"Observed y: {y_real.item():.4f}")
            
            # 4) データに追加してベイズ更新
            train_X = torch.cat([train_X, x_real.unsqueeze(0)], dim=0)
            train_Y = torch.cat([train_Y, y_real.unsqueeze(0)], dim=0)
            
            # 現在の最良値
            current_best = train_Y.max().item()
            
            # 実行時間とメモリの記録
            iter_time = time.time() - iter_start_time
            mem_info = self.compute_config.monitor_gpu_memory()
            
            self.history["iteration"].append(t)
            self.history["best_y"].append(current_best)
            self.history["current_y"].append(y_real.item())
            self.history["execution_time"].append(iter_time)
            self.history["gpu_memory_gb"].append(mem_info[0] if mem_info else 0.0)
            
            if verbose:
                print(f"Current best y: {current_best:.4f}")
                print(f"Iteration time: {iter_time:.2f}s")
                if mem_info:
                    print(f"GPU memory: {mem_info[0]:.2f} GB / {mem_info[1]:.2f} GB")
        
        # 最終結果の取得
        best_idx = torch.argmax(train_Y)
        best_x = train_X[best_idx]
        best_y = train_Y[best_idx]
        
        results = {
            "best_x": best_x.cpu(),
            "best_y": best_y.cpu().item(),
            "train_X": train_X.cpu(),
            "train_Y": train_Y.cpu(),
            "history": self.history,
            "final_model": model,
        }
        
        if verbose:
            print("\n" + "=" * 70)
            print("Optimization Complete")
            print("=" * 70)
            print(f"Best y found: {best_y.item():.4f}")
            print(f"Best x: {best_x.cpu().numpy()}")
            print(f"Total iterations: {T}")
            print(f"Total time: {sum(self.history['execution_time']):.2f}s")
            print("=" * 70 + "\n")
        
        return results
    
    def get_results(self) -> Dict[str, Any]:
        """
        最適化履歴を取得
        
        Returns:
            履歴データの辞書
        """
        return self.history
    
    def predict(
        self,
        model: SingleTaskGP,
        X_test: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPモデルによる予測
        
        Args:
            model: 学習済みGPモデル
            X_test: テストデータ（n × d）
        
        Returns:
            mean: 予測平均（n,）
            std: 予測標準偏差（n,）
        """
        X_test = X_test.to(self.device)
        model.eval()
        
        with torch.no_grad():
            posterior = model.posterior(X_test)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance)
        
        return mean.cpu(), std.cpu()
