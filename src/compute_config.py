"""
計算環境管理モジュール

3つの計算モード（no_parallel, cpu_16cores, gpu_16gb）を管理し、
PyTorchのデバイス設定、スレッド数制御、GPUメモリ管理を行います。
"""

import os
import yaml
import torch
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class ComputeConfig:
    """
    計算環境設定管理クラス
    
    config.yamlからの設定読み込み、デバイス自動検出、
    torch設定の適用、GPUメモリ管理を行います。
    """
    
    VALID_MODES = ["auto", "no_parallel", "cpu_16cores", "gpu_16gb"]
    
    def __init__(self, config_path: Optional[str] = None, mode: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: config.yamlファイルのパス（Noneの場合はデフォルト）
            mode: 計算モードの上書き（Noneの場合はconfig.yamlの設定を使用）
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        
        # モードの決定（引数 > config.yaml）
        if mode:
            if mode not in self.VALID_MODES:
                raise ValueError(f"Invalid mode: {mode}. Must be one of {self.VALID_MODES}")
            self.mode = mode
        else:
            self.mode = self.config.get("compute_mode", "auto")
        
        # autoモードの場合、GPU利用可能性で自動決定
        if self.mode == "auto":
            self.mode = "gpu_16gb" if torch.cuda.is_available() else "cpu_16cores"
            print(f"[ComputeConfig] Auto mode: Selected '{self.mode}' (GPU available: {torch.cuda.is_available()})")
        
        # モード設定の取得
        self.mode_config = self.config.get(self.mode, {})
        
        # デバイスの設定
        self.device = self._setup_device()
        
        # PyTorch設定の適用
        self._apply_torch_settings()
        
        print(f"[ComputeConfig] Compute mode: {self.mode}")
        print(f"[ComputeConfig] Device: {self.device}")
    
    def _load_config(self) -> Dict[str, Any]:
        """config.yamlを読み込み"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            warnings.warn(f"Config file not found: {self.config_path}. Using defaults.")
            return self._get_default_config()
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "compute_mode": "auto",
            "no_parallel": {"device": "cpu", "num_threads": 1, "batch_size": 1},
            "cpu_16cores": {"device": "cpu", "num_threads": 16, "batch_size": 1},
            "gpu_16gb": {
                "device": "cuda",
                "max_memory_gb": 16,
                "batch_size": "auto",
                "num_restarts": 10,
                "raw_samples": 128
            },
            "optimization": {
                "M": 10,
                "T": 20,
                "seed": 42,
                "acq_function": "EI"
            }
        }
    
    def _setup_device(self) -> torch.device:
        """デバイスの設定"""
        device_name = self.mode_config.get("device", "cpu")
        
        # GPU指定だが利用不可の場合、CPUにフォールバック
        if device_name == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                f"GPU mode '{self.mode}' specified but CUDA is not available. "
                f"Falling back to CPU."
            )
            device_name = "cpu"
            self.mode_config["device"] = "cpu"
        
        device = torch.device(device_name)
        
        return device
    
    def _apply_torch_settings(self):
        """PyTorchの設定を適用"""
        # スレッド数の設定
        num_threads = self.mode_config.get("num_threads")
        if num_threads:
            torch.set_num_threads(num_threads)
            print(f"[ComputeConfig] Set PyTorch threads: {num_threads}")
        
        # double精度をデフォルトに
        torch.set_default_dtype(torch.double)
        
        # 再現性のための設定（オプション）
        seed = self.config.get("optimization", {}).get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def get_acqf_params(self) -> Dict[str, int]:
        """
        獲得関数最適化のパラメータを取得
        
        Returns:
            num_restarts, raw_samplesを含む辞書
        """
        if self.mode == "gpu_16gb":
            num_restarts = self.mode_config.get("num_restarts", 10)
            raw_samples = self.mode_config.get("raw_samples", 128)
        else:
            # CPUモードではより控えめな値
            num_restarts = 5
            raw_samples = 64
        
        return {
            "num_restarts": num_restarts,
            "raw_samples": raw_samples
        }
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """最適化パラメータを取得"""
        return self.config.get("optimization", {
            "M": 10,
            "T": 20,
            "seed": 42,
            "acq_function": "EI"
        })
    
    def monitor_gpu_memory(self) -> Optional[Tuple[float, float]]:
        """
        GPUメモリ使用量を監視
        
        Returns:
            (使用中メモリGB, 総メモリGB) または None（GPU非使用時）
        """
        if self.device.type != "cuda":
            return None
        
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated() / 1e9  # GB
        reserved = torch.cuda.memory_reserved() / 1e9    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return (reserved, total)
    
    def log_config(self) -> str:
        """
        設定情報を文字列として返す
        
        Returns:
            設定情報の文字列
        """
        lines = [
            "=" * 70,
            "Compute Configuration",
            "=" * 70,
            f"Mode: {self.mode}",
            f"Device: {self.device}",
        ]
        
        if self.device.type == "cuda":
            mem_info = self.monitor_gpu_memory()
            if mem_info:
                lines.append(f"GPU Memory: {mem_info[0]:.2f} GB / {mem_info[1]:.2f} GB")
        
        num_threads = self.mode_config.get("num_threads")
        if num_threads:
            lines.append(f"CPU Threads: {num_threads}")
        
        acqf_params = self.get_acqf_params()
        lines.append(f"Acquisition Function Params: {acqf_params}")
        
        opt_params = self.get_optimization_params()
        lines.append(f"Optimization Params: M={opt_params['M']}, T={opt_params['T']}, "
                    f"seed={opt_params['seed']}, acq_func={opt_params['acq_function']}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ComputeConfig(mode='{self.mode}', device={self.device})"
