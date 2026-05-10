"""
可視化モジュール

最適化プロセスと結果を可視化する関数群を提供します。
収束曲線、探索空間、GP予測分布、獲得関数、計算パフォーマンスを可視化します。
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from itertools import combinations

# スタイル設定
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_convergence(
    history: Dict[str, List],
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    収束曲線をプロット
    
    Args:
        history: 最適化履歴
        output_path: 保存先パス（Noneの場合は保存しない）
        dpi: 解像度
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = history["iteration"]
    best_y = history["best_y"]
    current_y = history["current_y"]
    
    # 最良値の推移
    ax.plot(iterations, best_y, 'o-', linewidth=2, markersize=6,
            label='Best y (cumulative)', color='#2E86AB')
    
    # 各反復での観測値
    ax.scatter(iterations[1:], current_y[1:], alpha=0.5, s=50,
              label='Observed y (each iteration)', color='#A23B72')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value (log10 cycles)', fontsize=12)
    ax.set_title('Bayesian Optimization Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualization] Saved convergence plot to {output_path}")
    
    return fig


def plot_explored_space(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    feature_names: List[str],
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    探索された条件空間を可視化（2D散布図の組み合わせ）
    
    Args:
        train_X: 訓練データの設計変数（n × d）
        train_Y: 訓練データの目的変数（n × 1）
        feature_names: 設計変数名のリスト
        output_path: 保存先パス
        dpi: 解像度
    
    Returns:
        matplotlib Figure
    """
    X_np = train_X.cpu().numpy()
    y_np = train_Y.cpu().numpy().flatten()
    
    d = X_np.shape[1]
    n_combinations = len(list(combinations(range(d), 2)))
    
    # サブプロットのグリッドサイズを計算
    n_cols = min(3, n_combinations)
    n_rows = (n_combinations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_combinations == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 各2D組み合わせをプロット
    for idx, (i, j) in enumerate(combinations(range(d), 2)):
        ax = axes[idx]
        
        scatter = ax.scatter(
            X_np[:, i], X_np[:, j], c=y_np,
            cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidth=0.5
        )
        
        ax.set_xlabel(feature_names[i], fontsize=11)
        ax.set_ylabel(feature_names[j], fontsize=11)
        ax.set_title(f'{feature_names[i]} vs {feature_names[j]}', fontsize=12)
        
        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log10(cycles)', fontsize=10)
    
    # 未使用のサブプロットを非表示
    for idx in range(n_combinations, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Explored Design Space', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualization] Saved explored space plot to {output_path}")
    
    return fig


def plot_gp_predictions(
    model,
    bounds: torch.Tensor,
    feature_names: List[str],
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    device: torch.device,
    output_path: Optional[str] = None,
    dpi: int = 300,
    n_points: int = 100
) -> plt.Figure:
    """
    GP予測分布を1Dスライスで可視化
    
    Args:
        model: 学習済みGPモデル
        bounds: 探索空間の境界（2 × d）
        feature_names: 設計変数名のリスト
        train_X: 訓練データの設計変数
        train_Y: 訓練データの目的変数
        device: PyTorchデバイス
        output_path: 保存先パス
        dpi: 解像度
        n_points: 各軸のサンプル点数
    
    Returns:
        matplotlib Figure
    """
    d = bounds.shape[1]
    
    fig, axes = plt.subplots(1, d, figsize=(6 * d, 5))
    if d == 1:
        axes = [axes]
    
    # 各設計変数に対して1Dスライス
    for dim_idx in range(d):
        ax = axes[dim_idx]
        
        # 他の次元は中央値に固定
        X_test_base = train_X.median(dim=0).values.clone()
        
        # 対象次元を変化させる
        x_range = torch.linspace(
            bounds[0, dim_idx],
            bounds[1, dim_idx],
            n_points
        ).to(device)
        
        X_test = X_test_base.unsqueeze(0).repeat(n_points, 1)
        X_test[:, dim_idx] = x_range
        
        # 予測
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_test)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            std = torch.sqrt(posterior.variance).squeeze(-1).cpu().numpy()
        
        x_plot = x_range.cpu().numpy()
        
        # プロット
        ax.plot(x_plot, mean, 'b-', linewidth=2, label='GP mean')
        ax.fill_between(
            x_plot, mean - 2 * std, mean + 2 * std,
            alpha=0.3, color='blue', label='GP ±2σ'
        )
        
        # 訓練データ点
        train_x_dim = train_X[:, dim_idx].cpu().numpy()
        train_y_vals = train_Y.cpu().numpy().flatten()
        ax.scatter(train_x_dim, train_y_vals, c='red', s=30,
                  alpha=0.6, zorder=10, label='Observed data')
        
        ax.set_xlabel(feature_names[dim_idx], fontsize=11)
        ax.set_ylabel('Objective (log10 cycles)', fontsize=11)
        ax.set_title(f'GP Prediction: {feature_names[dim_idx]}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Gaussian Process Predictions (1D Slices)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualization] Saved GP predictions plot to {output_path}")
    
    return fig


def plot_acquisition_function(
    acq_func,
    bounds: torch.Tensor,
    feature_names: List[str],
    train_X: torch.Tensor,
    device: torch.device,
    output_path: Optional[str] = None,
    dpi: int = 300,
    n_points: int = 100
) -> plt.Figure:
    """
    獲得関数を1Dスライスで可視化
    
    Args:
        acq_func: 獲得関数
        bounds: 探索空間の境界
        feature_names: 設計変数名のリスト
        train_X: 訓練データの設計変数
        device: PyTorchデバイス
        output_path: 保存先パス
        dpi: 解像度
        n_points: 各軸のサンプル点数
    
    Returns:
        matplotlib Figure
    """
    d = bounds.shape[1]
    
    fig, axes = plt.subplots(1, d, figsize=(6 * d, 5))
    if d == 1:
        axes = [axes]
    
    for dim_idx in range(d):
        ax = axes[dim_idx]
        
        # 他の次元は中央値に固定
        X_test_base = train_X.median(dim=0).values.clone()
        
        # 対象次元を変化させる
        x_range = torch.linspace(
            bounds[0, dim_idx],
            bounds[1, dim_idx],
            n_points
        ).to(device)
        
        X_test = X_test_base.unsqueeze(0).repeat(n_points, 1)
        X_test[:, dim_idx] = x_range
        
        # 獲得関数値の計算
        with torch.no_grad():
            acq_values = acq_func(X_test.unsqueeze(1)).cpu().numpy()
        
        x_plot = x_range.cpu().numpy()
        
        # プロット
        ax.plot(x_plot, acq_values, 'g-', linewidth=2)
        ax.fill_between(x_plot, 0, acq_values, alpha=0.3, color='green')
        
        ax.set_xlabel(feature_names[dim_idx], fontsize=11)
        ax.set_ylabel('Acquisition Value', fontsize=11)
        ax.set_title(f'Acquisition Function: {feature_names[dim_idx]}', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Acquisition Function (1D Slices)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualization] Saved acquisition function plot to {output_path}")
    
    return fig


def plot_compute_performance(
    history: Dict[str, List],
    compute_mode: str,
    output_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    計算パフォーマンスを可視化（実行時間、GPUメモリ）
    
    Args:
        history: 最適化履歴
        compute_mode: 計算モード名
        output_path: 保存先パス
        dpi: 解像度
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = history["iteration"][1:]  # 初期状態を除く
    exec_times = history["execution_time"][1:]
    gpu_memory = history["gpu_memory_gb"][1:]
    
    # 実行時間
    ax1 = axes[0]
    ax1.bar(iterations, exec_times, color='#3A86FF', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Execution Time (s)', fontsize=12)
    ax1.set_title(f'Iteration Time ({compute_mode})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 平均実行時間の表示
    mean_time = np.mean(exec_times)
    ax1.axhline(mean_time, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_time:.2f}s')
    ax1.legend(fontsize=10)
    
    # GPUメモリ使用量
    ax2 = axes[1]
    if any(gpu_memory):
        ax2.plot(iterations, gpu_memory, 'o-', linewidth=2, markersize=6,
                color='#FB5607', label='GPU Memory Used')
        ax2.set_ylabel('GPU Memory (GB)', fontsize=12)
        ax2.set_title('GPU Memory Usage', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'GPU not used', ha='center', va='center',
                fontsize=14, transform=ax2.transAxes)
        ax2.set_title('GPU Memory Usage', fontsize=13, fontweight='bold')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Compute Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualization] Saved compute performance plot to {output_path}")
    
    return fig


def save_results_summary(
    results: Dict[str, Any],
    feature_names: List[str],
    output_dir: str,
    compute_mode: str
):
    """
    最適化結果のサマリーをCSVとJSONで保存
    
    Args:
        results: 最適化結果
        feature_names: 設計変数名のリスト
        output_dir: 出力ディレクトリ
        compute_mode: 計算モード名
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 最適化履歴をCSVで保存
    history = results["history"]
    df_history = pd.DataFrame({
        "iteration": history["iteration"],
        "best_y": history["best_y"],
        "current_y": history["current_y"],
        "execution_time_s": history["execution_time"],
        "gpu_memory_gb": history["gpu_memory_gb"]
    })
    
    csv_path = output_path / "optimization_history.csv"
    df_history.to_csv(csv_path, index=False)
    print(f"[Visualization] Saved optimization history to {csv_path}")
    
    # 2. 最良条件をJSONで保存
    best_x = results["best_x"].numpy()
    best_conditions = {
        "compute_mode": compute_mode,
        "best_objective_value": float(results["best_y"]),
        "best_conditions": {
            feature_names[i]: float(best_x[i])
            for i in range(len(feature_names))
        },
        "total_iterations": len(history["iteration"]) - 1,
        "total_time_s": sum(history["execution_time"]),
        "mean_iteration_time_s": np.mean(history["execution_time"][1:])
    }
    
    import json
    json_path = output_path / "best_conditions.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_conditions, f, indent=2, ensure_ascii=False)
    print(f"[Visualization] Saved best conditions to {json_path}")
    
    # 3. サマリーテキスト
    summary_text = f"""
Material Usability Maximizer - Optimization Summary
{"=" * 60}

Compute Mode: {compute_mode}
Total Iterations: {best_conditions['total_iterations']}
Total Time: {best_conditions['total_time_s']:.2f} seconds
Mean Iteration Time: {best_conditions['mean_iteration_time_s']:.2f} seconds

Best Objective Value (log10 cycles): {best_conditions['best_objective_value']:.4f}
Estimated Cycles to Failure: {10**best_conditions['best_objective_value']:.2e}

Best Conditions:
"""
    for name, value in best_conditions['best_conditions'].items():
        summary_text += f"  {name}: {value:.4f}\n"
    
    summary_text += f"\n{'=' * 60}\n"
    
    txt_path = output_path / "summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"[Visualization] Saved summary to {txt_path}")
    
    print(summary_text)
