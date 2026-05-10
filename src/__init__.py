"""
Material Usability Maximizer MVP

ベイズ最適化（BoTorch + GP）による材料寿命最大化の適応的実験計画システム
"""

__version__ = "0.1.1"
__author__ = "Takato Yasuno"

from .compute_config import ComputeConfig
from .data_loader import load_ccfatigue_data, prepare_initial_data, DatabaseEvaluator
from .bayesian_optimizer import BayesianOptimizer
from .visualization import (
    plot_convergence,
    plot_explored_space,
    plot_gp_predictions,
    plot_acquisition_function,
    plot_compute_performance,
    save_results_summary,
)

__all__ = [
    "ComputeConfig",
    "load_ccfatigue_data",
    "prepare_initial_data",
    "DatabaseEvaluator",
    "BayesianOptimizer",
    "plot_convergence",
    "plot_explored_space",
    "plot_gp_predictions",
    "plot_acquisition_function",
    "plot_compute_performance",
    "save_results_summary",
]
