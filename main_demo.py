"""
Material Usability Maximizer MVP - メイン実行スクリプト

ベイズ最適化（BoTorch + GP）による材料寿命最大化の適応的実験計画システム

使用例:
    python main_demo.py
    python main_demo.py --compute-mode cpu_16cores
    python main_demo.py --M 15 --T 30 --seed 12345
"""

import argparse
import sys
from pathlib import Path
import warnings

# srcモジュールをインポート
from src.compute_config import ComputeConfig
from src.data_loader import (
    load_ccfatigue_data,
    prepare_initial_data,
    DatabaseEvaluator,
    get_search_bounds,
    describe_data
)
from src.bayesian_optimizer import BayesianOptimizer
from src.visualization import (
    plot_convergence,
    plot_explored_space,
    plot_gp_predictions,
    plot_compute_performance,
    save_results_summary
)


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="Material Usability Maximizer MVP - Bayesian Optimization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_demo.py
  python main_demo.py --compute-mode cpu_16cores
  python main_demo.py --M 15 --T 30 --seed 12345
  python main_demo.py --compute-mode gpu_16gb --output results_gpu
        """
    )
    
    # 計算環境設定
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '--compute-mode',
        type=str,
        choices=['auto', 'no_parallel', 'cpu_16cores', 'gpu_16gb'],
        help='Compute mode (overrides config.yaml)'
    )
    
    # 最適化パラメータ
    parser.add_argument(
        '--M',
        type=int,
        help='Initial sample size (overrides config.yaml)'
    )
    parser.add_argument(
        '--T',
        type=int,
        help='Number of BO iterations (overrides config.yaml)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed (overrides config.yaml)'
    )
    parser.add_argument(
        '--acq-function',
        type=str,
        choices=['EI', 'LogEI'],
        help='Acquisition function (overrides config.yaml)'
    )
    
    # 出力設定
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output (default: True)'
    )
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    print("\n" + "=" * 70)
    print("Material Usability Maximizer MVP")
    print("Bayesian Optimization for Material Lifetime Maximization")
    print("=" * 70 + "\n")
    
    # 引数の解析
    args = parse_arguments()
    
    # 1. 計算環境の設定
    print("[Step 1/6] Setting up compute configuration...")
    compute_config = ComputeConfig(
        config_path=args.config,
        mode=args.compute_mode
    )
    print(compute_config.log_config())
    
    # config.yamlから最適化パラメータを取得（コマンドライン引数で上書き可能）
    opt_params = compute_config.get_optimization_params()
    M = args.M if args.M is not None else opt_params.get('M', 10)
    T = args.T if args.T is not None else opt_params.get('T', 20)
    seed = args.seed if args.seed is not None else opt_params.get('seed', 42)
    acq_func = args.acq_function if args.acq_function else opt_params.get('acq_function', 'EI')
    
    print(f"\nOptimization parameters:")
    print(f"  Initial samples (M): {M}")
    print(f"  BO iterations (T): {T}")
    print(f"  Random seed: {seed}")
    print(f"  Acquisition function: {acq_func}")
    
    # 2. データの読み込み
    print(f"\n[Step 2/6] Loading CCFatigue data...")
    try:
        X_df, y_df, feature_names, objective_name = load_ccfatigue_data(
            dataset_name=opt_params.get('dataset_name', 'tension_compression'),
            design_variables=opt_params.get('design_variables'),
            objective_variable=opt_params.get('objective_variable', 'cycles_to_failure'),
            objective_transform=opt_params.get('objective_transform', 'log10')
        )
        
        # データの統計情報
        data_info = describe_data(X_df, y_df)
        print(f"  Total samples: {data_info['n_samples']}")
        print(f"  Features: {data_info['feature_names']}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # 3. 初期データの準備
    print(f"\n[Step 3/6] Preparing initial data...")
    train_X, train_Y, pool_X, pool_Y = prepare_initial_data(X_df, y_df, M=M, seed=seed)
    
    # 全データを結合（データベースとして使用）
    import torch
    X_all = torch.cat([train_X, pool_X], dim=0)
    y_all = torch.cat([train_Y, pool_Y], dim=0)
    
    # 探索空間の境界
    bounds = get_search_bounds(X_df)
    
    # 仮想実験評価器
    evaluator = DatabaseEvaluator(X_all, y_all, compute_config.device)
    
    # 4. ベイズ最適化の実行
    print(f"\n[Step 4/6] Running Bayesian Optimization...")
    optimizer = BayesianOptimizer(
        bounds=bounds,
        evaluator=evaluator,
        compute_config=compute_config,
        acq_function=acq_func
    )
    
    results = optimizer.optimize_loop(
        train_X=train_X,
        train_Y=train_Y,
        T=T,
        verbose=args.verbose
    )
    
    # 5. 結果の保存
    print(f"\n[Step 5/6] Saving results to {args.output}/...")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_results_summary(
        results=results,
        feature_names=feature_names,
        output_dir=args.output,
        compute_mode=compute_config.mode
    )
    
    # 6. 可視化
    if not args.no_plot:
        print(f"\n[Step 6/6] Generating visualizations...")
        
        # 収束曲線
        plot_convergence(
            history=results["history"],
            output_path=output_path / "convergence_curve.png"
        )
        
        # 探索空間
        plot_explored_space(
            train_X=results["train_X"],
            train_Y=results["train_Y"],
            feature_names=feature_names,
            output_path=output_path / "explored_space.png"
        )
        
        # GP予測分布
        plot_gp_predictions(
            model=results["final_model"],
            bounds=bounds,
            feature_names=feature_names,
            train_X=results["train_X"],
            train_Y=results["train_Y"],
            device=compute_config.device,
            output_path=output_path / "gp_predictions.png"
        )
        
        # 計算パフォーマンス
        plot_compute_performance(
            history=results["history"],
            compute_mode=compute_config.mode,
            output_path=output_path / "compute_performance.png"
        )
        
        print(f"\nAll plots saved to {args.output}/")
    else:
        print("\n[Step 6/6] Skipping visualization (--no-plot specified)")
    
    # 最終サマリー
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print(f"\nBest objective value (log10 cycles): {results['best_y']:.4f}")
    print(f"Estimated cycles to failure: {10**results['best_y']:.2e}")
    print(f"\nBest conditions:")
    best_x = results['best_x'].numpy()
    for i, name in enumerate(feature_names):
        print(f"  {name}: {best_x[i]:.4f}")
    print(f"\nResults saved to: {args.output}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
