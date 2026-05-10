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
  python main_demo.py --dataset-name complex_alloy --M 10 --T 20
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
    
    # データセット設定
    parser.add_argument(
        '--dataset-name',
        type=str,
        choices=['snc', 'tests', 'tension_compression', 'complex_alloy'],
        help='Dataset name (overrides config.yaml)'
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
    dataset_name = args.dataset_name if args.dataset_name else opt_params.get('dataset_name', 'tension_compression')
    
    # データセット別の設定を読み込む
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    
    # dataset_nameに応じた設定を取得
    dataset_config = {}
    if dataset_name == 'complex_alloy' and 'complex_alloy' in full_config:
        dataset_config = full_config['complex_alloy']
        print(f"[Config] Using complex_alloy specific configuration")
    else:
        dataset_config = full_config.get('optimization', {})
    
    M = args.M if args.M is not None else opt_params.get('M', 10)
    T = args.T if args.T is not None else opt_params.get('T', 20)
    seed = args.seed if args.seed is not None else opt_params.get('seed', 42)
    # complex_alloyの場合はLogEIを推奨
    default_acq = 'LogEI' if dataset_name == 'complex_alloy' else 'EI'
    acq_func = args.acq_function if args.acq_function else opt_params.get('acq_function', default_acq)
    
    print(f"\nOptimization parameters:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Initial samples (M): {M}")
    print(f"  BO iterations (T): {T}")
    print(f"  Random seed: {seed}")
    print(f"  Acquisition function: {acq_func}")
    
    # 2. データの読み込み
    print(f"\n[Step 2/6] Loading data (dataset: {dataset_name})...")
    pca_transformer = None  # PCA変換器を保存（逆変換用）
    composition_cols_original = None  # 元の組成列名を保存
    
    try:
        X_df, y_df, feature_names, objective_name = load_ccfatigue_data(
            dataset_name=dataset_name,
            design_variables=dataset_config.get('design_variables'),
            objective_variable=dataset_config.get('objective_variable', opt_params.get('objective_variable', 'cycles_to_failure')),
            objective_transform=dataset_config.get('objective_transform', opt_params.get('objective_transform', 'log10')),
            filter_conditions=dataset_config.get('filter_conditions') if dataset_name == 'complex_alloy' else None
        )
        
        # データの統計情報
        data_info = describe_data(X_df, y_df)
        print(f"  Total samples: {data_info['n_samples']}")
        print(f"  Features (before PCA): {data_info['feature_names']}")
        
        # complex_alloyの場合、PCAを適用
        if dataset_name == 'complex_alloy' and dataset_config.get('use_pca', False):
            print(f"\n[Step 2b/6] Applying PCA to composition data...")
            from src.feature_engineering import CompositionPCA
            
            # 組成列と試験条件列を識別
            composition_cols = dataset_config.get('composition_columns', ['Fe', 'Ni', 'Cr', 'Co', 'Al'])
            # X_dfに存在する組成列のみを使用
            composition_cols = [col for col in composition_cols if col in X_df.columns]
            other_cols = [col for col in X_df.columns if col not in composition_cols]
            
            if composition_cols:
                # PCA変換
                pca_transformer = CompositionPCA(
                    variance_threshold=dataset_config.get('pca_variance_threshold', 0.95),
                    n_components=dataset_config.get('pca_n_components')
                )
                X_df = pca_transformer.fit_transform(X_df, composition_cols, other_cols)
                feature_names = list(X_df.columns)
                composition_cols_original = composition_cols  # 逆変換用に保存
                
                # PCA結果の表示
                pca_info = pca_transformer.get_explained_variance_info()
                print(f"  PCA: {len(composition_cols)} → {pca_info['n_components']} components")
                print(f"  Cumulative variance: {pca_info['cumulative_variance_ratio'][-1]:.2%}")
                print(f"  Features (after PCA): {feature_names}")
            else:
                print(f"  Warning: No composition columns found for PCA")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
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
    
    # PCA逆変換：主成分を元の組成に戻す
    if pca_transformer is not None and composition_cols_original is not None:
        print(f"\nOptimal Material Composition (PCA inverse transform):")
        print("="*50)
        
        # PC列のインデックスを取得
        pc_cols = [name for name in feature_names if name.startswith('PC')]
        pc_indices = [i for i, name in enumerate(feature_names) if name.startswith('PC')]
        
        if pc_indices:
            # PC値を抽出
            best_pcs = best_x[pc_indices].reshape(1, -1)
            
            # 元の組成に逆変換
            composition_df = pca_transformer.inverse_transform_composition(
                best_pcs, 
                normalize=True
            )
            
            # 組成を表示
            print("\nElement concentrations (wt%):")
            for col in composition_df.columns:
                value = composition_df[col].iloc[0]
                print(f"  {col}: {value:.2f}%")
            
            # 組成をJSONファイルに保存
            import json
            composition_dict = {
                'composition_wt_percent': {col: float(composition_df[col].iloc[0]) for col in composition_df.columns},
                'pca_values': {pc_cols[i]: float(best_x[idx]) for i, idx in enumerate(pc_indices)},
                'test_conditions': {name: float(best_x[i]) for i, name in enumerate(feature_names) if not name.startswith('PC')}
            }
            
            composition_file = output_path / "optimal_composition.json"
            with open(composition_file, 'w', encoding='utf-8') as f:
                json.dump(composition_dict, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Optimal composition saved to: {composition_file}")
            print("="*50)
    
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
