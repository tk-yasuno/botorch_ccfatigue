# Material Usability Maximizer MVP

**Adaptive Experimental Design for Material Lifetime Maximization using Bayesian Optimization (BoTorch + Gaussian Process)**

[![Version](https://img.shields.io/badge/version-v0.1.1-blue.svg)](https://github.com/tk-yasuno/botorch_ccfatigue/releases)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![BoTorch](https://img.shields.io/badge/BoTorch-0.17.2-orange.svg)](https://botorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## Overview

This project demonstrates an **MVP (Minimum Viable Product)** for adaptive experimental design that efficiently searches for manufacturing conditions maximizing material fatigue lifetime. Using real CCFatigue dataset (762 samples), we validate the feasibility of Bayesian optimization for material lifetime prediction.

### Key Features

- **BoTorch-based Bayesian Optimization**: Advanced exploration using SingleTaskGP model with Expected Improvement acquisition function
- **3 Compute Modes**: 
  - `no_parallel`: Single-threaded execution (reproducibility focused)
  - `cpu_16cores`: 16-core CPU parallel processing
  - `gpu_16gb`: GPU acceleration (16GB memory constraint handling)
- **Comprehensive Visualization**: Convergence curves, explored space, GP predictions, compute performance
- **Flexible Configuration**: Control via `config.yaml` and command-line arguments
- **Interactive Demo**: Step-by-step execution with Jupyter Notebook

### Verified Results (CCFatigue Dataset)

Using **762 real fatigue test samples** from EPFL CCFatigue Platform:

- **Optimal Lifetime Found**: **1.33 billion cycles** (log10: 9.12)
- **Optimal Conditions**: 
  - `stress_max = 7.70 MPa` (low-to-moderate stress)
  - `stress_ratio = 0.1` (standard R-value)
- **Efficiency**: Found optimal conditions in **10 iterations** (3.87 seconds, CPU 16-core)
- **Data Efficiency**: 20 evaluations from 762-point database

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/tk-yasuno/botorch_ccfatigue.git
cd botorch_ccfatigue

# Create virtual environment (recommended)
python -m venv .venv-bo
.venv-bo\Scripts\activate  # Windows
# source .venv-bo/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download CCFatigue data
mkdir data
curl -o data/ccfatigue_snc.csv https://raw.githubusercontent.com/EPFL-ENAC/CCFatiguePlatform/develop/Data/samples/SNC_sample_2022-09.csv
```

### Basic Execution

Run with default settings (auto device detection):

```bash
python main_demo.py
```

### Quick Demo (Small Scale)

```bash
# Fast validation: 5 initial points, 3 iterations
python main_demo.py --M 5 --T 3 --seed 42
```

Expected output:
```
Best objective value (log10 cycles): 8.9191
Estimated cycles to failure: 8.30e+08
Best conditions: stress_max=5.75 MPa, stress_ratio=-1.0
```

## Usage

### Compute Mode Selection

```bash
# Single-threaded (reproducibility priority)
python main_demo.py --compute-mode no_parallel

# 16-core CPU parallel processing
python main_demo.py --compute-mode cpu_16cores

# GPU acceleration
python main_demo.py --compute-mode gpu_16gb
```

### Parameter Customization

```bash
# Change initial sample size and iterations
python main_demo.py --M 15 --T 30

# Specify random seed (reproducibility)
python main_demo.py --seed 12345

# Use LogEI acquisition function (numerically stable)
python main_demo.py --acq-function LogEI

# Specify output directory
python main_demo.py --output results_custom

# Disable visualization (headless mode)
python main_demo.py --no-plot
```

### Configuration File

Edit `config.yaml` to change default settings:

```yaml
compute_mode: cpu_16cores
optimization:
  M: 10                      # Initial sample size
  T: 20                      # Number of BO iterations
  seed: 42                   # Random seed
  acq_function: LogEI        # Acquisition function (EI or LogEI)
  dataset_name: snc          # CCFatigue dataset name
  design_variables:
    - stress_max
    - stress_ratio
  objective_variable: cycles_to_failure
  objective_transform: log10
```

### Interactive Jupyter Demo

Launch interactive step-by-step demonstration:

```bash
jupyter notebook demo_notebook.ipynb
```

## Results & Visualization

### Convergence Curve
![Convergence Curve](results/convergence_curve.png)

**Analysis**: Rapid convergence at iteration 2, reaching optimal lifetime of 1.33 billion cycles. The algorithm efficiently identifies the best conditions within 20 evaluations from a 762-sample database.

### Explored Design Space
![Explored Space](results/explored_space.png)

**Analysis**: The optimization effectively explores the low-stress region (5-10 MPa) where maximum lifetime is achieved. Blue points indicate high objective values (long lifetime), confirming optimal region discovery.

### Gaussian Process Predictions
![GP Predictions](results/gp_predictions.png)

**Analysis**: GP model accurately learns the stress-lifetime relationship. The mean prediction (blue line) closely follows training data (orange points), with uncertainty bands (shaded regions) appropriately quantifying prediction confidence.

### Compute Performance
![Compute Performance](results/compute_performance.png)

**Analysis**: Stable iteration times averaging 0.39 seconds with CPU 16-core mode. First iteration takes longer (0.89s) due to initial GP training, while subsequent iterations remain consistent at 0.2-0.4s.

## Project Structure

```
botorch_ccfatigue/
├── src/                          # Source code
│   ├── __init__.py
│   ├── compute_config.py         # Compute environment management
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── bayesian_optimizer.py     # Bayesian optimization engine
│   └── visualization.py          # Visualization functions
├── data/                         # CCFatigue data (downloaded separately)
│   ├── ccfatigue_snc.csv         # S-N curve data (762 samples)
│   └── ccfatigue_tests.csv       # Fatigue test data (optional)
├── doc/                          # Documentation
│   ├── Concept_20260510.pdf      # Concept diagram
│   ├── Implementation_Notes.md   # Technical implementation details
│   └── CCFatigue_Data_Lesson.md  # Data acquisition guide
├── results/                      # Results output (gitignored)
│   ├── convergence_curve.png
│   ├── explored_space.png
│   ├── gp_predictions.png
│   ├── compute_performance.png
│   ├── optimization_history.csv
│   ├── best_conditions.json
│   └── summary.txt
├── main_demo.py                  # Standalone execution script
├── demo_notebook.ipynb           # Jupyter demo
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── README.md                     # This file (English)
└── README_JP.md                  # Japanese version
```

## Concept

This system implements the following adaptive experimental design cycle:

1. **t: iteration** - Iteratively explore experimental conditions (starting from t=1)
2. **ym: deterioration index** - Measure deterioration indicator (fatigue lifetime)
3. **Bayes Optimize fm(x,t)** - Propose next candidate point via Bayesian optimization
4. **Bayes update → t+1** - Update GP model with observed data and proceed to next iteration

See `doc/Concept_20260510.pdf` for detailed workflow diagram.

## Technical Stack

- **Optimization**: BoTorch 0.17.2, GPyTorch 1.15.2
- **Machine Learning**: PyTorch 2.11.0
- **Data Processing**: pandas, NumPy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Material Data**: CCFatigue (EPFL)

## Output Files

After execution, the following files are saved to `results/` directory:

- **Visualizations** (PNG images):
  - `convergence_curve.png`: Optimization convergence plot (best_y vs iteration)
  - `explored_space.png`: 2D scatter of explored design space
  - `gp_predictions.png`: GP posterior predictions with uncertainty (1D slices)
  - `compute_performance.png`: Iteration times and memory usage
  
- **Data Files**:
  - `optimization_history.csv`: Detailed iteration history (iteration, best_y, current_y, time, memory)
  - `best_conditions.json`: Discovered optimal conditions in JSON format
  - `summary.txt`: Execution summary report

## CCFatigue Dataset

### Data Source
- **Provider**: EPFL (École Polytechnique Fédérale de Lausanne)
- **Platform**: CCFatigue Platform - https://ccfatigue.epfl.ch
- **GitHub**: https://github.com/EPFL-ENAC/CCFatiguePlatform

### Dataset Statistics
- **Dataset**: SNC_sample_2022-09.csv (S-N Curve)
- **Total Samples**: 762
- **Design Variables**:
  - `stress_max`: Maximum stress [5.53 - 32.91 MPa]
  - `stress_ratio`: Stress ratio R-value [-1.0 - 10.0]
- **Objective Variable**: 
  - `cycles_to_failure`: Cycles to failure [1 - 10^9]
  - Transform: log10 scale [0.00 - 9.12]

### Data Acquisition

Download real CCFatigue data:

```powershell
# PowerShell (Windows)
New-Item -ItemType Directory -Force -Path "data"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/EPFL-ENAC/CCFatiguePlatform/develop/Data/samples/SNC_sample_2022-09.csv" -OutFile "data\ccfatigue_snc.csv"
```

```bash
# Bash (Linux/Mac)
mkdir -p data
curl -o data/ccfatigue_snc.csv https://raw.githubusercontent.com/EPFL-ENAC/CCFatiguePlatform/develop/Data/samples/SNC_sample_2022-09.csv
```

See `doc/CCFatigue_Data_Lesson.md` for detailed data acquisition guide.

## Validation Results

### Experiment Configuration
- **Dataset**: 762 real fatigue test samples
- **Initial Points (M)**: 10
- **BO Iterations (T)**: 10
- **Acquisition Function**: LogEI
- **Compute Mode**: CPU 16-core
- **Random Seed**: 42

### Optimization Performance
- **Total Time**: 3.87 seconds
- **Mean Iteration Time**: 0.39 seconds
- **Convergence**: Reached optimal at iteration 2 (early convergence)
- **Data Efficiency**: 20 evaluations (2.6% of 762 samples)

### Discovered Optimal Conditions
- **Maximum Lifetime**: **1.33 billion cycles** (1.33×10^9)
- **Objective Value**: 9.12 (log10 scale)
- **Optimal stress_max**: 7.70 MPa (low-to-moderate stress region)
- **Optimal stress_ratio**: 0.1 (standard tensile-tensile R-value)

### Key Insights
1. **Low Stress = Long Life**: Optimal conditions found in low stress region (5-10 MPa)
2. **R-value Impact**: stress_ratio = 0.1 is optimal for this material
3. **Efficient Exploration**: Found optimum with only 20 evaluations from 762-point database
4. **Fast Execution**: Practical speed (<4 seconds) for iterative design cycles

## Troubleshooting

### GPU Mode on Non-GPU System

Automatically falls back to CPU mode with a warning message.

### CCFatigue Data Not Found

If `data/ccfatigue_snc.csv` is missing:
1. Download data as described in "Data Acquisition" section
2. System will use mock data as fallback (warning displayed)

### Memory Insufficient Error

If GPU memory is insufficient, adjust in `config.yaml`:
```yaml
gpu_16gb:
  num_restarts: 5   # Reduce from default 10
  raw_samples: 64   # Reduce from default 128
```

### Numerical Warnings (ExpectedImprovement)

If you see warnings about numerical issues:
```bash
# Use LogEI instead of EI (recommended)
python main_demo.py --acq-function LogEI
```

Or update `config.yaml`:
```yaml
optimization:
  acq_function: LogEI
```

## Future Extensions

- **Real Equipment Integration**: Connect to actual testing machines for closed-loop optimization
- **Multi-Objective Optimization**: Simultaneous optimization of lifetime + cost + manufacturability
- **Explicit Constraints**: Handle manufacturing constraints (stress limits, safety factors)
- **Batch Experiments**: Parallel experiments using qEI (qExpectedImprovement)
- **Dynamic Degradation Modeling**: Time-series modeling of material degradation
- **Distributed Computing**: Scale to multiple GPUs/nodes

## Dependencies

Core packages (see `requirements.txt` for complete list):

```txt
botorch>=0.9.0
gpytorch>=1.11
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
pyyaml>=6.0
psutil>=5.9.0
jupyter>=1.0.0
notebook>=6.5.0
ipywidgets>=8.0.0
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

### Development Setup

```bash
# Clone repository
git clone https://github.com/tk-yasuno/botorch_ccfatigue.git
cd botorch_ccfatigue

# Create development environment
python -m venv .venv-bo
source .venv-bo/bin/activate

# Install with development dependencies
pip install -r requirements.txt

# Run tests (future work)
pytest tests/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{material_usability_maximizer_2026,
  title = {Material Usability Maximizer MVP},
  author = {Yasuno, Takato},
  year = {2026},
  note = {Adaptive experimental design for material lifetime maximization},
  url = {https://github.com/tk-yasuno/botorch_ccfatigue}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Contact

For questions, bug reports, or feature requests, please open an issue on GitHub:

**GitHub Issues**: https://github.com/tk-yasuno/botorch_ccfatigue/issues

Project Maintainer: [@tk-yasuno](https://github.com/tk-yasuno)

## Acknowledgments

- **BoTorch Team**: High-quality Bayesian optimization framework
- **CCFatigue Project (EPFL)**: Providing open fatigue test datasets
- **PyTorch Team**: Robust deep learning and tensor computation infrastructure
- **GPyTorch Team**: Efficient Gaussian process implementations

---

**Version**: v0.1.1

**Built with**: BoTorch 0.17.2 | PyTorch 2.11.0 | Python 3.12.10

**Last Updated**: 2026-05-10
