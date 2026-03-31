# Benchmark Scripts — Thesis Experiments

## Quick Start

```bash
# 1. Start the cluster
bash scripts/start_cluster_batch.sh

# 2. Pre-download models (only needed once)
python benchmarks/download_models.py

# 3. Run all experiments
python benchmarks/run_all_experiments.py

# 4. Run specific experiments (e.g., 1, 3, 7)
python benchmarks/run_all_experiments.py 1 3 7

# 5. Plot results
python benchmarks/plot_results.py
```

## Experiment Index

| # | Script | Thesis | Purpose |
|---|--------|--------|---------|
| 1 | `exp1_strategy_makespan.py` | §4.2 | FIFO / Random / Grouping / Shared-Aware comparison |
| 2 | `exp2_model_switching.py` | §4.3 | Model switching overhead quantification |
| 3 | `exp3_concurrency_mode.py` | §4.4 | Sync vs semaphore concurrency modes |
| 4 | `exp4_prefetch.py` | §4.5 | Checkpoint prefetch ablation |
| 5 | `exp5_scalability.py` | §4.6 | Makespan scalability across batch sizes |
| 6 | `exp5_standard_workload.py` | §4.5b | 500-task production workload validation |
| 7 | `exp6_model_size.py` | §4.7 | Per-model throughput baselines (0.6B/7B/8B) |
| 8 | `exp_multi_gpu_verify.py` | §4.8 | Multi-GPU parallel execution verification |

See **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** for full methodology, prerequisites, and troubleshooting.

## File Structure

```
benchmarks/
├── exp_common.py              # Shared utilities (API, submit, metrics)
├── exp1_strategy_makespan.py  # Exp 1: Strategy comparison
├── exp2_model_switching.py    # Exp 2: Switching overhead
├── exp3_concurrency_mode.py   # Exp 3: Concurrency modes
├── exp4_prefetch.py           # Exp 4: Prefetching
├── exp5_scalability.py        # Exp 5: Scalability
├── exp5_standard_workload.py  # Exp 6: Production workload
├── exp6_johnsons_rule.py      # Johnson's Rule ablation (in guide)
├── exp6_model_size.py         # Exp 7: Model size baselines
├── exp7_multi_gpu.py          # Multi-GPU scaling (in guide)
├── exp_multi_gpu_verify.py    # Exp 8: Multi-GPU verification
├── mock_params.py             # Mock parameters for unit tests
├── run_all_experiments.py     # Master runner
├── download_models.py         # Download required models
├── plot_results.py            # Generate figures from results
├── plot_analytical.py         # Analytical model plots
└── EXPERIMENT_GUIDE.md        # Full documentation
```
