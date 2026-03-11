# Benchmark Scripts — Thesis Experiments

## Quick Start

```bash
# Run all 6 experiments
python benchmarks/run_all_experiments.py

# Run specific experiments (e.g., 1, 3, 6)
python benchmarks/run_all_experiments.py 1 3 6

# Run single experiment directly
python benchmarks/exp2_model_switching.py
```

## Prerequisites

1. Cluster must be running: `./scripts/start_cluster_batch.sh`
2. Admin API must be available at `http://localhost:8343/admin/set_strategy`
3. At least models `Qwen/Qwen3-0.6B` and `Qwen/Qwen2.5-7B-Instruct` should be accessible

---

## Experiment Index

| # | Script | Thesis §  | Purpose | Duration |
|---|--------|-----------|---------|----------|
| 1 | `exp1_strategy_makespan.py` | §4.2 | Compare FIFO/Random/Grouping/Shared-Aware on Makespan | ~30 min |
| 2 | `exp2_model_switching.py` | §4.3 | Quantify model switching overhead (4 scenarios) | ~30 min |
| 3 | `exp3_prefetching.py` | §4.4 | Measure checkpoint prefetching effect | ~20 min |
| 4 | `exp4_prefix_sharing.py` | §4.5 | Prefix-sharing-aware scheduling effect | ~15 min |
| 5 | `exp5_scalability.py` | §4.6 | Makespan vs batch size (10→200) | ~60 min |
| 6 | `exp6_model_size.py` | §4.7 | Model size impact (0.6B / 7B / 8B) | ~15 min |

**Total estimated time: ~3 hours**

---

## File Structure

```
benchmarks/
├── exp_common.py              # Shared utilities (API, submit, metrics)
├── exp1_strategy_makespan.py  # Exp 1: Strategy comparison
├── exp2_model_switching.py    # Exp 2: Switching overhead
├── exp3_prefetching.py        # Exp 3: Prefetching
├── exp4_prefix_sharing.py     # Exp 4: Prefix sharing
├── exp5_scalability.py        # Exp 5: Scalability
├── exp6_model_size.py         # Exp 6: Model size
├── run_all_experiments.py     # Master runner
├── download_models.py         # Download required models
├── benchmark_utils.py         # Legacy sllm-store utils (kept)
└── README.md                  # This file

results/
├── exp1_strategy_makespan/    # → summary.json + per-config .json
├── exp2_model_switching/      # → summary.json + per-scenario .json
├── exp3_prefetching/          # → summary.json + per-config .json
├── exp4_prefix_sharing/       # → summary.json + per-order .json
├── exp5_scalability/          # → summary.json + per-strategy-size .json
└── exp6_model_size/           # → summary.json + per-model .json
```

---

## Experiment Details

### Exp 1: Scheduling Strategy Impact on Makespan
- **Workload**: 100 tasks, 2 models interleaved (ABABAB...)
- **Configs**: FIFO (sync, no grouping), Random (semaphore, no grouping), Model Grouping (semaphore, grouping), Shared-Aware (semaphore, grouping)
- **Metric**: Makespan = wall-clock time from submit to last task done
- **Expected**: Shared-Aware < Grouping < FIFO ≈ Random

### Exp 2: Model Switching Overhead
- **Workload**: 20 tasks, 2 models
- **Scenarios**: Single-A (0 switches), Single-B (0 switches), Grouped (1 switch), Alternating (19 switches)
- **Mode**: sync + no grouping → forces exact submission order
- **Expected**: Alternating ≫ Grouped ≈ Single

### Exp 3: Checkpoint Prefetching
- **Workload**: 90 tasks, 3 models × 30 each, grouped
- **Configs**: no_prefetch (sync) vs with_overlap (semaphore — allows concurrent load/compute)
- **Expected**: Overlap config hides some model load latency

### Exp 4: Prefix Sharing
- **Workload**: 50 tasks, single model, shared 2000-token prefix
- **Configs**: Shuffled order (bad for cache) vs Sequential order (good for cache)
- **Mode**: sync → ordering matters
- **Expected**: Sequential order benefits from prefix cache hits

### Exp 5: Scalability
- **Workload**: 10/25/50/100/200 tasks, 2 models interleaved
- **Strategies**: FIFO vs Shared-Aware
- **Expected**: Shared-Aware advantage grows with batch size

### Exp 6: Model Size Impact
- **Workload**: 50 tasks per model size
- **Models**: Qwen3-0.6B, Qwen2.5-7B, Qwen3-8B
- **Purpose**: Establish per-model throughput baselines for cost model

---

## Admin API Reference

```bash
# Set scheduling strategy
curl -X POST http://localhost:8343/admin/set_strategy \
  -H "Content-Type: application/json" \
  -d '{"strategy":"semaphore","buffer_limit":10,"enable_model_grouping":true}'

# Get current strategy
curl http://localhost:8343/admin/get_strategy
```

Valid strategies: `sync`, `chunked`, `semaphore`