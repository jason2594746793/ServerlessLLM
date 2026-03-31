# ServerlessLLM Thesis Experiment Guide

All experiments are in `benchmarks/` and share utilities from `exp_common.py`.
Results are written to `benchmarks/results/<exp_name>/`.

---

## Prerequisites

```bash
# 1. Start the full cluster (pylet head + worker + sllm gateway with batch scheduler)
bash scripts/start_cluster_batch.sh

# The script sets required env vars (CUDA_MPS_PIPE_DIRECTORY, LD_LIBRARY_PATH),
# kills any previous run, and starts all three processes.
# Logs: pylet_head_batch.log, pylet_worker_batch.log, sllm_head_batch.log

# 2. Verify the cluster is healthy
curl http://localhost:8343/status

# 3. Pre-download models (only needed once)
python benchmarks/download_models.py
```

> **Prefetch experiments (exp4, exp5_standard_workload):** The script does not set
> `--mem-pool-size`. Edit `scripts/start_cluster_batch.sh` and add
> `--mem-pool-size 20000000000` (20 GB for 7–8 B models) to the `$SLLM_BIN start`
> command, or prefetch will silently fall back to cold loading.
>
> The pool must be ≥ the largest model checkpoint size.

---

## Running all experiments

```bash
cd /mnt/raid0sata2/jingsheng/ServerlessLLM

# Each experiment is self-contained — run individually:
python benchmarks/exp1_strategy_makespan.py
python benchmarks/exp2_model_switching.py
python benchmarks/exp3_concurrency_mode.py
python benchmarks/exp4_prefetch.py
python benchmarks/exp5_scalability.py
python benchmarks/exp5_standard_workload.py
python benchmarks/exp6_johnsons_rule.py
python benchmarks/exp7_multi_gpu.py    # requires manual cluster reconfiguration
```

---

## Experiment descriptions

### Exp 1 — Scheduling Strategy Impact on Makespan
**File:** `exp1_strategy_makespan.py`
**Thesis section:** §4.2
**Question:** How much does the scheduling strategy (FIFO / Random / Model-Grouping) affect total makespan?

**What it does:**
- 100 tasks, 2 models (50 each), submitted interleaved (ABABAB…)
- Runs 3 strategies in sequence: FIFO baseline, Random, Model-Grouping
- 3 runs per strategy, reports mean ± std makespan

**Expected result:** Grouping < FIFO ≈ Random (grouping eliminates redundant model switches)

**Output:** `results/exp1_strategy_makespan/{strategy}_run{n}.json`, `summary.json`

---

### Exp 2 — Model Switching Cost
**File:** `exp2_model_switching.py`
**Thesis section:** §4.3
**Question:** What is the raw cost of switching between models?

**What it does:**
- Measures the cold load time for each model individually
- Then measures alternating A→B→A→B patterns (2, 4, 8 switches)
- Quantifies the per-switch overhead in seconds

**Expected result:** Each switch adds ~load_time(model_B) to makespan; grouping eliminates N-1 switches.

**Output:** `results/exp2_model_switching/`

---

### Exp 3 — Concurrency Mode Comparison
**File:** `exp3_concurrency_mode.py`
**Thesis section:** §4.4
**Question:** Does the semaphore-based concurrency mode outperform the synchronous (FIFO) mode?

**What it does:**
- 60 tasks, 3 models (interleaved)
- Compares `sync` (one task at a time) vs `semaphore` (buffer_limit=8 pipeline)
- Both run with model-grouping ON to isolate the concurrency effect
- 3 runs each

**Expected result:** Semaphore mode reduces makespan by overlapping I/O and compute across tasks.

**Output:** `results/exp3_concurrency_mode/`

---

### Exp 4 — Checkpoint Prefetch Ablation
**File:** `exp4_prefetch.py`
**Thesis section:** §4.5
**Question:** Does prefetching the next model's checkpoint to CPU while the current group runs reduce makespan?

**What it does:**
- 90 tasks, 3 models (interleaved → sorted to 3 groups)
- 2 configs: `enable_prefetch=False` (cold switch) vs `enable_prefetch=True, threshold=0.0` (eager)
- Measures makespan and per-group latency
- 3 runs per config

**Expected result:** Prefetch reduces the inter-group gap (SSD→CPU overlaps with GPU inference).

**Prefetch architecture:**
`prefetch_to_cpu()` → `SllmStoreClient.load_into_cpu()` → NVMe → pinned memory pool
Later: pinned memory → GPU via PCIe (`cudaMemcpyHostToDevice`)

**Output:** `results/exp4_prefetch/`

---

### Exp 5 — Makespan Scalability
**File:** `exp5_scalability.py`
**Thesis section:** §4.6
**Question:** Does the gap between FIFO and the Shared-Aware strategy grow as batch size increases?

**What it does:**
- Batch sizes: 10, 30, 60, 120, 200 tasks
- 3 models, interleaved submission
- 2 strategies: FIFO baseline vs Shared-Aware (grouping + JR + prefetch)
- 3 runs per (strategy, batch_size) pair

**Expected result:** FIFO makespan scales roughly linearly with tasks (constant switch ratio); Shared-Aware amortises switch cost, so the gap widens.

**Output:** `results/exp5_scalability/{strategy}_{N}tasks_run{n}.json`

---

### Exp 5b — Standard Production Workload
**File:** `exp5_standard_workload.py`
**Thesis section:** §4.5 (production-scale validation)
**Question:** Do prefetch benefits hold under realistic production-like conditions?

**What it does:**
- 500 tasks, 3 models with realistic mix: 50% small, 35% medium, 15% large
- 2 configs: prefetch OFF vs prefetch ON (eager, threshold=0.0)
- Single run (500 tasks takes significant time)

**Standard workload definition:**

| Model | Size | Share |
|---|---|---|
| Qwen3-0.6B | ~1.2 GB | 50% (250 tasks) |
| Qwen2.5-7B-Instruct | ~14.3 GB | 35% (175 tasks) |
| Qwen3-8B | ~15.4 GB | 15% (75 tasks) |

**Output:** `results/exp5_standard_workload/`

---

### Exp 6 — Johnson's Rule Ablation
**File:** `exp6_johnsons_rule.py`
**Thesis section:** §4.7
**Question:** Does Johnson's Rule group ordering (vs alphabetical) reduce makespan?

**What it does:**
- 90 tasks, uneven mix: 50× small, 30× medium, 10× large
  (uneven counts amplify ordering effects)
- 2 configs: grouping + prefetch with JR OFF vs JR ON
- Records actual group execution order and per-group timing
- 3 runs per config

**Johnson's Rule mapping:**
- Machine A = I/O (SSD → CPU): time proportional to checkpoint size
- Machine B = GPU compute: inference time per group
- With NVMe at ~3 GB/s, all groups are compute-heavy (P >> I)
- JR sorts compute-heavy groups by I ascending → 0.6B → 7B → 8B

**Expected result:** Small benefit from JR on NVMe (I times are short); larger benefit on HDD.

**Output:** `results/exp6_johnsons_rule/`

---

### Exp 7 — Multi-GPU Scaling
**File:** `exp7_multi_gpu.py`
**Thesis section:** §4.8
**Question:** How does the Shared-Aware scheduler scale across 1, 2, 4, 8 GPUs?

**What it does:**
- 200 tasks, 3 models, interleaved
- Full strategy (grouping + JR + prefetch) for all GPU counts
- **Pauses between GPU counts and prompts operator to reconfigure cluster**
- 3 runs per GPU count

**Cluster reconfiguration steps (between GPU counts):**

```bash
# Stop current workers
sllm-store stop

# Restart with N GPUs (set CUDA_VISIBLE_DEVICES accordingly)
CUDA_VISIBLE_DEVICES=0,1 sllm-store start --mem-pool-size 20000000000  # 2 GPUs
# Then press ENTER in the experiment script
```

**Expected result:**
- Near-linear speedup up to 3 GPUs (one per model group)
- Beyond 3 GPUs: diminishing returns (more GPUs than distinct model groups)
- Throughput scales better than makespan at high GPU counts (inference parallelism)

**Output:** `results/exp7_multi_gpu/gpu{N}_run{n}.json`

---

## Cold-start protocol

Every experiment run calls `clear_deployments()` + `verify_cold_state()` between runs.
This ensures:
1. All model replicas are stopped (GPU memory freed)
2. No residual CUDA contexts affect the next run
3. A 15-second stabilisation pause after GPU release

If `verify_cold_state()` fails, the run proceeds with a `[WARN]` but is flagged in the result JSON.

---

## Result format

Every run saves a JSON file with at least:

```json
{
  "run": 1,
  "makespan": 42.3,
  "throughput": 2.13,
  "avg_latency": 3.8,
  "p50_latency": 3.5,
  "p95_latency": 5.2,
  "p99_latency": 6.1,
  "total_tasks": 90,
  "batch_id": "..."
}
```

Aggregate files add:

```json
{
  "aggregate": {
    "mean": 41.8,
    "std": 1.2,
    "min": 40.5,
    "max": 43.1,
    "values": [40.5, 41.8, 43.1]
  }
}
```

---

## Plotting results

```bash
# After running experiments, generate all figures
python benchmarks/plot_results.py

# Figures are written to figures/
ls figures/
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConnectionRefusedError` | API gateway not running | `python -m sllm.api_gateway` |
| Prefetch has no effect | `mem_pool_size` < model size | Increase `--mem-pool-size` |
| `clear_deployments` timeout | GPU stuck in zombie process | Restart sllm-store |
| All makespans identical | Strategy API not applied | Check `/admin/set_strategy` response |
| Exp 7 wrong GPU count | `CUDA_VISIBLE_DEVICES` not set | Set before starting sllm-store |

---

## Experiment dependency