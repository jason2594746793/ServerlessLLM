#!/usr/bin/env python3
"""
Exp W3 — Small-scale FIFO Measurement (3 models, 2 tasks each)
==============================================================
Submits tasks ONE AT A TIME in round-robin order (0.6B, 8B, 32B, 0.6B, 8B, 32B),
forcing a cold model load for every task. This directly measures the cold-switch
cost for each transition type and extrapolates to 501 tasks.

Switch sequence (2 per model):
  0.6B → 8B → 32B → 0.6B → 8B → 32B
  Types: 0.6B→8B (×2), 8B→32B (×2), 32B→0.6B (×1)

Each task is submitted as its own single-task batch so the system can't
pipeline tasks across model switches. clear_deployments() between each task
ensures GPU is fully released before the next model loads.

Per-switch cost formula (per task):
  l_switch_i = makespan_i - infer_latency_i  (cold load overhead)

Extrapolation to 501 tasks:
  C_FIFO_501 = 500 × mean(l_switch) + 501 × mean(infer)

GPU requirement: 4 GPUs (TP=4 for 32B).
Runtime: ~5–8 min per run, ~20–25 min for 3 runs.
Results saved to: results/exp_fifo_small/
"""

import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy,
    submit_batch,
    extract_task_metrics,
    save_result,
    make_task,
    clear_deployments,
    verify_cold_state,
    register_model_deployment,
    evict_model_page_cache,
    aggregate_runs,
    print_header,
    print_separator,
    MODEL_SMALL,
    MODEL_LARGE,
    MODEL_XLARGE,
    RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp_fifo_small"
EXP_MODELS  = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]   # 0.6B, 8B, 32B
N_PER_MODEL = 2                                           # 6 total tasks, 5 switches
NUM_RUNS    = 3
MODEL_LABELS = {MODEL_SMALL: "0.6B", MODEL_LARGE: "8B", MODEL_XLARGE: "32B"}

# 501-task extrapolation constants
EXTRAP_TASKS    = 501
EXTRAP_SWITCHES = 500
SHARED_AWARE_MAKESPAN = 164.7   # from Exp 1


def cold_start_before_task(model: str):
    """Evict page cache + clear all deployments before each task."""
    for m in EXP_MODELS:
        evict_model_page_cache(m)
    if not clear_deployments():
        print("  [WARN] Cold-start verification failed, proceeding anyway")
    if not verify_cold_state():
        print("  [WARN] System may not be fully cold")
    # Pre-register 32B with TP=4 whenever the next task needs it,
    # to ensure the auto-creation path uses the right config.
    if model == MODEL_XLARGE:
        register_model_deployment(MODEL_XLARGE, {
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
        })


def submit_one(model: str, task_id: str) -> tuple[float, float]:
    """Submit a single task; return (makespan, infer_latency)."""
    task = make_task(model, "Explain neural networks in one sentence.", task_id)
    batch = submit_batch([task])
    makespan = batch["duration"]
    metrics  = extract_task_metrics(batch["tasks_data"])
    infer    = metrics["latencies"][0] if metrics.get("latencies") else makespan
    return makespan, infer


def run_once(run_idx: int) -> dict:
    """Run one full sequence of 6 tasks (one at a time). Return per-task data."""
    sequence = []   # list of dicts: model, makespan, infer, l_switch

    # Set FIFO strategy once (no grouping, no prefetch)
    set_strategy(
        strategy="semaphore",
        buffer_limit=1,
        enable_model_grouping=False,
        enable_johnsons_rule=False,
        enable_prefetch=False,
        prefetch_threshold=0.0,
    )

    task_idx = 0
    for repeat in range(N_PER_MODEL):
        for model in EXP_MODELS:
            label    = MODEL_LABELS[model]
            task_id  = f"fifo-r{run_idx+1}-t{task_idx}-{label}"
            task_idx += 1

            print(f"\n  ── Task {task_idx}/{N_PER_MODEL*3}  model={label} ──")
            cold_start_before_task(model)

            makespan, infer = submit_one(model, task_id)
            l_switch = makespan - infer

            print(f"  Makespan: {makespan:.1f}s  |  Inference: {infer:.3f}s  |  "
                  f"Cold-load overhead: {l_switch:.1f}s")

            sequence.append({
                "task_idx": task_idx,
                "model":    label,
                "task_id":  task_id,
                "makespan": round(makespan, 2),
                "infer_latency": round(infer, 3),
                "cold_load_overhead": round(l_switch, 2),
            })

    # Compute per-switch stats
    all_makespans  = [s["makespan"] for s in sequence]
    all_infers     = [s["infer_latency"] for s in sequence]
    all_overheads  = [s["cold_load_overhead"] for s in sequence]

    avg_infer    = statistics.mean(all_infers)
    avg_overhead = statistics.mean(all_overheads)

    # Total wall-clock FIFO time for the 6 tasks (sum of individual makespans)
    # In real FIFO: tasks run back-to-back after each model switch.
    # We use implied_switch = avg cold-load overhead (= makespan - infer per task).
    implied_switch = avg_overhead

    # Extrapolate to 501 tasks
    extrap = round(EXTRAP_SWITCHES * implied_switch + EXTRAP_TASKS * avg_infer, 0)
    speedup = round(extrap / SHARED_AWARE_MAKESPAN, 1)

    print(f"\n  ── Run {run_idx+1} summary ──")
    print(f"  Avg cold-load overhead (l_switch): {avg_overhead:.1f}s")
    print(f"  Avg inference latency:             {avg_infer:.3f}s")
    print(f"  Extrapolated FIFO 501 tasks:       {extrap:.0f}s  "
          f"({extrap/3600:.2f}h)  → {speedup}× vs Shared-Aware")

    return {
        "run":            run_idx + 1,
        "sequence":       sequence,
        "avg_cold_load":  round(avg_overhead, 1),
        "avg_infer":      round(avg_infer, 3),
        "implied_switch": round(implied_switch, 1),
        "extrap_501":     extrap,
        "speedup":        speedup,
    }


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print_header("Exp W3 — Small-scale FIFO Measurement", {
        "Models":    "0.6B, 8B, 32B (sequential, one task at a time)",
        "Sequence":  "0.6B→8B→32B→0.6B→8B→32B  (6 tasks, 5 cold switches)",
        "Method":    "Each task submitted alone; clear_deployments() before each",
        "Runs":      NUM_RUNS,
        "Extrap to": f"{EXTRAP_TASKS} tasks ({EXTRAP_SWITCHES} switches)",
        "Ref":       f"Analytical estimate: 42,001s (83.3s/switch)",
        "GPUs":      "4 (TP=4 for 32B)",
    })

    all_runs = []
    for run_idx in range(NUM_RUNS):
        print_separator(f"Run {run_idx + 1}/{NUM_RUNS}")
        result = run_once(run_idx)
        all_runs.append(result)
        save_result(result, OUTDIR / f"run{run_idx+1}.json")

    # Aggregate
    switch_values = [r["implied_switch"] for r in all_runs]
    infer_values  = [r["avg_infer"]      for r in all_runs]
    extrap_values = [r["extrap_501"]     for r in all_runs]

    mean_switch = statistics.mean(switch_values)
    std_switch  = statistics.stdev(switch_values) if len(switch_values) > 1 else 0.0
    mean_infer  = statistics.mean(infer_values)
    mean_extrap = round(statistics.mean(extrap_values), 0)
    std_extrap  = round(statistics.stdev(extrap_values) if len(extrap_values) > 1 else 0.0, 0)
    mean_speedup = round(mean_extrap / SHARED_AWARE_MAKESPAN, 1)

    summary = {
        "experiment":        "exp_fifo_small",
        "n_per_model":       N_PER_MODEL,
        "n_tasks":           N_PER_MODEL * 3,
        "n_switches":        N_PER_MODEL * 3 - 1,
        "runs":              all_runs,
        "mean_switch_cost":  round(mean_switch, 1),
        "std_switch_cost":   round(std_switch, 1),
        "mean_infer":        round(mean_infer, 3),
        "extrap_501_mean":   mean_extrap,
        "extrap_501_std":    std_extrap,
        "speedup_vs_shared": mean_speedup,
        "analytical_ref":    42001,
    }
    save_result(summary, OUTDIR / "summary.json")

    print("\n" + "=" * 68)
    print("RESULTS — Small-scale FIFO Extrapolation (W3)")
    print(f"  {'Metric':<42} {'Value'}")
    print("  " + "-" * 62)
    print(f"  {'Measured l_switch (mean ± std):':<42} "
          f"{mean_switch:.1f} ± {std_switch:.1f}s")
    print(f"  {'Avg inference latency:':<42} {mean_infer:.3f}s")
    print(f"  {'Extrapolated FIFO (501 tasks, mean ± std):':<42} "
          f"{mean_extrap:.0f} ± {std_extrap:.0f}s  ({mean_extrap/3600:.2f}h)")
    print(f"  {'Speedup vs Shared-Aware (164.7s):':<42} {mean_speedup}×")
    print("  " + "-" * 62)
    print(f"  Thesis analytical estimate:            42,001s  (83.3s/switch)")
    diff = mean_extrap - 42001
    print(f"  Difference from analytical:            {diff:+.0f}s  "
          f"({(mean_extrap/42001-1)*100:+.1f}%)")
    print("=" * 68)


if __name__ == "__main__":
    run()
