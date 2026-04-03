#!/usr/bin/env python3
"""
Exp 1 — FIFO Concurrent baseline (run this manually, takes ~2–3 hrs)
=====================================================================
Runs 3 real FIFO batches (semaphore, no grouping) with 501 tasks across
3 models (0.6B, 8B, 32B). Each run takes ~40–60 min due to constant
model switching. Results saved to exp1_strategy_makespan/ for merging
into the Exp 1 summary.

Run with:
    python benchmarks/exp1_fifo_only.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy,
    submit_batch,
    extract_task_metrics,
    analyze_model_groups,
    save_result,
    make_task,
    clear_deployments,
    verify_cold_state,
    register_model_deployment,
    aggregate_runs,
    print_header,
    MODEL_SMALL,
    MODEL_LARGE,
    MODEL_XLARGE,
    RESULTS_ROOT,
    DEFAULT_RUNS,
)

OUTDIR = RESULTS_ROOT / "exp1_strategy_makespan"
NUM_TASKS = 501
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]


def generate_workload():
    tasks = []
    for i in range(NUM_TASKS):
        m = EXP_MODELS[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(make_task(m, f"Explain concept {i + 1} in one sentence.", f"task-{i}-{tag}"))
    return tasks


def run():
    print_header("EXP 1 — FIFO Concurrent (standalone, slow run)", {
        "Models": f"{MODEL_SMALL}, {MODEL_LARGE}, {MODEL_XLARGE}",
        "Tasks": f"{NUM_TASKS} interleaved (no grouping → ~500 model switches)",
        "Runs": DEFAULT_RUNS,
        "Warning": "Each run takes ~40–60 min. Total ~2–3 hrs.",
    })

    runs = []
    for run_idx in range(DEFAULT_RUNS):
        print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

        if not clear_deployments():
            print("  [WARN] Clear incomplete")
        if not verify_cold_state():
            print("  [WARN] May not be cold")
        register_model_deployment(MODEL_XLARGE, {
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
        })

        set_strategy(
            strategy="semaphore",
            buffer_limit=8,
            enable_model_grouping=False,
            enable_johnsons_rule=False,
            enable_prefetch=False,
            prefetch_threshold=0.0,
        )

        batch = submit_batch(generate_workload(), allow_failures=True)
        metrics = extract_task_metrics(batch["tasks_data"])
        makespan = batch["duration"]
        failed = batch.get("failed_count", 0)
        throughput = (NUM_TASKS - failed) / makespan if makespan > 0 else 0

        result = {
            "run": run_idx + 1,
            "makespan": makespan,
            "throughput": round(throughput, 2),
            "failed": failed,
            "groups": analyze_model_groups(batch["tasks_data"]),
            **metrics,
            "batch_id": batch["batch_id"],
        }
        runs.append(result)
        save_result(result, OUTDIR / f"fifo_concurrent_run{run_idx + 1}.json")
        print(f"  Makespan:   {makespan:.1f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        if failed:
            print(f"  Failed:     {failed}")

    agg = aggregate_runs(runs, key="makespan")
    save_result({"config": {"name": "fifo_concurrent"}, "aggregate": agg, "runs": runs},
                OUTDIR / "fifo_concurrent_aggregate.json")

    print(f"\nFIFO Concurrent: {agg['mean']:.1f} ± {agg['std']:.1f}s")
    print(f"Values: {agg['values']}")


if __name__ == "__main__":
    run()
