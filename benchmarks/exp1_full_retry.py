#!/usr/bin/env python3
"""
Exp 1 — Full (Shared-Aware) retry
==================================
Re-runs only the 'full' config (grouping + JR + prefetch) with 501 tasks.
Saves results to exp1_strategy_makespan/ overwriting the old 60-task results.

Run with:
    python benchmarks/exp1_full_retry.py
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
    evict_model_page_cache,
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

CFG = {
    "name": "full",
    "description": "Full (Shared-Aware) — grouping + JR + eager prefetch",
    "strategy": "semaphore",
    "buffer_limit": 8,
    "enable_model_grouping": True,
    "enable_johnsons_rule": True,
    "enable_prefetch": True,
    "prefetch_threshold": 0.0,
}


def generate_workload():
    tasks = []
    for i in range(NUM_TASKS):
        m = EXP_MODELS[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(make_task(m, f"Explain concept {i + 1} in one sentence.", f"task-{i}-{tag}"))
    return tasks


def run():
    print_header("EXP 1 — Full (Shared-Aware) retry, 501 tasks", {
        "Models": f"{MODEL_SMALL}, {MODEL_LARGE}, {MODEL_XLARGE}",
        "Tasks": f"{NUM_TASKS} ({NUM_TASKS // 3} per model, interleaved)",
        "Runs": DEFAULT_RUNS,
        "Config": "grouping + JR + prefetch",
    })

    runs = []
    for run_idx in range(DEFAULT_RUNS):
        print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

        if not clear_deployments():
            print("  [WARN] Cold-start verification failed, proceeding anyway")
        if not verify_cold_state():
            print("  [WARN] System may not be fully cold")
        for m in EXP_MODELS:
            evict_model_page_cache(m)
        register_model_deployment(MODEL_XLARGE, {
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
        })

        set_strategy(
            strategy=CFG["strategy"],
            buffer_limit=CFG["buffer_limit"],
            enable_model_grouping=CFG["enable_model_grouping"],
            enable_johnsons_rule=CFG["enable_johnsons_rule"],
            enable_prefetch=CFG["enable_prefetch"],
            prefetch_threshold=CFG["prefetch_threshold"],
        )

        tasks = generate_workload()
        batch = submit_batch(tasks)
        metrics = extract_task_metrics(batch["tasks_data"])
        groups = analyze_model_groups(batch["tasks_data"])
        makespan = batch["duration"]
        failed_count = batch.get("failed_count", 0)
        completed_count = NUM_TASKS - failed_count
        throughput = completed_count / makespan if makespan > 0 else 0

        run_result = {
            "run": run_idx + 1,
            "makespan": makespan,
            "throughput": round(throughput, 2),
            "completed": completed_count,
            "failed": failed_count,
            "fail_rate": round(failed_count / NUM_TASKS, 3) if NUM_TASKS > 0 else 0,
            "groups": groups,
            **metrics,
            "batch_id": batch["batch_id"],
        }
        runs.append(run_result)
        save_result(run_result, OUTDIR / f"full_run{run_idx + 1}.json")

        print(f"  Makespan:   {makespan:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Completed:  {completed_count}/{NUM_TASKS}")
        if failed_count > 0:
            print(f"  Failed:     {failed_count}")

    agg = aggregate_runs(runs, key="makespan")
    save_result({"config": CFG, "aggregate": agg, "runs": runs}, OUTDIR / "full_aggregate.json")

    print(f"\nFull (Shared-Aware): {agg['mean']:.1f} ± {agg['std']:.1f}s")
    print(f"Values: {agg['values']}")


if __name__ == "__main__":
    run()
