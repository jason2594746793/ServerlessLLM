#!/usr/bin/env python3
"""
Experiment 1 — Scheduling Strategy Impact on Makespan
=====================================================
Thesis §4.2: Compare FIFO / Random / Model-Grouping strategies
on a mixed-model batch and measure total Makespan.

Methodology:
  - 100 tasks using 2 models (50 each, interleaved ABABAB... in submission)
  - 3 scheduling configs via admin API
  - Measure: Makespan (wall-clock from submit to last task done)

Expected: Grouping < FIFO ≈ Random
"""

import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy, submit_batch, extract_task_metrics,
    save_result, make_task, MODEL_MEDIUM, MODEL_SMALL, RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp1_strategy_makespan"

# ── workload: interleaved two-model tasks ──────────────────────────────────
def generate_workload(num_tasks: int = 100):
    """Create ABABAB... interleaved tasks."""
    tasks = []
    models = [MODEL_MEDIUM, MODEL_SMALL]
    for i in range(num_tasks):
        m = models[i % 2]
        tag = "A" if i % 2 == 0 else "B"
        tasks.append(make_task(
            model=m,
            prompt=f"Briefly explain concept {i+1} in one sentence.",
            task_id=f"task-{i}-{tag}",
        ))
    return tasks

# ── scheduling configs ─────────────────────────────────────────────────────
CONFIGS = [
    {
        "name": "fifo",
        "description": "FIFO — original submission order, no grouping",
        "strategy": "sync",
        "buffer_limit": 1,
        "enable_model_grouping": False,
    },
    {
        "name": "random",
        "description": "Random — semaphore concurrency, no grouping (mixed execution order)",
        "strategy": "semaphore",
        "buffer_limit": 10,
        "enable_model_grouping": False,
    },
    {
        "name": "grouping",
        "description": "Model Grouping — sort by model, semaphore execution",
        "strategy": "semaphore",
        "buffer_limit": 10,
        "enable_model_grouping": True,
    },
]


def run():
    print("=" * 70)
    print("EXP 1 — Scheduling Strategy Impact on Makespan")
    print("=" * 70)

    results = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n{'─'*50}")
        print(f"Config: {cfg['description']}")
        print(f"{'─'*50}")

        set_strategy(
            strategy=cfg["strategy"],
            buffer_limit=cfg["buffer_limit"],
            enable_model_grouping=cfg["enable_model_grouping"],
        )

        tasks = generate_workload(100)
        batch = submit_batch(tasks)

        metrics = extract_task_metrics(batch["tasks_data"])
        makespan = batch["duration"]
        throughput = batch["num_tasks"] / makespan if makespan > 0 else 0

        result = {
            "config": cfg,
            "makespan": makespan,
            "throughput": throughput,
            **metrics,
            "batch_id": batch["batch_id"],
        }
        results[name] = result
        save_result(result, OUTDIR / f"{name}.json")

        print(f"  Makespan : {makespan:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")

        # cooldown between configs
        time.sleep(10)

    # ── summary ────────────────────────────────────────────────────────────
    save_result(results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Makespan by Strategy")
    print(f"{'Config':<16} {'Makespan (s)':<14} {'Throughput':<12}")
    print("-" * 44)
    for name in ["fifo", "random", "grouping"]:
        r = results.get(name)
        if r:
            print(f"{name:<16} {r['makespan']:<14.2f} {r['throughput']:<12.2f}")

    if "fifo" in results and "grouping" in results:
        reduction = (1 - results["grouping"]["makespan"]
                     / results["fifo"]["makespan"]) * 100
        print(f"\nGrouping vs FIFO: {reduction:+.1f}% makespan reduction")
    print("=" * 70)


if __name__ == "__main__":
    run()
