#!/usr/bin/env python3
"""
Experiment 3 — Concurrency Mode Comparison
==========================================
Thesis §4.4: Compare sync / chunked / semaphore execution modes
with model grouping enabled for all three.

This isolates the effect of concurrency strategy: how tasks are
dispatched to the inference backend.

Modes:
  - sync:      One task at a time, wait for completion (sequential)
  - chunked:   Fire a chunk of N tasks, wait for ALL to finish, then next chunk
  - semaphore: Fire all tasks, but only N run concurrently (sliding window)

All modes have enable_model_grouping=True, so task ORDER is the same.
The only variable is HOW tasks are dispatched.

Expected: semaphore ≤ chunked < sync
  (semaphore avoids chunked's "stop-and-wait" gap between chunks)
"""

import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy, submit_batch, extract_task_metrics,
    save_result, make_task, MODEL_SMALL, MODEL_LARGE, RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp3_concurrency_mode"
NUM_TASKS = 50   # 25 per model, interleaved submission


def generate_workload(n: int = NUM_TASKS):
    """ABABAB... interleaved, but grouping will sort them into AAAA...BBBB."""
    tasks = []
    models = [MODEL_SMALL, MODEL_LARGE]  # Qwen3-0.6B vs Qwen3-8B
    for i in range(n):
        m = models[i % 2]
        tag = "A" if i % 2 == 0 else "B"
        tasks.append(make_task(m, f"Explain topic {i+1} briefly.", f"task-{i}-{tag}"))
    return tasks


CONFIGS = [
    {
        "name": "sync",
        "description": "Sync — sequential, one task at a time (grouping ON)",
        "strategy": "sync",
        "buffer_limit": 1,       # ignored by sync, but set for consistency
        "enable_model_grouping": True,
    },
    {
        "name": "chunked",
        "description": "Chunked — fire chunk of 10, wait for all, repeat (grouping ON)",
        "strategy": "chunked",
        "buffer_limit": 10,
        "enable_model_grouping": True,
    },
    {
        "name": "semaphore",
        "description": "Semaphore — fire all, max 10 concurrent (grouping ON)",
        "strategy": "semaphore",
        "buffer_limit": 10,
        "enable_model_grouping": True,
    },
]


def run():
    print("=" * 70)
    print("EXP 3 — Concurrency Mode Comparison")
    print("       (all with model grouping ON)")
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

        tasks = generate_workload()
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

        print(f"  Makespan:   {makespan:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")

        time.sleep(15)

    # ── summary ────────────────────────────────────────────────────────────
    save_result(results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Concurrency Mode (all grouping ON)")
    print(f"{'Mode':<14} {'Makespan (s)':<14} {'Throughput':<12}")
    print("-" * 42)
    for name in ["sync", "chunked", "semaphore"]:
        r = results.get(name)
        if r:
            print(f"{name:<14} {r['makespan']:<14.2f} {r['throughput']:<12.2f}")

    if "sync" in results and "semaphore" in results:
        reduction = (1 - results["semaphore"]["makespan"]
                     / results["sync"]["makespan"]) * 100
        print(f"\nSemaphore vs Sync: {reduction:+.1f}% makespan reduction")

    if "chunked" in results and "semaphore" in results:
        diff = (1 - results["semaphore"]["makespan"]
                / results["chunked"]["makespan"]) * 100
        print(f"Semaphore vs Chunked: {diff:+.1f}% makespan reduction")
    print("=" * 70)


if __name__ == "__main__":
    run()
