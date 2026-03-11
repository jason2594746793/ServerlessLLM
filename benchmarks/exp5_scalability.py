#!/usr/bin/env python3
"""
Experiment 5 — Makespan Scalability across Batch Sizes
======================================================
Thesis §4.6: Test how well each strategy scales as batch size grows.

Method:
  - Batch sizes: 10, 25, 50, 100, 200
  - 2 models (interleaved submission)
  - Strategies: FIFO vs Model Grouping

Expected: Grouping advantage grows with batch size.
"""

import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy, submit_batch, extract_task_metrics,
    save_result, make_task, MODEL_MEDIUM, MODEL_SMALL, RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp5_scalability"
BATCH_SIZES = [10, 25, 50, 100, 200]

STRATEGIES = [
    {
        "name": "fifo",
        "strategy": "sync",
        "buffer_limit": 1,
        "enable_model_grouping": False,
    },
    {
        "name": "grouping",
        "strategy": "semaphore",
        "buffer_limit": 10,
        "enable_model_grouping": True,
    },
]


def generate_interleaved(n: int):
    models = [MODEL_MEDIUM, MODEL_SMALL]
    tasks = []
    for i in range(n):
        m = models[i % 2]
        tag = "A" if i % 2 == 0 else "B"
        tasks.append(make_task(m, f"Explain item {i+1}.", f"task-{i}-{tag}"))
    return tasks


def run():
    print("=" * 70)
    print("EXP 5 — Makespan Scalability across Batch Sizes")
    print("=" * 70)

    results = {}

    for strat in STRATEGIES:
        sname = strat["name"]
        results[sname] = {}
        print(f"\n{'━'*60}")
        print(f"Strategy: {sname.upper()}")
        print(f"{'━'*60}")

        for n in BATCH_SIZES:
            print(f"\n  Batch size = {n}")
            set_strategy(
                strategy=strat["strategy"],
                buffer_limit=strat["buffer_limit"],
                enable_model_grouping=strat["enable_model_grouping"],
            )

            tasks = generate_interleaved(n)
            batch = submit_batch(tasks)
            metrics = extract_task_metrics(batch["tasks_data"])
            makespan = batch["duration"]

            result = {
                "strategy": sname,
                "batch_size": n,
                "makespan": makespan,
                "throughput": n / makespan if makespan > 0 else 0,
                **metrics,
                "batch_id": batch["batch_id"],
            }
            results[sname][n] = result
            save_result(result, OUTDIR / f"{sname}_{n}tasks.json")

            print(f"    Makespan: {makespan:.2f}s  |  Throughput: {result['throughput']:.2f} req/s")

            time.sleep(10)

    # ── summary ────────────────────────────────────────────────────────────
    save_result(results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Makespan by Strategy × Batch Size")
    print(f"{'Size':<8}", end="")
    for s in STRATEGIES:
        print(f"  {s['name']:<16}", end="")
    print("  Reduction")
    print("-" * 60)

    for n in BATCH_SIZES:
        row = f"{n:<8}"
        vals = {}
        for s in STRATEGIES:
            r = results.get(s["name"], {}).get(n)
            ms = r["makespan"] if r else float("nan")
            vals[s["name"]] = ms
            row += f"  {ms:<16.2f}"
        if vals.get("fifo") and vals.get("grouping"):
            red = (1 - vals["grouping"] / vals["fifo"]) * 100
            row += f"  {red:+.1f}%"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    run()
