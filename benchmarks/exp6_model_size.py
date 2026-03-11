#!/usr/bin/env python3
"""
Experiment 6 — Model Size Impact on Throughput
===============================================
Thesis §4.7: Establish throughput baselines for different model sizes.

Method:
  - 50 tasks per model (single model per run)
  - Models: Qwen3-0.6B (small), Qwen2.5-7B (medium), Qwen3-8B (large)
  - Single-model batches → no switching overhead
  - Measure: throughput (req/s) and average latency per model size

Purpose: Understand how model size affects inference throughput,
which informs the cost model for autoscaling decisions.
"""

import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy, submit_batch, extract_task_metrics,
    save_result, make_task,
    MODEL_SMALL, MODEL_MEDIUM, MODEL_LARGE, RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp6_model_size"
NUM_TASKS = 50

MODELS = [
    ("small",  MODEL_SMALL,  "0.6B"),
    ("medium", MODEL_MEDIUM, "7B"),
    ("large",  MODEL_LARGE,  "8B"),
]


def generate_single_model(model: str, n: int):
    return [make_task(model, f"Explain concept {i+1}.", f"task-{i}")
            for i in range(n)]


def run():
    print("=" * 70)
    print("EXP 6 — Model Size Impact on Scheduling Decisions")
    print("=" * 70)

    # Use semaphore + grouping (shared-aware) for all runs
    set_strategy("semaphore", buffer_limit=10, enable_model_grouping=True)

    results = {}

    for tag, model, size_label in MODELS:
        print(f"\n{'─'*50}")
        print(f"Model: {model} ({size_label} params)")
        print(f"{'─'*50}")

        tasks = generate_single_model(model, NUM_TASKS)
        batch = submit_batch(tasks)
        metrics = extract_task_metrics(batch["tasks_data"])
        makespan = batch["duration"]
        throughput = NUM_TASKS / makespan if makespan > 0 else 0

        result = {
            "model": model,
            "size_label": size_label,
            "num_tasks": NUM_TASKS,
            "makespan": makespan,
            "throughput": throughput,
            **metrics,
            "batch_id": batch["batch_id"],
        }
        results[tag] = result
        save_result(result, OUTDIR / f"{tag}.json")

        print(f"  Makespan:   {makespan:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Avg latency: {metrics['avg_latency']:.2f}s")

        time.sleep(10)

    # ── summary ────────────────────────────────────────────────────────────
    save_result(results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Model Size Impact")
    print(f"{'Model':<30} {'Size':<8} {'Makespan (s)':<14} {'Throughput':<12}")
    print("-" * 66)
    for tag, model, size_label in MODELS:
        r = results.get(tag)
        if r:
            print(f"{model:<30} {size_label:<8} "
                  f"{r['makespan']:<14.2f} {r['throughput']:<12.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run()
