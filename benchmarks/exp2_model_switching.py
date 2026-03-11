#!/usr/bin/env python3
"""
Experiment 2 — Model Switching Overhead Quantification
======================================================
Thesis §4.3: Prove model loading is the dominant cost in multi-model
batches, justifying the need for model-aware grouping.

Scenarios (20 tasks each, sync/sequential, no grouping):
  1. single_a  — all Model A  (0 switches)
  2. single_b  — all Model B  (0 switches)
  3. grouped   — AAAA…BBBB    (1 switch)
  4. alternating — ABABABAB…  (up to 19 switches)

Expected: alternating ≫ grouped ≈ single_{a,b}
"""

import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy, submit_batch, extract_task_metrics,
    save_result, make_task, MODEL_MEDIUM, MODEL_SMALL, RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp2_model_switching"
NUM_PAIRS = 10  # → 20 tasks total for each scenario


def gen_single(model, label, count):
    return [make_task(model, f"Count to {i+1}.", f"task-{i}-{label}")
            for i in range(count)]


def gen_grouped(n_pairs):
    tasks_a = [make_task(MODEL_MEDIUM, f"Count to {i+1}.", f"task-{i}-A")
               for i in range(n_pairs)]
    tasks_b = [make_task(MODEL_SMALL, f"Count to {i+1}.", f"task-{i}-B")
               for i in range(n_pairs)]
    return tasks_a + tasks_b


def gen_alternating(n_pairs):
    tasks = []
    for i in range(n_pairs):
        tasks.append(make_task(MODEL_MEDIUM, f"Count to {i+1}.", f"task-{i}-A"))
        tasks.append(make_task(MODEL_SMALL,  f"Count to {i+1}.", f"task-{i}-B"))
    return tasks


SCENARIOS = [
    ("single_a",    "Single Model A (0 switches)",       lambda: gen_single(MODEL_MEDIUM, "A", NUM_PAIRS * 2)),
    ("single_b",    "Single Model B (0 switches)",       lambda: gen_single(MODEL_SMALL,  "B", NUM_PAIRS * 2)),
    ("grouped",     "Grouped AAAA…BBBB (1 switch)",      lambda: gen_grouped(NUM_PAIRS)),
    ("alternating", "Alternating ABAB… (19 switches)",   lambda: gen_alternating(NUM_PAIRS)),
]


def run():
    print("=" * 70)
    print("EXP 2 — Model Switching Overhead Quantification")
    print("=" * 70)

    results = {}

    for name, desc, gen_fn in SCENARIOS:
        print(f"\n{'─'*50}")
        print(f"Scenario: {desc}")
        print(f"{'─'*50}")

        # Force sequential execution, no grouping → scheduler preserves order
        set_strategy("sync", buffer_limit=1, enable_model_grouping=False)

        tasks = gen_fn()
        batch = submit_batch(tasks)
        metrics = extract_task_metrics(batch["tasks_data"])

        makespan = batch["duration"]
        throughput = batch["num_tasks"] / makespan if makespan > 0 else 0

        # Estimate loading vs inference from per-task data
        # Inference is ~0.3-1s per task; the rest is model loading wait
        INFERENCE_ESTIMATE = 0.5  # seconds — based on Exp 4 measurements
        total_inference_est = len(tasks) * INFERENCE_ESTIMATE
        loading_est = makespan - total_inference_est

        result = {
            "scenario": name,
            "description": desc,
            "num_tasks": len(tasks),
            "makespan": makespan,
            "throughput": throughput,
            "loading_time_est": max(0, loading_est),
            "inference_time_est": total_inference_est,
            "loading_pct": max(0, loading_est) / makespan * 100 if makespan > 0 else 0,
            **metrics,
            "batch_id": batch["batch_id"],
        }
        results[name] = result
        save_result(result, OUTDIR / f"{name}.json")

        print(f"  Makespan:  {makespan:.2f}s")
        print(f"  Loading:   ~{loading_est:.1f}s ({result['loading_pct']:.1f}%)")
        print(f"  Inference: ~{total_inference_est:.1f}s")

        time.sleep(10)

    # ── summary ────────────────────────────────────────────────────────────
    save_result(results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Model Switching Overhead")
    print(f"{'Scenario':<16} {'Makespan (s)':<14} {'Loading %':<12} {'Throughput':<12}")
    print("-" * 56)
    for name, _, _ in SCENARIOS:
        r = results.get(name)
        if r:
            print(f"{name:<16} {r['makespan']:<14.2f} "
                  f"{r['loading_pct']:<12.1f} {r['throughput']:<12.2f}")

    if "alternating" in results and "grouped" in results:
        overhead = ((results["alternating"]["makespan"]
                     - results["grouped"]["makespan"])
                    / results["grouped"]["makespan"] * 100)
        print(f"\nThrashing overhead (alternating vs grouped): {overhead:+.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    run()
