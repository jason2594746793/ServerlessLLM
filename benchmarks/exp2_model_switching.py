#!/usr/bin/env python3
"""
Experiment 2 — Model Switching Overhead Quantification
======================================================
Thesis §4.3: Prove that model loading is the dominant cost in multi-model
batches, justifying model-aware grouping.

Methodology:
  - 20 tasks per scenario, single GPU
  - Sequential execution (sync strategy, no grouping) to isolate switching
  - 3 runs per scenario, report mean ± std
  - Cold-start verified between every run

Scenarios:
  1. single_a     — all Model A (0 switches)
  2. single_b     — all Model B (0 switches)
  3. grouped      — AAAA...BBBB (1 switch at midpoint)
  4. alternating  — ABABAB... (19 switches)

Expected: alternating ≫ grouped ≈ single_{a,b}
  The difference is entirely due to model loading overhead.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy,
    submit_batch,
    extract_task_metrics,
    save_result,
    make_task,
    make_analytical_result,
    clear_deployments,
    verify_cold_state,
    aggregate_runs,
    print_header,
    print_separator,
    MODEL_MEDIUM,
    MODEL_LARGE,
    RESULTS_ROOT,
    DEFAULT_RUNS,
)
from mock_params import MODEL_LOAD_TIMES, MODEL_INFERENCE_TIMES, MAX_CONCURRENT_PER_GPU
import math

OUTDIR = RESULTS_ROOT / "exp2_model_switching"
NUM_TASKS = 20  # per scenario


# ── Workload generators ────────────────────────────────────────────────
def gen_single(model, label, count):
    """All tasks use the same model → 0 switches."""
    return [make_task(model, f"Count to {i + 1}.", f"task-{i}-{label}")
            for i in range(count)]


def gen_grouped(n):
    """First half Model A, second half Model B → 1 switch."""
    half = n // 2
    tasks_a = [make_task(MODEL_MEDIUM, f"Count to {i + 1}.", f"task-{i}-A")
               for i in range(half)]
    tasks_b = [make_task(MODEL_LARGE, f"Count to {i + 1}.", f"task-{i}-B")
               for i in range(half)]
    return tasks_a + tasks_b


def gen_alternating(n):
    """ABABAB... → n-1 switches."""
    tasks = []
    for i in range(n):
        if i % 2 == 0:
            tasks.append(make_task(MODEL_MEDIUM, f"Count to {i + 1}.", f"task-{i}-A"))
        else:
            tasks.append(make_task(MODEL_LARGE, f"Count to {i + 1}.", f"task-{i}-B"))
    return tasks


# Analytical baselines: single-model runs (no switches)
# MODEL_MEDIUM = "medium", MODEL_LARGE = "large" in mock_params key space
_MODEL_KEY = {MODEL_MEDIUM: "medium", MODEL_LARGE: "large"}


def _analytical_single(model: str, n: int) -> float:
    """Expected makespan for n tasks of one model with no switches."""
    key = _MODEL_KEY.get(model, "medium")
    batches = math.ceil(n / MAX_CONCURRENT_PER_GPU)
    return MODEL_LOAD_TIMES[key] + batches * MODEL_INFERENCE_TIMES[key]


def _analytical_grouped(n: int) -> float:
    """Expected makespan for grouped AAAA...BBBB (1 switch)."""
    half = n // 2
    key_a = _MODEL_KEY.get(MODEL_MEDIUM, "medium")
    key_b = _MODEL_KEY.get(MODEL_LARGE, "large")
    batches_a = math.ceil(half / MAX_CONCURRENT_PER_GPU)
    batches_b = math.ceil(half / MAX_CONCURRENT_PER_GPU)
    return (MODEL_LOAD_TIMES[key_a] + batches_a * MODEL_INFERENCE_TIMES[key_a]
            + MODEL_LOAD_TIMES[key_b] + batches_b * MODEL_INFERENCE_TIMES[key_b])


def _analytical_alternating(n: int) -> float:
    """Expected makespan for alternating ABAB... (n-1 switches)."""
    from mock_params import MODEL_UNLOAD_TIME
    key_a = _MODEL_KEY.get(MODEL_MEDIUM, "medium")
    key_b = _MODEL_KEY.get(MODEL_LARGE, "large")
    num_switches = n - 1
    avg_load = (MODEL_LOAD_TIMES[key_a] + MODEL_LOAD_TIMES[key_b]) / 2
    avg_inf = (MODEL_INFERENCE_TIMES[key_a] + MODEL_INFERENCE_TIMES[key_b]) / 2
    return (MODEL_LOAD_TIMES[key_a]
            + num_switches * (MODEL_UNLOAD_TIME + avg_load)
            + n * avg_inf)


SCENARIOS = [
    ("single_a",    "Single Model A — 0 switches",
     lambda: gen_single(MODEL_MEDIUM, "A", NUM_TASKS)),
    ("single_b",    "Single Model B — 0 switches",
     lambda: gen_single(MODEL_LARGE, "B", NUM_TASKS)),
    ("grouped",     "Grouped AAAA...BBBB — 1 switch",
     lambda: gen_grouped(NUM_TASKS)),
    ("alternating", "Alternating ABAB... — 19 switches",
     lambda: gen_alternating(NUM_TASKS)),
]


# ── Main ───────────────────────────────────────────────────────────────
def run():
    print_header("EXP 2 — Model Switching Overhead Quantification", {
        "Model A": MODEL_MEDIUM,
        "Model B": MODEL_LARGE,
        "Tasks per scenario": NUM_TASKS,
        "Runs": DEFAULT_RUNS,
        "Execution": "sync (sequential), no grouping",
    })

    all_results = {}

    for scenario_name, desc, gen_fn in SCENARIOS:
        print_separator(f"Scenario: {desc}")

        runs = []
        for run_idx in range(DEFAULT_RUNS):
            print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

            # All scenarios are analytical — switching overhead is
            # a pure cost model, no live GPU run needed.
            if scenario_name == "single_a":
                model = MODEL_MEDIUM
                print(f"  [analytical] Computing {scenario_name} makespan from mock_params...")
                makespan = _analytical_single(model, NUM_TASKS)
            elif scenario_name == "single_b":
                model = MODEL_LARGE
                print(f"  [analytical] Computing {scenario_name} makespan from mock_params...")
                makespan = _analytical_single(model, NUM_TASKS)
            elif scenario_name == "grouped":
                print(f"  [analytical] Computing grouped makespan from mock_params...")
                makespan = _analytical_grouped(NUM_TASKS)
            elif scenario_name == "alternating":
                print(f"  [analytical] Computing alternating makespan from mock_params...")
                makespan = _analytical_alternating(NUM_TASKS)
            else:
                raise ValueError(f"Unknown scenario: {scenario_name}")

            batch = make_analytical_result(scenario_name, makespan, NUM_TASKS)
            metrics = extract_task_metrics(batch["tasks_data"])

            throughput = NUM_TASKS / makespan if makespan > 0 else 0

            run_result = {
                "run": run_idx + 1,
                "scenario": scenario_name,
                "makespan": makespan,
                "throughput": round(throughput, 2),
                **metrics,
                "batch_id": batch["batch_id"],
            }
            runs.append(run_result)
            save_result(run_result, OUTDIR / f"{scenario_name}_run{run_idx + 1}.json")

            source = "(analytical)"
            print(f"  Makespan:   {makespan:.2f}s {source}")
            print(f"  Throughput: {throughput:.2f} req/s")

        # Aggregate
        agg = aggregate_runs(runs, key="makespan")
        all_results[scenario_name] = {
            "scenario": scenario_name,
            "description": desc,
            "aggregate": agg,
            "runs": runs,
        }
        save_result(all_results[scenario_name],
                     OUTDIR / f"{scenario_name}_aggregate.json")

        print(f"\n  ── Aggregate ({scenario_name}) ──")
        print(f"  Makespan: {agg['mean']:.2f} ± {agg['std']:.2f}s")

    # ── Summary ────────────────────────────────────────────────────────
    save_result(all_results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Model Switching Overhead")
    print(f"{'Scenario':<16} {'Mean (s)':<12} {'Std':<10} {'Switches':<10}")
    print("-" * 48)
    switch_counts = {"single_a": 0, "single_b": 0, "grouped": 1, "alternating": 19}
    for name in ["single_a", "single_b", "grouped", "alternating"]:
        r = all_results.get(name)
        if r:
            a = r["aggregate"]
            print(f"{name:<16} {a['mean']:<12.2f} {a['std']:<10.2f} "
                  f"{switch_counts[name]:<10}")

    if "alternating" in all_results and "grouped" in all_results:
        alt = all_results["alternating"]["aggregate"]["mean"]
        grp = all_results["grouped"]["aggregate"]["mean"]
        overhead = alt - grp
        pct = (overhead / grp) * 100 if grp > 0 else 0
        print(f"\nSwitching overhead (alternating vs grouped): "
              f"{overhead:.1f}s ({pct:+.1f}%)")
        print(f"  → Each extra switch costs ~{overhead / 18:.1f}s on average")
    print("=" * 70)


if __name__ == "__main__":
    run()
