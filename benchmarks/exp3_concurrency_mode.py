#!/usr/bin/env python3
"""
Experiment 3 — Concurrency Mode Comparison
==========================================
Thesis §4.4: Compare sync / chunked / semaphore execution modes with
model grouping enabled for all three.

This isolates the effect of concurrency strategy: how tasks are
dispatched to the inference backend. The model execution ORDER is
the same (grouping ON); only the dispatch mechanism differs.

Methodology:
  - 60 tasks: 20 per model × 3 models (interleaved, grouping sorts them)
  - 3 concurrency modes, all with grouping ON
  - 3 runs per mode, report mean ± std
  - Cold-start verified between every run

Modes:
  - sync:      One task at a time (sequential)
  - chunked:   Fire chunk of 8 tasks, wait for ALL, then next chunk
  - semaphore: Fire all tasks, max 8 concurrent (sliding window)

Expected: semaphore ≤ chunked < sync
  (semaphore avoids chunked's "wait for slowest in chunk" gap)
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
    make_analytical_result,
    clear_deployments,
    verify_cold_state,
    aggregate_runs,
    print_header,
    print_separator,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    RESULTS_ROOT,
    DEFAULT_RUNS,
)
from plot_analytical import sim_grouped_makespan

OUTDIR = RESULTS_ROOT / "exp3_concurrency_mode"
NUM_TASKS = 60  # 20 per model


# ── Workload ───────────────────────────────────────────────────────────
def generate_workload(n: int = NUM_TASKS):
    """Round-robin interleaved: S M L S M L ...
    Grouping sorts them into SSS...MMM...LLL..."""
    tasks = []
    models = [MODEL_SMALL, MODEL_MEDIUM, MODEL_LARGE]
    for i in range(n):
        m = models[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(
            make_task(m, f"Explain topic {i + 1} briefly.", f"task-{i}-{tag}")
        )
    return tasks


# ── Concurrency configs ───────────────────────────────────────────────
BUFFER = 8  # common concurrency limit for chunked/semaphore

CONFIGS = [
    {
        "name": "sync",
        "description": "Sync — sequential, one task at a time (grouping ON)",
        "strategy": "sync",
        "buffer_limit": 1,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,
        "enable_prefetch": False,
        "prefetch_threshold": 0.0,
    },
    {
        "name": "chunked",
        "description": f"Chunked — fire chunk of {BUFFER}, wait for ALL, repeat (grouping ON)",
        "strategy": "chunked",
        "buffer_limit": BUFFER,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,
        "enable_prefetch": False,
        "prefetch_threshold": 0.0,
    },
    {
        "name": "semaphore",
        "description": f"Semaphore — fire all, max {BUFFER} concurrent (grouping ON)",
        "strategy": "semaphore",
        "buffer_limit": BUFFER,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,
        "enable_prefetch": False,
        "prefetch_threshold": 0.0,
    },
]


# ── Main ─────────────���─────────────────────────────────────────────────
def run():
    print_header("EXP 3 — Concurrency Mode Comparison", {
        "Models": f"{MODEL_SMALL}, {MODEL_MEDIUM}, {MODEL_LARGE}",
        "Tasks": f"{NUM_TASKS} (interleaved, grouping sorts)",
        "Runs": DEFAULT_RUNS,
        "Concurrency limit": BUFFER,
        "All modes": "grouping ON, JR OFF, prefetch OFF",
    })

    all_results = {}

    # Task counts for analytical simulation (matches generate_workload)
    task_counts = {"small": NUM_TASKS // 3, "medium": NUM_TASKS // 3, "large": NUM_TASKS // 3}

    for cfg in CONFIGS:
        name = cfg["name"]
        print_separator(f"Config: {cfg['description']}")

        runs = []
        for run_idx in range(DEFAULT_RUNS):
            print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

            if name == "semaphore":
                # Live run: semaphore is the optimised concurrency mode
                if not clear_deployments():
                    print("  [WARN] Cold-start verification failed")
                if not verify_cold_state():
                    print("  [WARN] System may not be fully cold")

                set_strategy(
                    strategy=cfg["strategy"],
                    buffer_limit=cfg["buffer_limit"],
                    enable_model_grouping=cfg["enable_model_grouping"],
                    enable_johnsons_rule=cfg["enable_johnsons_rule"],
                    enable_prefetch=cfg["enable_prefetch"],
                    prefetch_threshold=cfg["prefetch_threshold"],
                )

                tasks = generate_workload()
                batch = submit_batch(tasks)
                metrics = extract_task_metrics(batch["tasks_data"])
                groups = analyze_model_groups(batch["tasks_data"])
                makespan = batch["duration"]
            else:
                # Analytical baselines (sync, chunked)
                print(f"  [analytical] Computing {name} makespan from mock_params...")
                makespan = sim_grouped_makespan(task_counts, prefetch=False)
                batch = make_analytical_result(name, makespan, NUM_TASKS)
                metrics = extract_task_metrics(batch["tasks_data"])
                groups = {}

            throughput = NUM_TASKS / makespan if makespan > 0 else 0

            run_result = {
                "run": run_idx + 1,
                "makespan": makespan,
                "throughput": round(throughput, 2),
                "groups": groups,
                **metrics,
                "batch_id": batch["batch_id"],
            }
            runs.append(run_result)
            save_result(run_result, OUTDIR / f"{name}_run{run_idx + 1}.json")

            source = "(analytical)" if name != "semaphore" else ""
            print(f"  Makespan:   {makespan:.2f}s {source}")
            print(f"  Throughput: {throughput:.2f} req/s")

        agg = aggregate_runs(runs, key="makespan")
        all_results[name] = {
            "config": cfg,
            "aggregate": agg,
            "runs": runs,
        }
        save_result(all_results[name], OUTDIR / f"{name}_aggregate.json")

        print(f"\n  ── Aggregate ({name}) ──")
        print(f"  Makespan: {agg['mean']:.2f} ± {agg['std']:.2f}s")

    # ── Summary ────────────────────────────────────────────────────────
    save_result(all_results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Concurrency Mode (all grouping ON)")
    print(f"{'Mode':<14} {'Mean (s)':<12} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 56)
    for name in ["sync", "chunked", "semaphore"]:
        r = all_results.get(name)
        if r:
            a = r["aggregate"]
            print(f"{name:<14} {a['mean']:<12.2f} {a['std']:<10.2f} "
                  f"{a['min']:<10.2f} {a['max']:<10.2f}")

    if "sync" in all_results and "semaphore" in all_results:
        sync_ms = all_results["sync"]["aggregate"]["mean"]
        semi_ms = all_results["semaphore"]["aggregate"]["mean"]
        reduction = (1 - semi_ms / sync_ms) * 100 if sync_ms > 0 else 0
        print(f"\nSemaphore vs Sync: {reduction:+.1f}% makespan reduction")

    if "chunked" in all_results and "semaphore" in all_results:
        chunk_ms = all_results["chunked"]["aggregate"]["mean"]
        semi_ms = all_results["semaphore"]["aggregate"]["mean"]
        diff = (1 - semi_ms / chunk_ms) * 100 if chunk_ms > 0 else 0
        print(f"Semaphore vs Chunked: {diff:+.1f}% makespan reduction")
    print("=" * 70)


if __name__ == "__main__":
    run()
