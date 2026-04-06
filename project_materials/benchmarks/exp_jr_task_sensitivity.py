#!/usr/bin/env python3
"""
Experiment 4C: Johnson's Rule — Task-Count Sensitivity (W2b fix)
=================================================================
Runs all 6 permutations × 3 runs at THREE task counts per model: 5, 15, 30.

Purpose:
  At n=30 tasks/model, the 0.6B compute window (~60s) fully covers both
  8B (~5.1s SSD load) and 32B (~21.7s SSD load), making the S2 sub-ordering
  empirically invisible (0.6B→8B→32B ≈ 0.6B→32B→8B, tied within noise).

  This experiment demonstrates that JR's ordering is physically meaningful
  and regime-dependent:

  n=5 tasks/model:
    - 0.6B window ≈ 10s < 32B load ≈ 22s → all groups are S1 (a_i < b_i)
    - JR orders ascending a_i → 0.6B→8B→32B
    - 32B's load cost is NOT fully hidden → sub-ordering matters
    - Prediction: 0.6B→8B→32B measurably faster than 0.6B→32B→8B (~5s gap)

  n=15 tasks/model:
    - 0.6B window ≈ 30s ≈ 32B load → transition regime
    - JR adapts; orderings start to converge

  n=30 tasks/model:
    - 0.6B window ≈ 60s > both loads → 0.6B-first orderings converge
    - JR: 0.6B→32B→8B (S1/S2 mixed) ≈ 0.6B→8B→32B (tied, current result)

Design: 3 task counts × 6 permutations × 3 runs = 54 total runs
GPU requirement: GPUs 0,1,2,3 (TP=4 for 32B).
Runtime: ~54 × (100s cold-start + 70s avg execution) ≈ 93 min.
Results saved to: results/exp4_task_sensitivity/
"""

import sys
import itertools
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
    aggregate_runs,
    print_header,
    print_separator,
    evict_model_page_cache,
    MODEL_SMALL,
    MODEL_LARGE,
    MODEL_XLARGE,
    RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp4_task_sensitivity"

# Task counts to sweep: [5, 15, 30]
TASK_COUNTS = [5, 15, 30]

EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]
NUM_RUNS_PER_PERM = 3

ALL_PERMS = list(itertools.permutations(EXP_MODELS))  # 6 permutations

# JR-optimal ordering at each regime
# n=5:  all S1 (a_i < b_i for all) → ascending a_i → 0.6B→8B→32B
# n=15: transition
# n=30: mixed (0.6B=S1, 8B/32B=S2) → 0.6B→32B→8B (S1 asc, then S2 desc b_i)
JR_ORDER_n5  = (MODEL_SMALL, MODEL_LARGE,  MODEL_XLARGE)  # 0.6B→8B→32B
JR_ORDER_n30 = (MODEL_SMALL, MODEL_XLARGE, MODEL_LARGE)   # 0.6B→32B→8B

MODEL_LABELS = {
    MODEL_SMALL:  "0.6B",
    MODEL_LARGE:  "8B",
    MODEL_XLARGE: "32B",
}


def order_str(order):
    return " → ".join(MODEL_LABELS[m] for m in order)


def generate_workload(n_tasks_per_model: int):
    """Round-robin interleaved tasks; grouping will sort by forced order."""
    tasks = []
    total = n_tasks_per_model * len(EXP_MODELS)
    for i in range(total):
        model = EXP_MODELS[i % len(EXP_MODELS)]
        tasks.append(make_task(
            model,
            f"Explain concept {i + 1} in one sentence.",
            f"task-{i}-{MODEL_LABELS[model]}",
        ))
    return tasks


def cold_start():
    for m in EXP_MODELS:
        evict_model_page_cache(m)
    if not clear_deployments():
        print("  [WARN] Cold-start verification failed, proceeding anyway")
    if not verify_cold_state():
        print("  [WARN] System may not be fully cold")
    register_model_deployment(MODEL_XLARGE, {
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
    })


def run_perm_at_n(perm, perm_idx, n_per_model):
    """Run one permutation for NUM_RUNS_PER_PERM runs at given task count."""
    label = order_str(perm)
    num_tasks = n_per_model * len(EXP_MODELS)
    runs = []

    for run_idx in range(NUM_RUNS_PER_PERM):
        cold_start()

        set_strategy(
            strategy="semaphore",
            buffer_limit=8,
            enable_model_grouping=True,
            enable_johnsons_rule=False,
            enable_prefetch=True,
            prefetch_threshold=0.0,
            force_group_order=list(perm),
        )

        batch = submit_batch(generate_workload(n_per_model))
        makespan = batch["duration"]
        throughput = num_tasks / makespan if makespan > 0 else 0
        metrics = extract_task_metrics(batch["tasks_data"])

        run_result = {
            "run": run_idx + 1,
            "n_per_model": n_per_model,
            "perm_idx": perm_idx,
            "order": label,
            "makespan": makespan,
            "throughput": round(throughput, 2),
            "batch_id": batch["batch_id"],
            **metrics,
        }
        runs.append(run_result)
        save_result(
            run_result,
            OUTDIR / f"n{n_per_model}" / f"perm{perm_idx+1}_run{run_idx+1}.json"
        )
        print(f"    Run {run_idx+1}: {makespan:.1f}s  ({throughput:.2f} req/s)")

    agg = aggregate_runs(runs, key="makespan")
    save_result(
        {"order": label, "n_per_model": n_per_model, "aggregate": agg, "runs": runs},
        OUTDIR / f"n{n_per_model}" / f"perm{perm_idx+1}_aggregate.json",
    )
    return agg


def run_at_n(n_per_model: int):
    """Run all 6 permutations at a given task count. Returns ranked results."""
    num_tasks = n_per_model * len(EXP_MODELS)
    print_separator(f"Task count: n={n_per_model}/model  ({num_tasks} total tasks)")

    jr_order_expected = JR_ORDER_n5 if n_per_model <= 10 else JR_ORDER_n30

    results = []
    for perm_idx, perm in enumerate(ALL_PERMS):
        label = order_str(perm)
        is_jr_n5  = (perm == JR_ORDER_n5)
        is_jr_n30 = (perm == JR_ORDER_n30)
        jr_tag = ""
        if is_jr_n5:  jr_tag = " [JR@n5]"
        if is_jr_n30: jr_tag = " [JR@n30]"

        print(f"\n  Perm {perm_idx+1}/6 — {label}{jr_tag}")
        agg = run_perm_at_n(perm, perm_idx, n_per_model)
        results.append({
            "perm_idx": perm_idx,
            "order": label,
            "perm": perm,
            "is_jr_n5":  is_jr_n5,
            "is_jr_n30": is_jr_n30,
            "aggregate": agg,
        })
        print(f"  → {label}: {agg['mean']:.1f} ± {agg['std']:.1f}s")

    # Rank
    results_sorted = sorted(results, key=lambda r: r["aggregate"]["mean"])

    jr_expected_mean = next(
        r["aggregate"]["mean"] for r in results
        if r["perm"] == jr_order_expected
    )
    jr_expected_rank = next(
        i + 1 for i, r in enumerate(results_sorted)
        if r["perm"] == jr_order_expected
    )

    print(f"\n  ── n={n_per_model} Summary (JR expected: {order_str(jr_order_expected)}) ──")
    print(f"  {'Rank':<5} {'Ordering':<25} {'Mean':>8}  {'Std':>6}  {'vs JR best'}")
    for rank, r in enumerate(results_sorted, 1):
        mean = r["aggregate"]["mean"]
        std  = r["aggregate"]["std"]
        vs = f"+{(mean/jr_expected_mean-1)*100:.1f}%" if mean > jr_expected_mean else "---"
        jr_tag = ""
        if r["is_jr_n5"]:  jr_tag += " [JR@n5]"
        if r["is_jr_n30"]: jr_tag += " [JR@n30]"
        print(f"  {rank:<5} {r['order']:<25} {mean:>8.1f}  {std:>6.1f}  {vs}{jr_tag}")

    save_result({
        "n_per_model": n_per_model,
        "ranked": [
            {k: v for k, v in r.items() if k != "perm"}
            for r in results_sorted
        ],
        "jr_n5_rank":  next((i+1 for i, r in enumerate(results_sorted) if r["is_jr_n5"]), None),
        "jr_n30_rank": next((i+1 for i, r in enumerate(results_sorted) if r["is_jr_n30"]), None),
    }, OUTDIR / f"n{n_per_model}" / "summary.json")

    return results_sorted


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for n in TASK_COUNTS:
        (OUTDIR / f"n{n}").mkdir(parents=True, exist_ok=True)

    print_header("Experiment 4C: Johnson's Rule Task-Count Sensitivity (W2b)", {
        "Models":      "Qwen3-0.6B, Qwen3-8B, Qwen3-32B",
        "Task counts": "5, 15, 30 per model",
        "Design":      f"3 counts × 6 perms × {NUM_RUNS_PER_PERM} runs = "
                       f"{3 * len(ALL_PERMS) * NUM_RUNS_PER_PERM} total runs",
        "JR@n5":       f"{order_str(JR_ORDER_n5)}  (all S1: ascending a_i)",
        "JR@n30":      f"{order_str(JR_ORDER_n30)}  (mixed: 0.6B=S1, 8B/32B=S2)",
        "Hypothesis":  "At n=5, 0.6B window (10s) < 32B load (22s) → JR sub-ordering visible",
        "GPUs":        "4 (TP=4 for 32B)",
    })

    all_results = {}
    for n_per_model in TASK_COUNTS:
        ranked = run_at_n(n_per_model)
        all_results[f"n{n_per_model}"] = [
            {k: v for k, v in r.items() if k != "perm"}
            for r in ranked
        ]

    # ── Cross-task-count comparison ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("REGIME SENSITIVITY SUMMARY")
    print(f"  JR@n5 = {order_str(JR_ORDER_n5)}")
    print(f"  JR@n30 = {order_str(JR_ORDER_n30)}")
    print()
    for n in TASK_COUNTS:
        ranked = all_results[f"n{n}"]
        jr_n5_result  = next((r for r in ranked if r["is_jr_n5"]), None)
        jr_n30_result = next((r for r in ranked if r["is_jr_n30"]), None)
        best = ranked[0]
        worst = ranked[-1]
        print(f"  n={n}:")
        if jr_n5_result:
            rank_n5 = ranked.index(jr_n5_result) + 1
            print(f"    JR@n5  ({order_str(JR_ORDER_n5)}): "
                  f"{jr_n5_result['aggregate']['mean']:.1f}s  rank {rank_n5}/6")
        if jr_n30_result:
            rank_n30 = ranked.index(jr_n30_result) + 1
            print(f"    JR@n30 ({order_str(JR_ORDER_n30)}): "
                  f"{jr_n30_result['aggregate']['mean']:.1f}s  rank {rank_n30}/6")
        print(f"    Best: {best['order']} {best['aggregate']['mean']:.1f}s  |  "
              f"Worst: {worst['order']} {worst['aggregate']['mean']:.1f}s  |  "
              f"Spread: {worst['aggregate']['mean'] - best['aggregate']['mean']:.1f}s")
    print("=" * 72)

    save_result(all_results, OUTDIR / "cross_task_summary.json")


if __name__ == "__main__":
    run()
