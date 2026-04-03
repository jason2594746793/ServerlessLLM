#!/usr/bin/env python3
"""
Experiment 4B: Johnson's Rule vs All 6 Permutations (W2 fix)
=============================================================
Runs all 3! = 6 permutations of (0.6B, 8B, 32B) group orderings,
each 3 times, to correctly separate measurement variance from
ordering variance. This replaces the flawed "3 different orderings × 1
run each" design of the original Exp 4 random condition.

Design:
  - 6 permutations × 3 runs = 18 total runs
  - 90 tasks: 30 per model (0.6B, 8B, 32B)
  - Model grouping ON, prefetch ON, JR OFF (force_group_order controls ordering)
  - Full OS page-cache eviction + cold-start verification between every run

JR optimal ordering for 30 tasks/model regime:
  - 0.6B: a≈2s < b=15s → S1 (compute-dominant), sorted ascending a_i
  - 8B:   a≈20s > b=15s → S2 (I/O-dominant), sorted descending b_i
  - 32B:  a≈60s > b=36s → S2 (I/O-dominant), sorted descending b_i
  JR result: 0.6B → 32B → 8B  (S1 ascending, then S2 descending b_i)

GPU requirement: GPUs 0,1,2,3 (TP=4 for 32B).
Runtime: ~18 × (2min cold-start + 2.5min run) ≈ 82 minutes.
Results saved to: results/exp4_all_perms/
"""

import sys
import itertools
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

OUTDIR = RESULTS_ROOT / "exp4_all_perms"
NUM_TASKS_PER_MODEL = 30
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]
NUM_TASKS = NUM_TASKS_PER_MODEL * len(EXP_MODELS)  # 90 total
NUM_RUNS_PER_PERM = 3

ALL_PERMS = list(itertools.permutations(EXP_MODELS))  # 6 permutations

MODEL_LABELS = {
    MODEL_SMALL:  "0.6B",
    MODEL_LARGE:  "8B",
    MODEL_XLARGE: "32B",
}

# JR-optimal ordering for the 30-tasks/model regime
JR_ORDER = (MODEL_SMALL, MODEL_XLARGE, MODEL_LARGE)  # 0.6B → 32B → 8B


def order_str(order):
    return " → ".join(MODEL_LABELS[m] for m in order)


def generate_workload():
    """Round-robin interleaved tasks; grouping will sort by forced order."""
    tasks = []
    for i in range(NUM_TASKS):
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


def run_perm(perm, perm_idx):
    """Run one permutation for NUM_RUNS_PER_PERM runs. Returns aggregate."""
    label = order_str(perm)
    is_jr = (perm == JR_ORDER)
    jr_marker = "  ← JR optimal" if is_jr else ""

    print_separator(f"Perm {perm_idx + 1}/6 — {label}{jr_marker}")

    runs = []
    for run_idx in range(NUM_RUNS_PER_PERM):
        print(f"\n  ── Run {run_idx + 1}/{NUM_RUNS_PER_PERM} ──")
        cold_start()

        set_strategy(
            strategy="semaphore",
            buffer_limit=8,
            enable_model_grouping=True,
            enable_johnsons_rule=False,   # force order explicitly
            enable_prefetch=True,
            prefetch_threshold=0.0,
            force_group_order=list(perm),
        )

        batch = submit_batch(generate_workload())
        makespan = batch["duration"]
        throughput = NUM_TASKS / makespan if makespan > 0 else 0
        metrics = extract_task_metrics(batch["tasks_data"])

        run_result = {
            "run": run_idx + 1,
            "perm_idx": perm_idx,
            "order": label,
            "is_jr_optimal": is_jr,
            "makespan": makespan,
            "throughput": round(throughput, 2),
            "batch_id": batch["batch_id"],
            **metrics,
        }
        runs.append(run_result)
        perm_key = f"perm{perm_idx + 1}"
        save_result(run_result, OUTDIR / f"{perm_key}_run{run_idx + 1}.json")
        print(f"  Makespan: {makespan:.1f}s   Throughput: {throughput:.2f} req/s")

    agg = aggregate_runs(runs, key="makespan")
    save_result(
        {"order": label, "is_jr_optimal": is_jr, "aggregate": agg, "runs": runs},
        OUTDIR / f"perm{perm_idx + 1}_aggregate.json",
    )
    print(f"\n  → {label}: {agg['mean']:.1f} ± {agg['std']:.1f}s")
    return agg


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print_header("Experiment 4B: Johnson's Rule vs All 6 Permutations (W2)", {
        "Models":     "Qwen3-0.6B (S1), Qwen3-8B (S2), Qwen3-32B (S2)",
        "Tasks":      f"{NUM_TASKS} total ({NUM_TASKS_PER_MODEL}/model)",
        "Design":     f"All {len(ALL_PERMS)} permutations × {NUM_RUNS_PER_PERM} runs = "
                      f"{len(ALL_PERMS) * NUM_RUNS_PER_PERM} total runs",
        "JR optimal": order_str(JR_ORDER) + "  (S1 ascending a_i, then S2 descending b_i)",
        "All perms":  "  |  ".join(order_str(p) for p in ALL_PERMS),
        "Purpose":    "Correct W2: separate measurement variance from ordering variance",
    })

    results = []
    for perm_idx, perm in enumerate(ALL_PERMS):
        agg = run_perm(perm, perm_idx)
        results.append({
            "perm_idx": perm_idx,
            "order": order_str(perm),
            "is_jr_optimal": (perm == JR_ORDER),
            "aggregate": agg,
        })

    # ── Rank all permutations by mean makespan ──────────────────────────────
    results_sorted = sorted(results, key=lambda r: r["aggregate"]["mean"])
    jr_rank = next(
        i + 1 for i, r in enumerate(results_sorted) if r["is_jr_optimal"]
    )
    jr_mean = next(r["aggregate"]["mean"] for r in results if r["is_jr_optimal"])

    print("\n" + "=" * 68)
    print("SUMMARY — All Permutations Ranked by Mean Makespan")
    print(f"  {'Rank':<6} {'Ordering':<25} {'Mean (s)':<10} {'Std':<8} {'vs JR':<8} JR?")
    print("  " + "-" * 60)
    for rank, r in enumerate(results_sorted, 1):
        mean = r["aggregate"]["mean"]
        std  = r["aggregate"]["std"]
        vs_jr = f"+{(mean / jr_mean - 1) * 100:.1f}%" if mean > jr_mean else "---"
        marker = " ← JR" if r["is_jr_optimal"] else ""
        print(f"  {rank:<6} {r['order']:<25} {mean:<10.1f} {std:<8.1f} {vs_jr:<8}{marker}")
    print("=" * 68)
    print(f"\n  Johnson's Rule rank: {jr_rank}/6  "
          f"({'OPTIMAL' if jr_rank == 1 else f'not optimal — rank {jr_rank}'})")

    worst = results_sorted[-1]
    print(f"  Worst ordering:  {worst['order']} — {worst['aggregate']['mean']:.1f}s")
    print(f"  Ordering spread: {results_sorted[-1]['aggregate']['mean'] - jr_mean:.1f}s "
          f"({(results_sorted[-1]['aggregate']['mean'] / jr_mean - 1) * 100:.1f}%)")

    save_result({
        "ranked": results_sorted,
        "jr_rank": jr_rank,
        "jr_mean": jr_mean,
    }, OUTDIR / "summary.json")


if __name__ == "__main__":
    run()
