#!/usr/bin/env python3
"""
Experiment 4: Johnson's Rule vs Random vs Worst-Case Ordering
==============================================================
Thesis Exp 4: Demonstrates that Johnson's Rule ordering consistently
outperforms random group ordering and avoids worst-case outcomes.

Three conditions (3 runs each, 9 total):
  1. JR:     Scheduler applies Johnson's Rule automatically (enable_johnsons_rule=ON)
  2. Random: Different random group ordering each run (seeds fixed for reproducibility)
  3. Worst:  Known worst-case ordering (32B → 8B → 0.6B)

Methodology:
  - 90 tasks: 30 per model (Qwen3-0.6B, Qwen3-8B, Qwen3-32B)
  - Model grouping ON, prefetch ON (ordering interacts with prefetch overlap)
  - OS page-cache eviction + cold-start verification between every run
  - Mixed regime (30 tasks): 0.6B is S1 (b=15s > a=2s), 8B borderline
    (b=15s ≈ a=20s), 32B is S2 (b=36s < a=60s)
  - JR optimal for mixed regime: 0.6B → 32B → 8B
    (S1 ascending a_i, then S2 descending b_i)
"""
import sys
import random
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

OUTDIR = RESULTS_ROOT / "exp4_jr_vs_random"
NUM_TASKS_PER_MODEL = 30
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]  # 0.6B, 8B, 32B
NUM_TASKS = NUM_TASKS_PER_MODEL * len(EXP_MODELS)       # 90 total
NUM_RUNS = 3

# Worst-case ordering: largest model first forces 60s GPU idle at start
WORST_ORDER = [MODEL_XLARGE, MODEL_LARGE, MODEL_SMALL]  # 32B → 8B → 0.6B

# Fixed seeds for reproducibility; each gives a distinct shuffle that is
# neither JR-optimal (0.6B→32B→8B) nor worst-case (32B→8B→0.6B)
RANDOM_SEEDS = [1, 5, 6]  # → 8B→32B→0.6B, 0.6B→8B→32B, 8B→0.6B→32B

MODEL_LABELS = {
    MODEL_SMALL:  "0.6B",
    MODEL_LARGE:  "8B",
    MODEL_XLARGE: "32B",
}


def order_str(order):
    return " → ".join(MODEL_LABELS[m] for m in order)


def generate_workload():
    """Round-robin interleaved tasks; grouping will sort by forced/JR order."""
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


def run_single(enable_jr, force_order=None):
    """Run one batch and return makespan."""
    set_strategy(
        strategy="semaphore",
        buffer_limit=8,
        enable_model_grouping=True,
        enable_johnsons_rule=enable_jr,
        enable_prefetch=True,
        prefetch_threshold=0.0,
        force_group_order=force_order,
    )
    batch = submit_batch(generate_workload())
    makespan = batch["duration"]
    throughput = NUM_TASKS / makespan if makespan > 0 else 0
    return batch, makespan, throughput


def run_condition(name, enable_jr, force_orders=None):
    """
    Run a condition for NUM_RUNS runs.
    force_orders: list of per-run orders (None = let JR/alphabetical decide).
    """
    runs = []
    for run_idx in range(NUM_RUNS):
        force_order = force_orders[run_idx] if force_orders else None
        label = order_str(force_order) if force_order else "JR (auto)"

        print(f"\n  ── Run {run_idx + 1}/{NUM_RUNS}  [{label}] ──")
        cold_start()

        batch, makespan, throughput = run_single(enable_jr, force_order)
        metrics = extract_task_metrics(batch["tasks_data"])

        run_result = {
            "run": run_idx + 1,
            "condition": name,
            "order": label,
            "makespan": makespan,
            "throughput": round(throughput, 2),
            "batch_id": batch["batch_id"],
            **metrics,
        }
        runs.append(run_result)
        save_result(run_result, OUTDIR / f"{name}_run{run_idx + 1}.json")
        print(f"  Makespan: {makespan:.1f}s   Throughput: {throughput:.2f} req/s")

    agg = aggregate_runs(runs, key="makespan")
    save_result({"condition": name, "aggregate": agg, "runs": runs},
                OUTDIR / f"{name}_aggregate.json")
    print(f"\n  ── {name} aggregate: {agg['mean']:.1f} ± {agg['std']:.1f}s ──")
    return agg


def run():
    # Pre-compute random orderings so they are visible in the header
    random_orders = []
    for seed in RANDOM_SEEDS:
        rng = random.Random(seed)
        order = list(EXP_MODELS)
        rng.shuffle(order)
        random_orders.append(order)

    print_header("Experiment 4: Johnson's Rule vs Random vs Worst-Case Ordering", {
        "Models":      "Qwen3-0.6B (S1), Qwen3-8B (borderline), Qwen3-32B (S2)",
        "Tasks":       f"{NUM_TASKS} total ({NUM_TASKS_PER_MODEL}/model)",
        "JR optimal":  "0.6B → 32B → 8B  (mixed-regime JR)",
        "Random runs": "  |  ".join(order_str(o) for o in random_orders),
        "Worst order": order_str(WORST_ORDER),
        "Runs":        f"{NUM_RUNS} per condition  ({NUM_RUNS * 3} total)",
    })

    # ── Condition 1: Johnson's Rule ─────────────────────────────────────
    print_separator("Condition 1/3 — Johnson's Rule (scheduler decides ordering)")
    jr_agg = run_condition("jr", enable_jr=True, force_orders=None)

    # ── Condition 2: Random orderings ───────────────────────────────────
    print_separator("Condition 2/3 — Random Ordering (one shuffle per run)")
    rand_agg = run_condition("random", enable_jr=False, force_orders=random_orders)

    # ── Condition 3: Worst-case ordering ────────────────────────────────
    print_separator("Condition 3/3 — Worst-Case Ordering (32B → 8B → 0.6B)")
    worst_agg = run_condition("worst", enable_jr=False,
                              force_orders=[WORST_ORDER] * NUM_RUNS)

    # ── Summary ─────────────────────────────────────────────────────────
    jr_mean    = jr_agg["mean"]
    rand_mean  = rand_agg["mean"]
    worst_mean = worst_agg["mean"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  {'Condition':<22} {'Mean (s)':<10} {'Std':<8} vs JR")
    print("  " + "-" * 50)
    print(f"  {'JR (optimal)':<22} {jr_mean:<10.1f} {jr_agg['std']:<8.1f} ---")
    print(f"  {'Random (avg)':<22} {rand_mean:<10.1f} {rand_agg['std']:<8.1f} "
          f"+{(rand_mean / jr_mean - 1) * 100:.1f}%")
    print(f"  {'Worst case':<22} {worst_mean:<10.1f} {worst_agg['std']:<8.1f} "
          f"+{(worst_mean / jr_mean - 1) * 100:.1f}%")
    print("=" * 60)

    save_result({
        "jr":     {"aggregate": jr_agg},
        "random": {"aggregate": rand_agg, "orders": [order_str(o) for o in random_orders]},
        "worst":  {"aggregate": worst_agg, "order": order_str(WORST_ORDER)},
    }, OUTDIR / "summary.json")


if __name__ == "__main__":
    run()
