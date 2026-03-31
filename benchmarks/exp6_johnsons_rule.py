#!/usr/bin/env python3
"""
Experiment 6 — Johnson's Rule Ablation
=======================================
Thesis §4.7: Isolate the effect of Johnson's Rule group ordering on
makespan. Uses an uneven task distribution to amplify ordering effects.

Methodology:
  - 90 tasks with UNEVEN distribution: 50× small, 30× medium, 10× large
    (uneven distribution creates groups of very different sizes,
     making ordering decisions matter more)
  - 2 configurations: grouping ON with JR OFF vs JR ON
  - Both use semaphore + prefetch ON (isolate ONLY the JR effect)
  - 3 runs per configuration, report mean ± std
  - Also computes per-model-group timing + transition analysis

Johnson's Rule context:
  - Machine A = I/O (SSD → CPU pinned memory)
  - Machine B = GPU compute (inference)
  - With NVMe at ~3 GB/s, all groups are compute-heavy (P >> I)
  - JR sorts compute-heavy groups by I ascending (smallest checkpoint first)
  - For our models: 0.6B (I≈0.4s) → 7B (I≈4.8s) → 8B (I≈5.1s)
  - Without JR, alphabetical ordering may differ from optimal
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

OUTDIR = RESULTS_ROOT / "exp6_johnsons_rule"

# Uneven distribution to amplify ordering effects
TASK_COUNTS = {
    MODEL_SMALL: 50,   # small, fast to load and infer
    MODEL_MEDIUM: 30,  # medium, moderate load time
    MODEL_LARGE: 10,   # large, slow to load
}
TOTAL_TASKS = sum(TASK_COUNTS.values())  # 90


# ── Workload ───────────────────────────────────────────────────────────
def generate_workload():
    """Create interleaved tasks with uneven model mix.
    Submission order is round-robin but with uneven counts —
    grouping will sort them by model name regardless.
    """
    tasks = []
    # Build list of (model, count) and interleave
    model_queues = {m: c for m, c in TASK_COUNTS.items()}
    models = list(model_queues.keys())
    idx = 0
    while any(v > 0 for v in model_queues.values()):
        m = models[idx % len(models)]
        if model_queues[m] > 0:
            short = m.split("/")[-1][:4]
            tasks.append(
                make_task(m, f"Explain concept {len(tasks) + 1}.",
                          f"task-{len(tasks)}-{short}")
            )
            model_queues[m] -= 1
        idx += 1
    return tasks


# ── Configs ────────────────────────────────────────────────────────────
CONFIGS = [
    {
        "name": "grouping_no_jr",
        "description": "Grouping + Prefetch, Johnson's Rule OFF (alphabetical order)",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,
        "enable_prefetch": True,
        "prefetch_threshold": 0.0,
    },
    {
        "name": "grouping_with_jr",
        "description": "Grouping + Prefetch, Johnson's Rule ON (optimal ordering)",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": True,
        "enable_prefetch": True,
        "prefetch_threshold": 0.0,
    },
]


# ── Main ───────────────────────────────────────────────────────────────
def run():
    print_header("EXP 6 — Johnson's Rule Ablation", {
        "Models": f"{MODEL_SMALL} (x{TASK_COUNTS[MODEL_SMALL]}), "
                  f"{MODEL_MEDIUM} (x{TASK_COUNTS[MODEL_MEDIUM]}), "
                  f"{MODEL_LARGE} (x{TASK_COUNTS[MODEL_LARGE]})",
        "Total tasks": TOTAL_TASKS,
        "Runs": DEFAULT_RUNS,
        "Comparison": "JR OFF (alphabetical) vs JR ON (optimal)",
    })

    all_results = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        print_separator(f"Config: {cfg['description']}")

        runs = []
        for run_idx in range(DEFAULT_RUNS):
            print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

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
            throughput = batch["num_tasks"] / makespan if makespan > 0 else 0

            # Log group execution order
            ordered_groups = sorted(
                [(m, g) for m, g in groups.items() if "first_start" in g],
                key=lambda x: x[1]["first_start"]
            )
            group_order = [m for m, _ in ordered_groups]
            print(f"  Group execution order: {' → '.join(group_order)}")

            run_result = {
                "run": run_idx + 1,
                "makespan": makespan,
                "throughput": round(throughput, 2),
                "groups": groups,
                "group_order": group_order,
                **metrics,
                "batch_id": batch["batch_id"],
            }
            runs.append(run_result)
            save_result(run_result, OUTDIR / f"{name}_run{run_idx + 1}.json")

            print(f"  Makespan:   {makespan:.2f}s")
            print(f"  Throughput: {throughput:.2f} req/s")
            for m, g in ordered_groups:
                print(f"    {m}: {g['completed']} tasks, "
                      f"avg_lat={g.get('avg_latency', 0):.1f}s, "
                      f"duration={g.get('duration', 0):.1f}s")

        agg = aggregate_runs(runs, key="makespan")
        all_results[name] = {
            "config": cfg,
            "aggregate": agg,
            "runs": runs,
        }
        save_result(all_results[name], OUTDIR / f"{name}_aggregate.json")

        print(f"\n  ── Aggregate ({name}) ──")
        print(f"  Makespan: {agg['mean']:.2f} ± {agg['std']:.2f}s")

    # ── Summary ─────────────────────────��──────────────────────────────
    save_result(all_results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Johnson's Rule Ablation")
    print(f"{'Config':<22} {'Mean (s)':<12} {'Std':<10} {'Values'}")
    print("-" * 65)
    for name in ["grouping_no_jr", "grouping_with_jr"]:
        r = all_results.get(name)
        if r:
            a = r["aggregate"]
            print(f"{name:<22} {a['mean']:<12.2f} {a['std']:<10.2f} {a['values']}")

    if "grouping_no_jr" in all_results and "grouping_with_jr" in all_results:
        no_jr = all_results["grouping_no_jr"]["aggregate"]["mean"]
        with_jr = all_results["grouping_with_jr"]["aggregate"]["mean"]
        if no_jr > 0:
            diff = (1 - with_jr / no_jr) * 100
            saved = no_jr - with_jr
            print(f"\nJohnson's Rule effect:")
            print(f"  Makespan change: {diff:+.1f}% ({saved:+.1f}s)")
            if abs(diff) < 3:
                print(f"  Note: With NVMe (~3 GB/s), all groups are compute-heavy")
                print(f"  (P >> I), so JR degenerates to 'smallest checkpoint first'.")
                print(f"  The benefit is primarily in initial load ordering.")
            else:
                print(f"  JR optimised the group execution order for better I/O overlap.")
    print("=" * 70)


if __name__ == "__main__":
    run()
