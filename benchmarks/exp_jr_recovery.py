#!/usr/bin/env python3
"""
Recovery script: run only orderings 32B→0.6B→8B and 32B→8B→0.6B,
then merge with existing results and print the final summary.
"""

import sys
import json
import itertools
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
    print_separator,
    MODEL_SMALL,
    MODEL_LARGE,
    MODEL_XLARGE,
    RESULTS_ROOT,
    DEFAULT_RUNS,
)
from exp_johnsons_rule_all6 import (
    short_label,
    compute_flowshop_prediction,
    generate_workload,
    FLOW_PARAMS,
    NUM_TASKS,
    NUM_TASKS_PER_MODEL,
)

OUTDIR = RESULTS_ROOT / "exp_johnsons_rule_all6"
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]

# Only run these 2 missing orderings
MISSING_ORDERINGS = [
    [MODEL_XLARGE, MODEL_SMALL, MODEL_LARGE],   # 32B → 0.6B → 8B
    [MODEL_XLARGE, MODEL_LARGE, MODEL_SMALL],   # 32B → 8B → 0.6B
]


def run():
    all_results = {}

    for perm_list in MISSING_ORDERINGS:
        labels = [short_label(m) for m in perm_list]
        name = "_".join(labels)
        flowshop_pred = compute_flowshop_prediction(perm_list)

        print_separator(f"Order: {' → '.join(labels)}  (flow-shop prediction: {flowshop_pred:.1f}s)")

        runs = []
        for run_idx in range(DEFAULT_RUNS):
            print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

            if not clear_deployments():
                print("  [WARN] Cold-start verification failed, proceeding anyway")
            if not verify_cold_state():
                print("  [WARN] System may not be fully cold")

            set_strategy(
                strategy="semaphore",
                buffer_limit=8,
                enable_model_grouping=True,
                enable_johnsons_rule=False,
                enable_prefetch=False,
                prefetch_threshold=0.0,
                force_group_order=perm_list,
            )

            tasks = generate_workload(perm_list)
            batch = submit_batch(tasks)
            metrics = extract_task_metrics(batch["tasks_data"])
            groups = analyze_model_groups(batch["tasks_data"])
            makespan = batch["duration"]
            throughput = NUM_TASKS / makespan if makespan > 0 else 0

            run_result = {
                "run": run_idx + 1,
                "order": labels,
                "order_models": perm_list,
                "makespan": makespan,
                "throughput": round(throughput, 2),
                "flowshop_prediction": flowshop_pred,
                "groups": groups,
                **metrics,
                "batch_id": batch["batch_id"],
            }
            runs.append(run_result)
            save_result(run_result, OUTDIR / f"order_{name}_run{run_idx + 1}.json")

            print(f"  Makespan:   {makespan:.2f}s  (predicted: {flowshop_pred:.1f}s)")
            print(f"  Throughput: {throughput:.2f} req/s")

        agg = aggregate_runs(runs, key="makespan")
        all_results[name] = {
            "order": labels,
            "order_models": perm_list,
            "flowshop_prediction": flowshop_pred,
            "aggregate": agg,
            "runs": runs,
        }
        save_result(all_results[name], OUTDIR / f"order_{name}_aggregate.json")

        print(f"\n  ── Aggregate ({' → '.join(labels)}) ──")
        print(f"  Makespan: {agg['mean']:.2f} ± {agg['std']:.2f}s")
        print(f"  Predicted: {flowshop_pred:.1f}s  (error: {abs(agg['mean'] - flowshop_pred) / flowshop_pred * 100:.1f}%)")

    # Load all 6 orderings and print combined summary
    print("\n\nLoading all results for combined summary...")
    combined = {}
    for p in OUTDIR.glob("order_*_aggregate.json"):
        name = p.stem.replace("order_", "").replace("_aggregate", "")
        with open(p) as f:
            combined[name] = json.load(f)

    save_result(combined, OUTDIR / "summary.json")

    sorted_results = sorted(combined.items(), key=lambda x: x[1]["aggregate"]["mean"])

    print("\n" + "=" * 75)
    print("SUMMARY — All 6 Orderings by Measured Makespan")
    print(f"{'Order':<22} {'Mean (s)':<12} {'Std':<10} {'Predicted':<12} {'Rank'}")
    print("-" * 66)
    for rank, (name, r) in enumerate(sorted_results, 1):
        a = r["aggregate"]
        order_str = " → ".join(r["order"])
        print(f"{order_str:<22} {a['mean']:<12.2f} {a['std']:<10.2f} "
              f"{r['flowshop_prediction']:<12.1f} #{rank}")

    jr_order = sorted(EXP_MODELS, key=lambda m: FLOW_PARAMS[m]["a"])
    jr_labels = [short_label(m) for m in jr_order]
    jr_name = "_".join(jr_labels)
    jr_pred = compute_flowshop_prediction(jr_order)

    print(f"\nJohnson's Rule optimal order: {' → '.join(jr_labels)}")
    print(f"  Flow-shop prediction:  {jr_pred:.1f}s")
    if jr_name in combined:
        jr_mean = combined[jr_name]["aggregate"]["mean"]
        worst_mean = sorted_results[-1][1]["aggregate"]["mean"]
        best_mean = sorted_results[0][1]["aggregate"]["mean"]
        print(f"  Measured makespan:     {jr_mean:.2f}s")
        print(f"  Speedup vs worst:      {worst_mean / jr_mean:.2f}×  ({(1 - jr_mean/worst_mean)*100:.1f}% reduction)")
        print(f"  JR matches best:       {'YES' if jr_name == sorted_results[0][0] else 'NO (see rank above)'}")
    print("=" * 75)


if __name__ == "__main__":
    run()
