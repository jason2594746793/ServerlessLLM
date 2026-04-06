#!/usr/bin/env python3
"""
Experiment 4 — Checkpoint Prefetch Effectiveness
================================================
Thesis Exp 2: Measure the impact of checkpoint prefetching on model-switch
latency by comparing cold switches vs eager prefetch.

Methodology:
  - 90 tasks: 30 per model x 3 models (interleaved, grouping sorts)
  - 2 configurations: prefetch OFF vs prefetch ON (threshold=0.0)
  - 3 runs per configuration, report mean +/- std
  - Cold-start verified between every run
  - Per-model-group timing + transition gap measurement

Models:
  - Qwen3-0.6B (~1.2GB, a=0.4s SSD load)  -- small, fast
  - Qwen3-8B (~15.4GB, a=5.1s SSD load)   -- medium
  - Qwen2.5-32B-Instruct (~64GB, a=21.7s SSD load, TP=4) -- large

The 32B model is essential: its 21.7s SSD read time exceeds the 8B group
compute time (15s), producing a measurable residual gap that prefetch
reduces. With three small/medium models the effect is not observable.

Configurations:
  A) enable_prefetch=False: cold switch, SSD->CPU->GPU at switch time
  B) enable_prefetch=True, threshold=0.0: eager prefetch of all subsequent
     models to CPU pinned memory; only GPU transfer at switch time
"""

import sys
import time
from pathlib import Path
from datetime import datetime

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
    register_model_deployment,
    evict_model_page_cache,
    aggregate_runs,
    print_header,
    print_separator,
    MODEL_SMALL,
    MODEL_LARGE,
    MODEL_XLARGE,
    RESULTS_ROOT,
    DEFAULT_RUNS,
)
OUTDIR = RESULTS_ROOT / "exp4_prefetch"
NUM_TASKS_PER_MODEL = 30
NUM_TASKS = NUM_TASKS_PER_MODEL * 3  # 90 total: 30 per model

# Three models with diverse I/O-compute ratios (required for observable prefetch effect)
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]   # 0.6B, 8B, 32B


# ── Workload ───────────────────────────────────────────────────────────
def generate_workload(n: int = NUM_TASKS):
    """500 tasks per model, interleaved; grouping sorts to SSS...MMM...LLL..."""
    tasks = []
    for i in range(n):
        m = EXP_MODELS[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(
            make_task(m, f"Explain topic {i + 1} briefly.", f"task-{i}-{tag}")
        )
    return tasks


# ── Transition gap analysis ───────────────────────────────────────────
def compute_transition_gaps(tasks_data):
    """Compute time gaps between consecutive model groups.

    Returns list of dicts: [{from, to, gap_seconds}, ...]
    A negative gap means the next group started BEFORE the previous ended
    (overlap = prefetch working).
    """
    groups = {}
    for t in tasks_data:
        model = t.get("body", {}).get("model", "unknown")
        if model not in groups:
            groups[model] = {"starts": [], "ends": []}

        s_str = t.get("started_at")
        e_str = t.get("completed_at")
        if s_str and e_str and t.get("status") == "completed":
            try:
                s_ts = datetime.fromisoformat(
                    s_str.replace("Z", "+00:00")).timestamp()
                e_ts = datetime.fromisoformat(
                    e_str.replace("Z", "+00:00")).timestamp()
                groups[model]["starts"].append(s_ts)
                groups[model]["ends"].append(e_ts)
            except ValueError:
                pass

    # Sort groups by first start time
    sorted_models = sorted(
        [(m, min(g["starts"])) for m, g in groups.items() if g["starts"]],
        key=lambda x: x[1]
    )

    gaps = []
    for i in range(len(sorted_models) - 1):
        m1 = sorted_models[i][0]
        m2 = sorted_models[i + 1][0]
        g1_end = max(groups[m1]["ends"])
        g2_start = min(groups[m2]["starts"])
        gap = g2_start - g1_end
        gaps.append({
            "from": m1.split("/")[-1],
            "to": m2.split("/")[-1],
            "gap_seconds": round(gap, 2),
        })

    return gaps


# ── Configs ────────────────────────────────────────────────────────────
# JR-optimal order: 0.6B → 8B → 32B (all in S1: b_i > a_i at 500 tasks/model)
# force_group_order activates the single-lane sequential path in batch_scheduler.py,
# ensuring models execute strictly in this order (required because TP=4 for 32B would
# otherwise be assigned to all 4 GPUs and scheduled first by the multi-GPU path).
FORCE_ORDER = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]

CONFIGS = [
    {
        "name": "no_prefetch",
        "description": "Baseline — grouping ON, prefetch OFF, JR order forced (cold switch)",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,  # order is forced externally
        "enable_prefetch": False,
        "prefetch_threshold": 0.0,
        "force_group_order": FORCE_ORDER,
    },
    {
        "name": "prefetch_eager",
        "description": "Prefetch EAGER — same JR order forced, prefetch ON (threshold=0.0)",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,  # order is forced externally
        "enable_prefetch": True,
        "prefetch_threshold": 0.0,
        "force_group_order": FORCE_ORDER,
    },
]


# ── Main ───────────────────────────────────────────────────────────────
def run():
    print_header("EXP 4 — Checkpoint Prefetch Effectiveness (Thesis Exp 2)", {
        "Models": f"{MODEL_SMALL}, {MODEL_LARGE}, {MODEL_XLARGE}",
        "Tasks": f"{NUM_TASKS} ({NUM_TASKS_PER_MODEL} per model, interleaved)",
        "Runs": DEFAULT_RUNS,
        "Comparison": "no_prefetch vs prefetch_eager (threshold=0.0)",
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
            # Evict all model checkpoints from OS page cache for fair cold-start
            for m in EXP_MODELS:
                evict_model_page_cache(m)
            # Pre-register 32B model with explicit TP=4 so auto-creation uses correct config
            register_model_deployment(MODEL_XLARGE, {"tensor_parallel_size": 4, "gpu_memory_utilization": 0.85, "enforce_eager": True})

            set_strategy(
                strategy=cfg["strategy"],
                buffer_limit=cfg["buffer_limit"],
                enable_model_grouping=cfg["enable_model_grouping"],
                enable_johnsons_rule=cfg["enable_johnsons_rule"],
                enable_prefetch=cfg["enable_prefetch"],
                prefetch_threshold=cfg["prefetch_threshold"],
                force_group_order=cfg.get("force_group_order"),
            )

            tasks = generate_workload()
            batch = submit_batch(tasks)
            metrics = extract_task_metrics(batch["tasks_data"])
            groups = analyze_model_groups(batch["tasks_data"])
            gaps = compute_transition_gaps(batch["tasks_data"])
            makespan = batch["duration"]
            throughput = NUM_TASKS / makespan if makespan > 0 else 0

            run_result = {
                "run": run_idx + 1,
                "makespan": makespan,
                "throughput": round(throughput, 2),
                "groups": groups,
                "transition_gaps": gaps,
                **metrics,
                "batch_id": batch["batch_id"],
            }
            runs.append(run_result)
            save_result(run_result, OUTDIR / f"{name}_run{run_idx + 1}.json")

            print(f"  Makespan:   {makespan:.2f}s")
            print(f"  Throughput: {throughput:.2f} req/s")
            for gap in gaps:
                status = "OVERLAP" if gap["gap_seconds"] < 0 else "GAP"
                print(f"  Transition {gap['from']} → {gap['to']}: "
                      f"{gap['gap_seconds']:+.2f}s ({status})")

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
    print("SUMMARY — Checkpoint Prefetch Effect")
    print(f"{'Config':<18} {'Mean (s)':<12} {'Std':<10} {'Values'}")
    print("-" * 60)
    for name in ["no_prefetch", "prefetch_eager"]:
        r = all_results.get(name)
        if r:
            a = r["aggregate"]
            print(f"{name:<18} {a['mean']:<12.2f} {a['std']:<10.2f} {a['values']}")

    if "no_prefetch" in all_results and "prefetch_eager" in all_results:
        baseline = all_results["no_prefetch"]["aggregate"]["mean"]
        improved = all_results["prefetch_eager"]["aggregate"]["mean"]
        if baseline > 0:
            reduction = (1 - improved / baseline) * 100
            saved = baseline - improved
            print(f"\nPrefetch effect:")
            print(f"  Makespan reduction: {reduction:+.1f}% ({saved:.1f}s saved)")
            print(f"  Interpretation: Checkpoint prefetching overlaps SSD→CPU")
            print(f"  loading with batch execution, hiding I/O latency.")
    print("=" * 70)


if __name__ == "__main__":
    run()
