#!/usr/bin/env python3
"""
Prefetch Ablation — 2-Model Comparison (fast run)
==================================================
2 models: Qwen3-0.6B (a=0.4s SSD load) and Qwen3-8B (a=5.1s SSD load)
500 tasks per model, interleaved → grouped to 0.6B → 8B (JR-optimal).

Flow-shop regime at 500 tasks:
  0.6B: b = 500 × 0.013s = 6.5s  >  a_8B = 5.1s  → prefetch fully covers 8B load
  Prefetch savings = 5.1s per run; transition gap drops from ~(init+5.1)s to ~init s.

Both configs use force_group_order=[0.6B, 8B] to guarantee single-lane sequential
execution (prevents multi-GPU scheduler from running models in parallel).

Runtime estimate: ~2.5 min/run × 6 runs = ~20 min total.
"""

import sys
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
    aggregate_runs,
    print_header,
    print_separator,
    evict_models_page_cache,
    MODEL_SMALL,
    MODEL_LARGE,
    RESULTS_ROOT,
    DEFAULT_RUNS,
)

OUTDIR = RESULTS_ROOT / "exp_prefetch_2model"
NUM_TASKS_PER_MODEL = 500
MODELS = [MODEL_SMALL, MODEL_LARGE]   # 0.6B, 8B  (single-GPU, no TP=4 needed)
NUM_TASKS = NUM_TASKS_PER_MODEL * len(MODELS)   # 1000

# JR-optimal order: 0.6B → 8B  (a_0.6B=0.4s < a_8B=5.1s, both in S1 at 500 tasks)
FORCE_ORDER = [MODEL_SMALL, MODEL_LARGE]

CONFIGS = [
    {
        "name": "no_prefetch",
        "description": "Baseline — 0.6B→8B forced order, prefetch OFF (cold switch)",
        "enable_prefetch": False,
    },
    {
        "name": "prefetch_eager",
        "description": "Prefetch ON — 8B checkpoint pre-loaded to CPU during 0.6B compute",
        "enable_prefetch": True,
    },
]


def generate_workload():
    tasks = []
    for i in range(NUM_TASKS):
        model = MODELS[i % len(MODELS)]
        tag = "S" if model == MODEL_SMALL else "M"
        tasks.append(make_task(model, f"Explain concept {i + 1} in one sentence.", f"task-{i}-{tag}"))
    return tasks


def compute_transition_gaps(tasks_data):
    groups = {}
    for t in tasks_data:
        model = t.get("body", {}).get("model", "unknown")
        if model not in groups:
            groups[model] = {"starts": [], "ends": []}
        s_str = t.get("started_at")
        e_str = t.get("completed_at")
        if s_str and e_str and t.get("status") == "completed":
            try:
                s = datetime.fromisoformat(s_str.replace("Z", "+00:00")).timestamp()
                e = datetime.fromisoformat(e_str.replace("Z", "+00:00")).timestamp()
                groups[model]["starts"].append(s)
                groups[model]["ends"].append(e)
            except ValueError:
                pass

    sorted_models = sorted(
        [(m, min(g["starts"])) for m, g in groups.items() if g["starts"]],
        key=lambda x: x[1]
    )
    gaps = []
    for i in range(len(sorted_models) - 1):
        m1, m2 = sorted_models[i][0], sorted_models[i + 1][0]
        gap = min(groups[m2]["starts"]) - max(groups[m1]["ends"])
        gaps.append({
            "from": m1.split("/")[-1],
            "to": m2.split("/")[-1],
            "gap_seconds": round(gap, 2),
        })
    return gaps


def run():
    print_header("EXP — Prefetch Ablation (2-Model, Fast)", {
        "Models": f"{MODEL_SMALL} (0.6B, a=0.4s), {MODEL_LARGE} (8B, a=5.1s)",
        "Tasks": f"{NUM_TASKS} ({NUM_TASKS_PER_MODEL}/model); 0.6B compute (6.5s) > 8B load (5.1s)",
        "Order": "Forced: 0.6B → 8B (JR-optimal)",
        "Runs": DEFAULT_RUNS,
        "Configs": "no_prefetch  vs  prefetch_eager",
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
            evict_models_page_cache(MODELS)   # flush OS page cache so SSD load is real
            if not verify_cold_state():
                print("  [WARN] System may not be fully cold")

            set_strategy(
                strategy="semaphore",
                buffer_limit=8,
                enable_model_grouping=True,
                enable_johnsons_rule=False,      # order forced externally
                enable_prefetch=cfg["enable_prefetch"],
                prefetch_threshold=0.0,
                force_group_order=FORCE_ORDER,   # single-lane sequential, no multi-GPU scatter
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
                status = "OVERLAP (prefetch working)" if gap["gap_seconds"] < 0 else "GAP"
                print(f"  Transition {gap['from']} → {gap['to']}: {gap['gap_seconds']:+.2f}s ({status})")

        agg = aggregate_runs(runs, key="makespan")
        all_results[name] = {
            "config": cfg,
            "aggregate": agg,
            "runs": runs,
        }
        save_result(all_results[name], OUTDIR / f"{name}_aggregate.json")

        print(f"\n  ── Aggregate ({name}) ──")
        print(f"  Makespan: {agg['mean']:.2f} ± {agg['std']:.2f}s")

    # ── Summary ────────────────────────────────────────────────────────────
    save_result(all_results, OUTDIR / "summary.json")

    print("\n" + "=" * 65)
    print("SUMMARY — Prefetch Effect (2-Model: 0.6B → 8B)")
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
            print(f"  Expected saving = 8B SSD load time = 5.1s (hidden during 0.6B compute)")

        # Transition gap comparison
        print("\nTransition gap (0.6B → 8B):")
        for name in ["no_prefetch", "prefetch_eager"]:
            r = all_results.get(name)
            if r and r["runs"]:
                gaps = [run.get("transition_gaps", [{}])[0].get("gap_seconds", 0)
                        for run in r["runs"] if run.get("transition_gaps")]
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    print(f"  {name:<18}: avg gap = {avg_gap:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    run()
