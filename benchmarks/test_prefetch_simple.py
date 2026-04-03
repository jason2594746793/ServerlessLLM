#!/usr/bin/env python3
"""
Simple Prefetch Verification Test
==================================
Two back-to-back batches on 1 GPU, 2 models (0.6B → 8B).

  - Batch A: prefetch DISABLED  (explicitly turn off)
  - Batch B: prefetch ENABLED   (default algorithm, no extra config)

Both use grouping + Johnson's Rule (both also default ON).
The only difference is whether the next model's weights are read into
the OS page cache while the current model is still computing.

Expected observable result:
  Transition gap 0.6B→8B drops by ~5–10 s when prefetch is ON
  because the 8B safetensors (~15 GB) are already warm in RAM
  by the time vLLM needs them.

Tasks: 200/model (smaller = faster; still enough compute to hide 5s I/O)
Runs:  2 per config (keeps total runtime < 15 min on 1 GPU)
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
    evict_model_page_cache,
    print_header,
    print_separator,
    aggregate_runs,
    MODEL_SMALL,
    MODEL_LARGE,
    RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "test_prefetch_simple"
NUM_TASKS_PER_MODEL = 200
MODELS = [MODEL_SMALL, MODEL_LARGE]
NUM_TASKS = NUM_TASKS_PER_MODEL * len(MODELS)
RUNS = 2


def make_workload():
    tasks = []
    for i in range(NUM_TASKS):
        model = MODELS[i % len(MODELS)]
        tasks.append(make_task(model, f"In one sentence, explain concept {i}.", f"t{i}"))
    return tasks


def run_config(name: str, enable_prefetch: bool, run_idx: int) -> dict:
    print(f"\n  ── {name} run {run_idx + 1}/{RUNS} ──")

    if not clear_deployments():
        print("  [WARN] Clear may not be complete")
    if not verify_cold_state():
        print("  [WARN] System may not be fully cold")

    for m in MODELS:
        evict_model_page_cache(m)

    # Prefetch-OFF: explicitly disable.
    # Prefetch-ON:  just call set_strategy with defaults — the algorithm
    #               automatically enables prefetch (enable_prefetch=True by default).
    set_strategy(
        strategy="semaphore",
        buffer_limit=8,
        enable_model_grouping=True,
        enable_johnsons_rule=True,
        enable_prefetch=enable_prefetch,   # False=disabled, True=default/automatic
        prefetch_threshold=0.0,
        force_group_order=MODELS,          # pin order so both configs are comparable
    )

    tasks = make_workload()
    batch = submit_batch(tasks)
    metrics = extract_task_metrics(batch["tasks_data"])
    groups = analyze_model_groups(batch["tasks_data"])
    makespan = batch["duration"]
    throughput = NUM_TASKS / makespan if makespan > 0 else 0

    # Compute transition gap: time between end of group 1 and start of group 2
    transition_gap = None
    group_names = list(groups.keys())
    if len(group_names) >= 2:
        g1 = groups.get(MODELS[0].split("/")[-1], {})
        g2 = groups.get(MODELS[1].split("/")[-1], {})
        if g1.get("last_end") and g2.get("first_start"):
            transition_gap = round(g2["first_start"] - g1["last_end"], 2)

    result = {
        "config": name,
        "run": run_idx + 1,
        "makespan": makespan,
        "throughput": round(throughput, 2),
        "transition_gap": transition_gap,
        "groups": groups,
        **metrics,
    }
    save_result(result, OUTDIR / f"{name}_run{run_idx + 1}.json")

    print(f"  Makespan:   {makespan:.1f}s")
    print(f"  Throughput: {throughput:.2f} req/s")
    if transition_gap is not None:
        print(f"  Transition gap (0.6B→8B): {transition_gap:.1f}s")
    return result


def run():
    print_header("Prefetch Verification (simple, 1 GPU)", {
        "Models": f"0.6B → 8B  ({NUM_TASKS_PER_MODEL} tasks each)",
        "Configs": "prefetch=OFF vs prefetch=ON (algorithm default)",
        "Runs": RUNS,
        "Expected": "~5–10 s reduction in transition gap when prefetch ON",
    })

    no_pf_runs, pf_runs = [], []

    # ── No-prefetch baseline ────────────────────────────────────────────────
    print_separator("Config: prefetch=OFF (baseline)")
    for i in range(RUNS):
        no_pf_runs.append(run_config("no_prefetch", enable_prefetch=False, run_idx=i))

    # ── Algorithm default (prefetch ON) ─────────────────────────────────────
    print_separator("Config: prefetch=ON  (algorithm default)")
    for i in range(RUNS):
        pf_runs.append(run_config("prefetch_on", enable_prefetch=True, run_idx=i))

    # ── Summary ────────────────────────────────────────────────────────────
    agg_no  = aggregate_runs(no_pf_runs, key="makespan")
    agg_pf  = aggregate_runs(pf_runs,   key="makespan")
    save_result({"no_prefetch": agg_no, "prefetch_on": agg_pf}, OUTDIR / "summary.json")

    def avg_gap(runs):
        gaps = [r["transition_gap"] for r in runs if r.get("transition_gap") is not None]
        return sum(gaps) / len(gaps) if gaps else float("nan")

    gap_no = avg_gap(no_pf_runs)
    gap_pf = avg_gap(pf_runs)
    improvement = (agg_no["mean"] - agg_pf["mean"]) / agg_no["mean"] * 100

    print("\n" + "=" * 60)
    print("RESULT")
    print(f"  no_prefetch : {agg_no['mean']:.1f} ± {agg_no['std']:.1f}s  (gap {gap_no:.1f}s)")
    print(f"  prefetch_on : {agg_pf['mean']:.1f} ± {agg_pf['std']:.1f}s  (gap {gap_pf:.1f}s)")
    print(f"  Improvement : {improvement:.1f}%  ({agg_no['mean'] - agg_pf['mean']:.1f}s saved)")
    print(f"  Gap reduction: {gap_no - gap_pf:.1f}s  ({(gap_no-gap_pf)/gap_no*100:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run()
