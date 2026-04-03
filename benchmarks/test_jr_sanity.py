#!/usr/bin/env python3
"""
Johnson's Rule Sanity Check (single GPU, 3 small models)
=========================================================
Tests all 6 orderings of [0.6B, 3B, 8B] to verify JR selects the
optimal ordering.

Flow-shop analysis (a_i = SSD load time, b_i = compute time for N tasks):
  0.6B: a=0.40s, 3B: a=1.63s, 8B: a=4.34s
  With 200 tasks: all in S1 regime (b > a), JR = ascending a_i = 0.6B→3B→8B

Expected: JR ordering ranks 1st; orderings with (0.6B→8B) pair rank last.
Caveat: flow-shop spread is ~4s — on top of ~70s fixed vLLM-init overhead
        this is ~5% relative difference. Small but detectable over 1 run/ordering.

Runtime: 6 orderings × 1 run × ~4 min = ~24 min
"""

import sys
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
    evict_model_page_cache,
    print_header,
    print_separator,
    RESULTS_ROOT,
    MODEL_SMALL,
    MODEL_LARGE,
)

# Three models available on single GPU
MODEL_MED = "Qwen/Qwen2.5-3B-Instruct"   # 5.7 GB, a≈1.63s

MODELS = [MODEL_SMALL, MODEL_MED, MODEL_LARGE]   # 0.6B, 3B, 8B
LABELS = {MODEL_SMALL: "0.6B", MODEL_MED: "3B", MODEL_LARGE: "8B"}

# Flow-shop params (a=SSD load time, b=compute for 200 tasks)
FLOW = {
    MODEL_SMALL: {"a": 0.40, "b": 2.60},
    MODEL_MED:   {"a": 1.63, "b": 5.00},
    MODEL_LARGE: {"a": 4.34, "b": 10.0},
}

OUTDIR = RESULTS_ROOT / "test_jr_sanity"
NUM_TASKS_PER_MODEL = 200
RUNS_PER_ORDERING = 1   # keep total runtime ~24 min


def flowshop(order):
    """Two-machine flow-shop C_max."""
    t_a, t_b = 0.0, 0.0
    for m in order:
        t_a += FLOW[m]["a"]
        t_b = max(t_b, t_a) + FLOW[m]["b"]
    return round(t_b, 2)


def label(order):
    return "→".join(LABELS[m] for m in order)


def run_ordering(order, run_idx):
    name = label(order).replace("→", "_")
    print(f"\n  ── {label(order)}  run {run_idx+1} ──")

    if not clear_deployments():
        print("  [WARN] Clear incomplete")
    if not verify_cold_state():
        print("  [WARN] May not be cold")
    for m in MODELS:
        evict_model_page_cache(m)

    # Prefetch ON (algorithm default) — ordering only matters when prefetch
    # can partially cover SSD load. JR minimises residual un-covered load time.
    set_strategy(
        strategy="semaphore",
        buffer_limit=8,
        enable_model_grouping=True,
        enable_johnsons_rule=False,     # OFF — we force the order manually
        enable_prefetch=True,           # ON — algorithm default
        prefetch_threshold=0.0,
        force_group_order=order,
    )

    tasks = []
    for i in range(NUM_TASKS_PER_MODEL * len(MODELS)):
        m = order[i % len(order)]
        tasks.append(make_task(m, f"Explain concept {i} briefly.", f"t{i}"))

    batch = submit_batch(tasks)
    makespan = batch["duration"]
    throughput = (NUM_TASKS_PER_MODEL * len(MODELS)) / makespan

    # Compute transition gaps from group timing
    groups = analyze_model_groups(batch["tasks_data"])
    gaps = []
    for i in range(len(order) - 1):
        g1 = groups.get(LABELS[order[i]], {})
        g2 = groups.get(LABELS[order[i+1]], {})
        if g1.get("last_end") and g2.get("first_start"):
            gaps.append(round(g2["first_start"] - g1["last_end"], 2))

    result = {
        "order": [LABELS[m] for m in order],
        "makespan": makespan,
        "throughput": round(throughput, 2),
        "flowshop_prediction": flowshop(order),
        "transition_gaps": gaps,
    }
    save_result(result, OUTDIR / f"order_{name}_run{run_idx+1}.json")

    gap_str = "  ".join(f"{g:.1f}s" for g in gaps)
    print(f"  Makespan:   {makespan:.1f}s  (predicted: {flowshop(order):.1f}s)")
    print(f"  Throughput: {throughput:.2f} req/s")
    print(f"  Gaps:       {gap_str}")
    return result


def run():
    all_perms = list(itertools.permutations(MODELS))

    # Compute JR optimal order (ascending a_i, S1 regime)
    jr_order = sorted(MODELS, key=lambda m: FLOW[m]["a"])

    print_header("Johnson's Rule Sanity Check (1 GPU, 3 models, 6 orderings)", {
        "Models": "0.6B (a=0.40s)  3B (a=1.63s)  8B (a=4.34s)",
        "Tasks": f"{NUM_TASKS_PER_MODEL}/model × 3 = {NUM_TASKS_PER_MODEL*3} total",
        "JR optimal": label(jr_order),
        "Flow-shop spread": f"~3.9s (best vs worst)",
        "Note": "Differences are small; test verifies JR ranks 1st, not magnitude",
    })

    # Pre-sort permutations by flow-shop prediction so we see rankings clearly
    ranked_perms = sorted(all_perms, key=lambda p: flowshop(list(p)))
    print("\nFlow-shop predictions (theory):")
    for i, perm in enumerate(ranked_perms, 1):
        pred = flowshop(list(perm))
        marker = "  ← JR optimal" if list(perm) == jr_order else ""
        print(f"  #{i} {label(list(perm)):<22}  {pred:.2f}s{marker}")

    results = []
    for perm in all_perms:
        order = list(perm)
        print_separator(f"Order: {label(order)}  (predicted {flowshop(order):.1f}s)")
        for r in range(RUNS_PER_ORDERING):
            res = run_ordering(order, r)
            results.append(res)

    # ── Summary ─────────────────────────────────────────────────────────────
    results.sort(key=lambda x: x["makespan"])
    save_result(results, OUTDIR / "summary.json")

    print("\n" + "=" * 65)
    print("SUMMARY — All 6 Orderings (ranked by measured makespan)")
    print(f"{'Order':<22} {'Measured':>10} {'Predicted':>10} {'Rank'}")
    print("-" * 55)
    for i, r in enumerate(results, 1):
        o = "→".join(r["order"])
        marker = "  ← JR optimal" if r["order"] == [LABELS[m] for m in jr_order] else ""
        print(f"{o:<22} {r['makespan']:>9.1f}s {r['flowshop_prediction']:>9.1f}s  #{i}{marker}")

    jr_label = label(jr_order)
    jr_result = next((r for r in results if "→".join(r["order"]) == jr_label), None)
    worst = results[-1]
    if jr_result:
        diff = worst["makespan"] - jr_result["makespan"]
        rank = results.index(jr_result) + 1
        print(f"\nJR order ({jr_label}): rank #{rank}/6")
        print(f"JR vs worst: {diff:+.1f}s  ({diff/worst['makespan']*100:.1f}% faster)")
    print("=" * 65)


if __name__ == "__main__":
    run()
