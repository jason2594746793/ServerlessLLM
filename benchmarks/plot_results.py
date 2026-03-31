#!/usr/bin/env python3
"""
Dissertation-quality graph generator for all experiments.

Reads JSON results from results/ and generates publication-quality
figures (PDF + PNG) using matplotlib. IEEE/ACM conference style.

All plots show mean +/- std error bars from multi-run experiments.

Usage:
  python benchmarks/plot_results.py              # all figures
  python benchmarks/plot_results.py --exp 1 8    # specific experiments
"""

import json
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style: IEEE/ACM conference ──────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent.parent / "figures"

# Grayscale-friendly color palette with distinct patterns
COLORS = ["#2c3e50", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad", "#3498db"]
HATCHES = ["", "//", "\\\\", "xx", "..", "++"]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    print(f"  Saved: figures/{name}.pdf + .png")
    plt.close(fig)


def _get_agg(entry, key="makespan"):
    """Extract mean and std from an experiment entry.

    Supports two formats:
      - New aggregate format: entry["aggregate"]["mean"] / ["std"]
      - Scalability format: entry["makespan_agg"]["mean"] / ["std"]
      - Direct format: entry[key] (no std available)
    """
    if "aggregate" in entry:
        return entry["aggregate"]["mean"], entry["aggregate"]["std"]
    agg_key = f"{key}_agg"
    if agg_key in entry:
        return entry[agg_key]["mean"], entry[agg_key]["std"]
    return entry.get(key, 0), 0


# ════════════════════════════════════════════════════════════════════════
# Exp 1: Strategy Makespan Comparison (5 strategies)
# ════════════════════════════════════════════════════════════════════════

def plot_exp1():
    path = RESULTS / "exp1_strategy_makespan" / "summary.json"
    if not path.exists():
        print("  [skip] exp1 results not found")
        return

    data = load_json(path)
    names = ["fifo_sync", "fifo_concurrent", "grouping", "grouping_jr", "full"]
    labels = ["FIFO\n(sync)", "FIFO\n(concurrent)", "Grouping", "Grouping\n+ JR", "Full\n(Shared-Aware)"]

    present = [(n, l) for n, l in zip(names, labels) if n in data]
    if not present:
        print("  [skip] exp1 no valid configs")
        return

    ns, ls = zip(*present)
    means = []
    stds = []
    for n in ns:
        m, s = _get_agg(data[n])
        means.append(m)
        stds.append(s)

    throughputs_mean = []
    throughputs_std = []
    for n in ns:
        r = data[n]
        runs = r.get("runs", [])
        tputs = [run.get("throughput", 0) for run in runs]
        if tputs:
            throughputs_mean.append(np.mean(tputs))
            throughputs_std.append(np.std(tputs))
        else:
            throughputs_mean.append(0)
            throughputs_std.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    x = np.arange(len(ns))
    bars = ax1.bar(x, means, yerr=stds, capsize=4, color=COLORS[:len(ns)],
                   edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.2})
    for bar, h in zip(bars, HATCHES[:len(ns)]):
        bar.set_hatch(h)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ls)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan by Strategy")

    for bar, m, s in zip(bars, means, stds):
        label = f"{m:.1f}s" if s == 0 else f"{m:.1f}\u00b1{s:.1f}s"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1,
                 label, ha="center", va="bottom", fontsize=7)

    bars2 = ax2.bar(x, throughputs_mean, yerr=throughputs_std, capsize=4,
                    color=COLORS[:len(ns)], edgecolor="black", linewidth=0.5,
                    error_kw={"linewidth": 1.2})
    for bar, h in zip(bars2, HATCHES[:len(ns)]):
        bar.set_hatch(h)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ls)
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_title("(b) Throughput by Strategy")

    # Mark calibrated analytical with asterisk
    if "fifo_sync" in data:
        source = data["fifo_sync"].get("config", {}).get("source", "")
        if "calibrated" in source:
            ax1.annotate("*calibrated", xy=(0, means[0] + stds[0] + 5),
                         fontsize=7, ha="center", style="italic", color="gray")

    fig.suptitle("Exp 1: Scheduling Strategy Impact on Makespan", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp1_strategy_makespan")


# ════════════════════════════════════════════════════════════════════════
# Exp 2: Model Switching Overhead
# ════════════════════════════════════════════════════════════════════════

def plot_exp2():
    path = RESULTS / "exp2_model_switching" / "summary.json"
    if not path.exists():
        print("  [skip] exp2 results not found")
        return

    data = load_json(path)
    names = ["single_a", "single_b", "grouped", "alternating"]
    labels = ["Single A\n(0 switches)", "Single B\n(0 switches)",
              "Grouped\n(1 switch)", "Alternating\n(N switches)"]

    present = [(n, l) for n, l in zip(names, labels) if n in data]
    if not present:
        print("  [skip] exp2 no valid scenarios")
        return

    ns, ls = zip(*present)
    means = []
    stds = []
    for n in ns:
        m, s = _get_agg(data[n])
        means.append(m)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(ns))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=COLORS[:len(ns)],
                  edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1.2})
    for bar, h in zip(bars, HATCHES[:len(ns)]):
        bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(ls)
    ax.set_ylabel("Makespan (s)")
    ax.set_title("Exp 2: Model Switching Overhead")

    for bar, m, s in zip(bars, means, stds):
        label = f"{m:.1f}s" if s == 0 else f"{m:.1f}\u00b1{s:.1f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1,
                label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_fig(fig, "exp2_model_switching")


# ════════════════════════════════════════════════════════════════════════
# Exp 3: Concurrency Mode Comparison
# ════════════════════════════════════════════════════════════════════════

def plot_exp3():
    path = RESULTS / "exp3_concurrency_mode" / "summary.json"
    if not path.exists():
        print("  [skip] exp3 results not found")
        return

    data = load_json(path)
    names = ["sync", "chunked", "semaphore"]
    labels = ["Sync\n(Sequential)", "Chunked", "Semaphore\n(Concurrent)"]

    present = [(n, l) for n, l in zip(names, labels) if n in data]
    if not present:
        print("  [skip] exp3 no valid configs")
        return

    ns, ls = zip(*present)
    ms_means, ms_stds = [], []
    tp_means, tp_stds = [], []
    for n in ns:
        m, s = _get_agg(data[n])
        ms_means.append(m)
        ms_stds.append(s)
        runs = data[n].get("runs", [])
        tputs = [r.get("throughput", 0) for r in runs]
        tp_means.append(np.mean(tputs) if tputs else 0)
        tp_stds.append(np.std(tputs) if tputs else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    x = np.arange(len(ns))

    bars = ax1.bar(x, ms_means, yerr=ms_stds, capsize=4,
                   color=COLORS[:len(ns)], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, HATCHES[:len(ns)]):
        bar.set_hatch(h)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ls)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan")

    bars2 = ax2.bar(x, tp_means, yerr=tp_stds, capsize=4,
                    color=COLORS[:len(ns)], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars2, HATCHES[:len(ns)]):
        bar.set_hatch(h)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ls)
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_title("(b) Throughput")

    fig.suptitle("Exp 3: Concurrency Mode Comparison", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp3_concurrency_mode")


# ════════════════════════════════════════════════════════════════════════
# Exp 4: Checkpoint Prefetch Effect
# ════════════════════════════════════════════════════════════════════════

def plot_exp4():
    path = RESULTS / "exp4_prefetch" / "summary.json"
    if not path.exists():
        print("  [skip] exp4 results not found")
        return

    data = load_json(path)
    names = ["no_prefetch", "prefetch_eager"]
    labels = ["No Prefetch\n(Cold Switch)", "Prefetch Eager\n(SSD\u2192CPU Overlap)"]

    present = [(n, l) for n, l in zip(names, labels) if n in data]
    if not present:
        print("  [skip] exp4 no valid configs")
        return

    ns, ls = zip(*present)
    ms_means, ms_stds = [], []
    tp_means, tp_stds = [], []
    for n in ns:
        m, s = _get_agg(data[n])
        ms_means.append(m)
        ms_stds.append(s)
        runs = data[n].get("runs", [])
        tputs = [r.get("throughput", 0) for r in runs]
        tp_means.append(np.mean(tputs) if tputs else 0)
        tp_stds.append(np.std(tputs) if tputs else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    bars = ax1.bar(ls, ms_means, yerr=ms_stds, capsize=5,
                   color=COLORS[:len(ns)], edgecolor="black", linewidth=0.5,
                   error_kw={"linewidth": 1.2})
    for bar, h in zip(bars, HATCHES[:len(ns)]):
        bar.set_hatch(h)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Total Makespan")
    for bar, m, s in zip(bars, ms_means, ms_stds):
        label = f"{m:.1f}s" if s == 0 else f"{m:.1f}\u00b1{s:.1f}s"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                 label, ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(ls, tp_means, yerr=tp_stds, capsize=5,
                    color=COLORS[:len(ns)], edgecolor="black", linewidth=0.5,
                    error_kw={"linewidth": 1.2})
    for bar, h in zip(bars2, HATCHES[:len(ns)]):
        bar.set_hatch(h)
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_title("(b) Throughput")

    # Show reduction percentage
    if len(ms_means) == 2 and ms_means[0] > 0:
        reduction = (1 - ms_means[1] / ms_means[0]) * 100
        ax1.annotate(f"{reduction:+.1f}%", xy=(1, ms_means[1]),
                     xytext=(0.5, (ms_means[0] + ms_means[1]) / 2),
                     fontsize=10, fontweight="bold", color=COLORS[2],
                     arrowprops=dict(arrowstyle="->", color=COLORS[2]),
                     ha="center")

    fig.suptitle("Exp 4: Checkpoint Prefetch Effect", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp4_prefetch")


# ════════════════════════════════════════════════════════════════════════
# Exp 5a: Standard Workload (500 tasks)
# ════════════════════════════════════════════════════════════════════════

def plot_exp5_standard():
    path = RESULTS / "exp5_standard_workload" / "summary.json"
    if not path.exists():
        print("  [skip] exp5_standard results not found")
        return

    data = load_json(path)
    configs = [c for c in ["no_prefetch", "prefetch_eager"] if c in data]
    labels = {"no_prefetch": "No Prefetch", "prefetch_eager": "Prefetch (Eager)"}

    makespans = [data[c]["makespan"] for c in configs]
    throughputs = [data[c]["throughput"] for c in configs]
    trans_times = [data[c].get("total_transition_time", 0) for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for i, (vals, ylabel, title) in enumerate([
        (makespans, "Makespan (s)", "(a) Makespan"),
        (throughputs, "Throughput (tasks/s)", "(b) Throughput"),
        (trans_times, "Transition Time (s)", "(c) Transition Overhead"),
    ]):
        bars = axes[i].bar([labels[c] for c in configs], vals,
                           color=COLORS[:len(configs)], edgecolor="black", linewidth=0.5)
        for bar, h in zip(bars, HATCHES[:len(configs)]):
            bar.set_hatch(h)
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(title)

    fig.suptitle("Exp 5: Standard Workload (500 Tasks, 3 Models)", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp5_standard_workload")


# ════════════════════════════════════════════════════════════════════════
# Exp 5b: Scalability (batch size) — line chart with error bars
# ════════════════════════════════════════════════════════════════════════

def plot_exp5_scalability():
    path = RESULTS / "exp5_scalability" / "summary.json"
    if not path.exists():
        print("  [skip] exp5_scalability results not found")
        return

    data = load_json(path)
    batch_sizes = [10, 30, 60, 120, 200]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    strategy_styles = {
        "fifo_sync": {"label": "FIFO (sync, calibrated)", "marker": "^",
                      "linestyle": "--", "color": COLORS[0]},
        "grouping": {"label": "Grouping Only", "marker": "s",
                     "linestyle": "-", "color": COLORS[1]},
        "shared_aware": {"label": "Shared-Aware", "marker": "D",
                         "linestyle": "-", "color": COLORS[2]},
    }

    for strategy, style in strategy_styles.items():
        if strategy not in data:
            continue
        strat_data = data[strategy]
        sizes, ms_means, ms_stds = [], [], []
        tp_means, tp_stds = [], []

        for s in batch_sizes:
            # JSON keys are strings
            key = str(s)
            entry = strat_data.get(key, strat_data.get(s))
            if entry is None:
                continue
            sizes.append(s)
            m, sd = _get_agg(entry, key="makespan")
            ms_means.append(m)
            ms_stds.append(sd)
            tm, tsd = _get_agg(entry, key="throughput")
            tp_means.append(tm)
            tp_stds.append(tsd)

        if not sizes:
            continue

        ms_means = np.array(ms_means)
        ms_stds = np.array(ms_stds)
        tp_means = np.array(tp_means)
        tp_stds = np.array(tp_stds)

        ax1.errorbar(sizes, ms_means, yerr=ms_stds, fmt=f"-{style['marker']}",
                     color=style["color"], linestyle=style["linestyle"],
                     label=style["label"], linewidth=1.5, markersize=5,
                     capsize=3, capthick=1)
        ax1.fill_between(sizes, ms_means - ms_stds, ms_means + ms_stds,
                         alpha=0.1, color=style["color"])

        ax2.errorbar(sizes, tp_means, yerr=tp_stds, fmt=f"-{style['marker']}",
                     color=style["color"], linestyle=style["linestyle"],
                     label=style["label"], linewidth=1.5, markersize=5,
                     capsize=3, capthick=1)
        ax2.fill_between(sizes, tp_means - tp_stds, tp_means + tp_stds,
                         alpha=0.1, color=style["color"])

    ax1.set_xlabel("Batch Size (tasks)")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan vs Batch Size")
    ax1.legend()

    ax2.set_xlabel("Batch Size (tasks)")
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_title("(b) Throughput vs Batch Size")
    ax2.legend()

    fig.suptitle("Exp 5: Scalability across Batch Sizes", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp5_scalability")


# ════════════════════════════════════════════════════════════════════════
# Exp 6a: Johnson's Rule Ablation
# ════════════════════════════════════════════════════════════════════════

def plot_exp6_jr():
    path = RESULTS / "exp6_johnsons_rule" / "summary.json"
    if not path.exists():
        print("  [skip] exp6_johnsons_rule results not found")
        return

    data = load_json(path)
    names = ["grouping_no_jr", "grouping_with_jr"]
    labels = ["Grouping\n(Alphabetical)", "Grouping\n+ Johnson's Rule"]

    present = [(n, l) for n, l in zip(names, labels) if n in data]
    if not present:
        print("  [skip] exp6_jr no valid configs")
        return

    ns, ls = zip(*present)
    ms_means, ms_stds = [], []
    tp_means, tp_stds = [], []
    for n in ns:
        m, s = _get_agg(data[n])
        ms_means.append(m)
        ms_stds.append(s)
        runs = data[n].get("runs", [])
        tputs = [r.get("throughput", 0) for r in runs]
        tp_means.append(np.mean(tputs) if tputs else 0)
        tp_stds.append(np.std(tputs) if tputs else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    bars = ax1.bar(ls, ms_means, yerr=ms_stds, capsize=5,
                   color=[COLORS[0], COLORS[2]], edgecolor="black", linewidth=0.5,
                   error_kw={"linewidth": 1.2})
    bars[0].set_hatch(HATCHES[0])
    bars[1].set_hatch(HATCHES[2])
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan")
    for bar, m, s in zip(bars, ms_means, ms_stds):
        label = f"{m:.1f}s" if s == 0 else f"{m:.1f}\u00b1{s:.1f}s"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                 label, ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(ls, tp_means, yerr=tp_stds, capsize=5,
                    color=[COLORS[0], COLORS[2]], edgecolor="black", linewidth=0.5,
                    error_kw={"linewidth": 1.2})
    bars2[0].set_hatch(HATCHES[0])
    bars2[1].set_hatch(HATCHES[2])
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_title("(b) Throughput")

    # Show reduction percentage
    if len(ms_means) == 2 and ms_means[0] > 0:
        reduction = (1 - ms_means[1] / ms_means[0]) * 100
        ax1.annotate(f"{reduction:+.1f}%", xy=(1, ms_means[1]),
                     xytext=(0.5, (ms_means[0] + ms_means[1]) / 2),
                     fontsize=10, fontweight="bold", color=COLORS[2],
                     arrowprops=dict(arrowstyle="->", color=COLORS[2]),
                     ha="center")

    fig.suptitle("Exp 6: Johnson's Rule Ablation", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp6_johnsons_rule")


# ════════════════════════════════════════════════════════════════════════
# Exp 6b: Model Size Robustness
# ════════════════════════════════════════════════════════════════════════

def plot_exp6_model_size():
    path = RESULTS / "exp6_model_size" / "summary.json"
    if not path.exists():
        print("  [skip] exp6_model_size results not found")
        return

    data = load_json(path)
    sizes = ["small", "medium", "large"]
    present = [s for s in sizes if s in data]
    if not present:
        print("  [skip] exp6_model_size no valid data")
        return

    labels = [data[s].get("size_label", s) for s in present]
    makespans = [data[s]["makespan"] for s in present]
    throughputs = [data[s]["throughput"] for s in present]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    bars = ax1.bar(labels, makespans, color=COLORS[:len(labels)], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, HATCHES[:len(labels)]):
        bar.set_hatch(h)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_xlabel("Model Size")
    ax1.set_title("(a) Makespan")

    bars2 = ax2.bar(labels, throughputs, color=COLORS[:len(labels)], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars2, HATCHES[:len(labels)]):
        bar.set_hatch(h)
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_xlabel("Model Size")
    ax2.set_title("(b) Throughput")

    fig.suptitle("Exp 8: Model Size Impact on Throughput", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp6_model_size")


# ════════════════════════════════════════════════════════════════════════
# Exp 7: Multi-GPU Scaling
# ════════════════════════════════════════════════════════════════════════

def plot_exp7_multi_gpu():
    path = RESULTS / "exp7_multi_gpu" / "summary.json"
    if not path.exists():
        print("  [skip] exp7_multi_gpu results not found")
        return

    data = load_json(path)
    gpu_counts = [1, 2, 4, 8]
    present = []
    for g in gpu_counts:
        key = str(g)
        if key in data:
            present.append((g, data[key]))

    if not present:
        print("  [skip] exp7_multi_gpu no valid data")
        return

    gpus = [p[0] for p in present]
    ms_means, ms_stds = [], []
    tp_means, tp_stds = [], []
    for _, entry in present:
        m, s = _get_agg(entry, key="makespan")
        ms_means.append(m)
        ms_stds.append(s)
        tm, ts = _get_agg(entry, key="throughput")
        tp_means.append(tm)
        tp_stds.append(ts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    x = np.arange(len(gpus))
    gpu_labels = [str(g) for g in gpus]

    bars = ax1.bar(x, ms_means, yerr=ms_stds, capsize=5,
                   color=COLORS[:len(gpus)], edgecolor="black", linewidth=0.5,
                   error_kw={"linewidth": 1.2})
    for bar, h in zip(bars, HATCHES[:len(gpus)]):
        bar.set_hatch(h)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gpu_labels)
    ax1.set_xlabel("Number of GPUs")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan")
    for bar, m, s in zip(bars, ms_means, ms_stds):
        label = f"{m:.1f}s" if s == 0 else f"{m:.1f}\u00b1{s:.1f}s"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                 label, ha="center", va="bottom", fontsize=8)

    # Speedup line on second panel
    baseline_ms = ms_means[0]
    speedups = [baseline_ms / m if m > 0 else 0 for m in ms_means]
    speedup_stds = [baseline_ms * s / (m ** 2) if m > 0 else 0
                    for m, s in zip(ms_means, ms_stds)]

    ax2.errorbar(gpus, speedups, yerr=speedup_stds, fmt="-Ds",
                 color=COLORS[2], linewidth=1.5, markersize=6, capsize=4)
    ax2.plot(gpus, gpus, "--", color="gray", alpha=0.5, label="Ideal linear")
    ax2.set_xlabel("Number of GPUs")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("(b) Speedup vs Ideal Linear")
    ax2.legend()

    fig.suptitle("Exp 9: Multi-GPU Scaling", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp7_multi_gpu")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

PLOT_MAP = {
    1: ("Exp 1: Strategy Makespan", plot_exp1),
    2: ("Exp 2: Model Switching", plot_exp2),
    3: ("Exp 3: Concurrency Mode", plot_exp3),
    4: ("Exp 4: Prefetch Effect", plot_exp4),
    "5s": ("Exp 5: Standard Workload", plot_exp5_standard),
    "5c": ("Exp 5: Scalability", plot_exp5_scalability),
    "6j": ("Exp 6: Johnson's Rule", plot_exp6_jr),
    "6m": ("Exp 6: Model Size", plot_exp6_model_size),
    7: ("Exp 7: Multi-GPU", plot_exp7_multi_gpu),
}


def main():
    parser = argparse.ArgumentParser(description="Generate dissertation figures")
    parser.add_argument("--exp", nargs="*", default=None,
                        help="Experiment numbers to plot (e.g., 1 4 5). Default: all")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating dissertation-quality figures")
    print(f"  Results: {RESULTS}")
    print(f"  Output:  {FIGURES}")
    print("=" * 60)

    if args.exp:
        selected = []
        for e in args.exp:
            e = e.strip()
            if e.isdigit():
                n = int(e)
                for key in PLOT_MAP:
                    if (isinstance(key, int) and key == n) or \
                       (isinstance(key, str) and key.startswith(str(n))):
                        selected.append(key)
            else:
                if e in PLOT_MAP:
                    selected.append(e)
    else:
        selected = list(PLOT_MAP.keys())

    for key in selected:
        name, fn = PLOT_MAP[key]
        print(f"\n{name}...")
        try:
            fn()
        except Exception as e:
            print(f"  [error] {e}")

    print(f"\nDone. Figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
