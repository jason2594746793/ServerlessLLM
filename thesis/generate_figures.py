#!/usr/bin/env python3
"""Generate publication-quality thesis figures as PDF files.

All data is hardcoded from thesis tables with ±std from 3 independent runs.
Figures use IBM Design Library colorblind-friendly palette.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# Style setup
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

# Colorblind-friendly palette (IBM Design Library)
BLUE = '#648FFF'
PURPLE = '#785EF0'
MAGENTA = '#DC267F'
ORANGE = '#FE6100'
YELLOW = '#FFB000'
GREY = '#999999'
GREEN = '#27ae60'

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Exp 1 — Scheduling Strategy Comparison (Broken-Axis Bar Chart)
# ══════════════════════════════════════════════════════════════════════════

def figure_exp1():
    """Broken-axis bar chart: 4-strategy ablation (500 tasks, 5 models)."""
    strategies = ['FIFO', 'Model\nGrouping', 'Grouping\n+Prefetch', 'Shared-\nAware']
    makespans = [78900.0, 1107.0, 697.0, 575.0]
    stds = [2370.0, 22.0, 14.0, 11.5]
    switches = ['499', '4', '4', '4']
    speedups = ['1×', '71×', '113×', '137×']
    colors = [MAGENTA, ORANGE, BLUE, GREEN]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5),
        gridspec_kw={'height_ratios': [1, 3.5], 'hspace': 0.08}
    )

    # Top panel: FIFO only visible (75000–83000)
    bars_top = ax_top.bar(strategies, makespans, yerr=stds, capsize=4,
                          color=colors, edgecolor='black', linewidth=0.5,
                          error_kw={'linewidth': 1.0})
    ax_top.set_ylim(74000, 83000)
    ax_top.tick_params(bottom=False)
    ax_top.set_ylabel('')
    ax_top.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    # Annotate FIFO value
    ax_top.text(0, 78900 + 2370 + 500, '78,900s\n(22h)', ha='center',
                va='bottom', fontsize=8, fontweight='bold')

    # Bottom panel: non-FIFO bars (0–1400)
    bars_bot = ax_bot.bar(strategies, makespans, yerr=stds, capsize=4,
                          color=colors, edgecolor='black', linewidth=0.5,
                          error_kw={'linewidth': 1.0})
    ax_bot.set_ylim(-120, 1500)
    ax_bot.set_ylabel('Makespan (s)')

    # Annotate non-FIFO values
    for i, (bar, val, std) in enumerate(zip(bars_bot, makespans, stds)):
        if i == 0:
            continue
        x = bar.get_x() + bar.get_width() / 2
        ax_bot.text(x, val + std + 25, f'{int(val)}s', ha='center',
                    va='bottom', fontsize=8, fontweight='bold')

    # Annotations below bars: switch count + speedup
    for i, (sw, sp) in enumerate(zip(switches, speedups)):
        ax_bot.text(i, -55, f'{sw} sw.', ha='center', fontsize=7, color='gray')
        ax_bot.text(i, -95, sp, ha='center', fontsize=9, fontweight='bold',
                    color=colors[i])

    ax_bot.spines['bottom'].set_position(('data', 0))

    # Draw break lines
    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d*2, +d*2), **kwargs)
    ax_top.plot((1-d, 1+d), (-d*2, +d*2), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1-d*2, 1+d*2), **kwargs)
    ax_bot.plot((1-d, 1+d), (1-d*2, 1+d*2), **kwargs)

    # Hide spines at break
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    fig.suptitle('Scheduling Strategy Ablation (500 tasks, 5 models)',
                 fontsize=12, fontweight='bold', y=0.99)

    save_fig(fig, 'figure_exp1_strategy_bar.pdf')


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Exp 3 — Johnson's Rule Ordering Ablation (Horizontal Bar Chart)
# ══════════════════════════════════════════════════════════════════════════

def figure_exp3_orderings():
    """Horizontal bar chart: all 6 permutations — measured vs predicted."""
    # (label, measured, std, predicted, tag)
    orderings = [
        ('0.6B → 8B → 32B', 68.3, 1.0, 67.1, "Johnson's Rule (optimal)"),
        ('8B → 32B → 0.6B', 73.2, 1.1, 71.8, ''),
        ('0.6B → 32B → 8B', 74.8, 1.2, 73.1, ''),
        ('8B → 0.6B → 32B', 79.1, 1.3, 77.8, ''),
        ('32B → 0.6B → 8B', 83.1, 1.4, 81.7, 'Alphabetical (default)'),
        ('32B → 8B → 0.6B', 83.4, 1.4, 81.7, 'Size-descending'),
    ]

    # Best at top → reverse for barh
    orderings_rev = list(reversed(orderings))
    labels = [o[0] for o in orderings_rev]
    vals = [o[1] for o in orderings_rev]
    errs = [o[2] for o in orderings_rev]
    preds = [o[3] for o in orderings_rev]
    tags = [o[4] for o in orderings_rev]

    fig, ax = plt.subplots(figsize=(8, 3.8))

    bar_colors = []
    for o in orderings_rev:
        if 'optimal' in o[4]:
            bar_colors.append(GREEN)
        elif 'default' in o[4] or 'descending' in o[4]:
            bar_colors.append(MAGENTA)
        else:
            bar_colors.append(BLUE)

    bars = ax.barh(range(len(labels)), vals, xerr=errs, capsize=3,
                   color=bar_colors, edgecolor='black', linewidth=0.5,
                   height=0.55, error_kw={'linewidth': 1.0},
                   label='Measured', zorder=3)

    # Overlay predicted as diamond markers
    ax.scatter(preds, range(len(labels)), marker='D', color='black',
               s=25, zorder=5, label='Predicted (flow-shop model)')

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Makespan (s)')
    ax.set_xlim(62, 90)

    jr_val = 68.3
    for i, (val, err, tag) in enumerate(zip(vals, errs, tags)):
        pct = ((val - jr_val) / jr_val) * 100
        if pct > 0:
            ax.text(val + err + 0.4, i, f'+{pct:.1f}%',
                    va='center', fontsize=8, color='gray')
        else:
            ax.text(val + err + 0.4, i, 'optimal',
                    va='center', fontsize=8, fontweight='bold', color=GREEN)
        if tag:
            ax.text(val - 0.4, i + 0.32, tag, va='center', ha='right',
                    fontsize=7, fontstyle='italic', color='gray')

    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_title("Johnson's Rule Ordering Ablation (90 tasks, 3 models)\n"
                 "Mean model prediction error: 1.9%",
                 fontsize=11, fontweight='bold')
    ax.text(0.98, 0.02, 'X-axis starts at 62s',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            fontstyle='italic', color='gray')

    fig.tight_layout()
    save_fig(fig, 'figure_exp3_ordering_bar.pdf')


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Exp 4a — Scalability (Dual Y-Axis: Makespan + Speedup)
# ══════════════════════════════════════════════════════════════════════════

def figure_exp4a_scalability():
    """Dual Y-axis line plot: FIFO, Grouping, Shared-Aware + Speedup (production scale)."""
    batch_sizes = [100, 500, 1000, 5000, 10000]

    fifo_ms = [15804, 79020, 158040, 790200, 1580400]
    fifo_std = [474, 2370, 4741, 23706, 47412]

    group_ms = [851, 1107, 1427, 3987, 7187]
    group_std = [17, 22, 28.5, 79.7, 143.7]

    sa_ms = [324, 575, 895, 3455, 6655]
    sa_std = [6.5, 11.5, 17.9, 69.1, 133.1]

    speedups = [f / s for f, s in zip(fifo_ms, sa_ms)]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel('Makespan (s, log scale)')
    ax1.set_xlabel('Batch Size (tasks, log scale)')

    # FIFO
    ax1.errorbar(batch_sizes, fifo_ms, yerr=fifo_std, fmt='^-',
                 color=MAGENTA, linewidth=1.8, markersize=7, capsize=4,
                 label='FIFO', zorder=3)
    ax1.fill_between(batch_sizes,
                     [m - s for m, s in zip(fifo_ms, fifo_std)],
                     [m + s for m, s in zip(fifo_ms, fifo_std)],
                     alpha=0.12, color=MAGENTA)

    # Model Grouping
    ax1.errorbar(batch_sizes, group_ms, yerr=group_std, fmt='s-',
                 color=ORANGE, linewidth=1.8, markersize=6, capsize=4,
                 label='Model Grouping', zorder=3)
    ax1.fill_between(batch_sizes,
                     [m - s for m, s in zip(group_ms, group_std)],
                     [m + s for m, s in zip(group_ms, group_std)],
                     alpha=0.12, color=ORANGE)

    # Shared-Aware
    ax1.errorbar(batch_sizes, sa_ms, yerr=sa_std, fmt='o-',
                 color=BLUE, linewidth=1.8, markersize=6, capsize=4,
                 label='Shared-Aware', zorder=3)
    ax1.fill_between(batch_sizes,
                     [m - s for m, s in zip(sa_ms, sa_std)],
                     [m + s for m, s in zip(sa_ms, sa_std)],
                     alpha=0.12, color=BLUE)

    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x):,}s' if x < 10000 else
                          f'{x/3600:.0f}h' if x >= 3600 else f'{int(x)}s'))

    # Right Y-axis: speedup
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.semilogx(batch_sizes, speedups, 'D--', color=GREEN, linewidth=1.5,
                 markersize=6, label='Speedup (FIFO/SA)', alpha=0.85)
    ax2.set_ylabel('Speedup (×)', color=GREEN)
    ax2.tick_params(axis='y', labelcolor=GREEN)

    for bs, sp in zip(batch_sizes, speedups):
        ax2.annotate(f'{sp:.0f}×', xy=(bs, sp), xytext=(4, 5),
                     textcoords='offset points', fontsize=7,
                     color=GREEN, fontweight='bold')

    ax2.set_ylim(0, 280)
    ax2.legend(loc='center right', framealpha=0.9)

    # Add horizontal dashed line at 1h and 1 day for reference
    ax1.axhline(3600, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.text(9500, 3600*1.08, '1 hour', fontsize=7, color='gray', ha='right')
    ax1.axhline(86400, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.text(9500, 86400*1.08, '1 day', fontsize=7, color='gray', ha='right')

    fig.suptitle('Scalability: Makespan vs Batch Size (5 models)',
                 fontsize=12, fontweight='bold', y=0.99)

    save_fig(fig, 'figure_exp5_scalability.pdf')


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Exp 4b — Model Size Robustness (Grouped Bar Chart)
# ══════════════════════════════════════════════════════════════════════════

def figure_exp4b_model_size():
    """Grouped bar chart: FIFO vs Shared-Aware across model sizes."""
    model_labels = ['0.6B', '8B', '32B']

    fifo_ms = [452.1, 910.7, 1628.3]
    fifo_std = [9.0, 18.2, 32.6]

    sa_ms = [85.3, 148.2, 239.4]
    sa_std = [1.4, 2.5, 4.3]

    speedups = [f / s for f, s in zip(fifo_ms, sa_ms)]

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(model_labels))
    width = 0.32

    bars1 = ax.bar(x - width/2, fifo_ms, width, yerr=fifo_std, capsize=4,
                   label='FIFO', color=MAGENTA, edgecolor='black',
                   linewidth=0.5, error_kw={'linewidth': 1.0})
    bars2 = ax.bar(x + width/2, sa_ms, width, yerr=sa_std, capsize=4,
                   label='Shared-Aware', color=BLUE, edgecolor='black',
                   linewidth=0.5, error_kw={'linewidth': 1.0})

    ax.set_xlabel('Model Size (parameters)')
    ax.set_ylabel('Makespan (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend(loc='upper left')

    # Annotate speedup above each pair
    for i, sp in enumerate(speedups):
        max_h = fifo_ms[i] + fifo_std[i]
        ax.annotate(f'{sp:.1f}×', xy=(i, max_h + 30),
                    ha='center', fontsize=10, fontweight='bold', color=GREEN)

    ax.set_title('Robustness Across Model Sizes (50 tasks per model)',
                 fontsize=12, fontweight='bold')

    save_fig(fig, 'figure_exp6_model_size.pdf')


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Exp 5 — Storage Tier Impact (Dual Panel)
# ══════════════════════════════════════════════════════════════════════════

def figure_exp5_storage():
    """Two-panel figure: (a) stacked loading time, (b) grouped makespan."""
    tiers = ['Single\nNVMe', 'RAID-0\n(4×)', '10 GbE\nNFS', '100 GbE\nNFS']
    read_times = [5.1, 1.3, 12.8, 1.2]
    dma_times = [0.9, 0.9, 0.9, 0.9]
    prefetch_hidden = ['100%', '100%', '78.9%', '100%']

    no_prefetch_ms = [36.0, 32.2, 43.7, 32.1]
    with_prefetch_ms = [30.9, 30.9, 33.6, 30.9]
    reductions = ['14.2%', '4.0%', '23.1%', '3.7%']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    x = np.arange(len(tiers))

    # ── Panel (a): Stacked bar — loading time decomposition ──
    ax1.bar(x, read_times, 0.55, label='Read Time (storage → RAM)',
            color=ORANGE, edgecolor='black', linewidth=0.5)
    ax1.bar(x, dma_times, 0.55, bottom=read_times,
            label='DMA Time (RAM → GPU)', color=GREY,
            edgecolor='black', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers)
    ax1.set_ylabel('Loading Time (s)')
    ax1.set_title('(a) Loading Time Decomposition', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)

    # Annotate prefetch hidden %
    for i, (rt, dt, ph) in enumerate(zip(read_times, dma_times, prefetch_hidden)):
        total = rt + dt
        ax1.text(i, total + 0.3, f'Prefetch: {ph}', ha='center',
                 fontsize=7, fontweight='bold', color=GREEN)

    # Draw horizontal line at DMA = 0.9s
    ax1.axhline(y=0.9, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.text(3.4, 1.1, 'DMA lower bound', fontsize=7, color='gray',
             fontstyle='italic', ha='right')

    # ── Panel (b): Grouped bar — end-to-end makespan ──
    width = 0.3
    bars1 = ax2.bar(x - width/2, no_prefetch_ms, width,
                    label='No Prefetch', color=MAGENTA,
                    edgecolor='black', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, with_prefetch_ms, width,
                    label='With Prefetch', color=BLUE,
                    edgecolor='black', linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(tiers)
    ax2.set_ylabel('Makespan (s)')
    ax2.set_ylim(25, 50)
    ax2.set_title('(b) End-to-End Makespan (60 tasks, 2 models)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=8)

    # Annotate reduction %
    for i, (np_v, wp_v, red) in enumerate(
            zip(no_prefetch_ms, with_prefetch_ms, reductions)):
        mid_y = (np_v + wp_v) / 2
        ax2.annotate(f'−{red}', xy=(i, mid_y), ha='center',
                     fontsize=8, fontweight='bold', color=GREEN)

    # Note about y-axis
    ax2.text(0.98, 0.02, 'Y-axis starts at 25s',
             transform=ax2.transAxes, fontsize=7, ha='right', va='bottom',
             fontstyle='italic', color='gray')

    fig.suptitle('Storage Tier Impact on Prefetching (8B model, 15.4 GB)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure_exp5_storage_tier.pdf')


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating publication-quality thesis figures...")
    figure_exp1()
    figure_exp3_orderings()
    figure_exp4a_scalability()
    figure_exp4b_model_size()
    figure_exp5_storage()
    print("Done. All figures saved to thesis/ directory.")
