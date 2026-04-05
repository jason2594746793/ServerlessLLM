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
    """Broken-axis bar chart: 4-strategy ablation (501 tasks, 3 models)."""
    strategies = ['FIFO\n(extrapolated)', 'Model\nGrouping', 'Grouping\n+Prefetch', 'Shared-\nAware']
    makespans = [45495.0, 228.5, 206.8, 164.7]
    stds     = [3353.0,  14.2,   4.6,   8.5]
    switches = ['500', '2', '2', '2']
    speedups = ['1×', '199×', '220×', '276×']
    colors   = [MAGENTA, ORANGE, BLUE, GREEN]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5.5),
        gridspec_kw={'height_ratios': [1, 3.5], 'hspace': 0.08}
    )

    # Top panel: FIFO only visible
    ax_top.bar(strategies, makespans, yerr=stds, capsize=4,
               color=colors, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1.0})
    ax_top.set_ylim(40000, 51000)
    ax_top.tick_params(bottom=False)
    ax_top.set_ylabel('')
    ax_top.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    # Annotate FIFO value
    ax_top.text(0, 45495 + 3353 + 400, '45,495s\n(12.6h)', ha='center',
                va='bottom', fontsize=8, fontweight='bold')

    # Bottom panel: non-FIFO bars
    bars_bot = ax_bot.bar(strategies, makespans, yerr=stds, capsize=4,
                          color=colors, edgecolor='black', linewidth=0.5,
                          error_kw={'linewidth': 1.0})
    ax_bot.set_ylim(0, 310)
    ax_bot.set_ylabel('Makespan (s)')

    # Annotate non-FIFO values above bars
    for i, (bar, val, std) in enumerate(zip(bars_bot, makespans, stds)):
        if i == 0:
            continue
        x = bar.get_x() + bar.get_width() / 2
        ax_bot.text(x, val + std + 4, f'{val:.1f}s', ha='center',
                    va='bottom', fontsize=8, fontweight='bold')

    # Speedup + switch count annotated inside or just above bottom of each bar
    for i, (sw, sp, color) in enumerate(zip(switches, speedups, colors)):
        x = bars_bot[i].get_x() + bars_bot[i].get_width() / 2
        # speedup below x-axis label via annotation box
        ax_bot.annotate(f'{sp}\n{sw} sw.',
                        xy=(x, 0), xytext=(x, -38),
                        ha='center', va='top', fontsize=8,
                        fontweight='bold', color=color,
                        annotation_clip=False)

    ax_bot.set_ylim(0, 310)

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

    fig.suptitle('Scheduling Strategy Ablation (501 tasks, 3 models: 0.6B, 8B, 32B)',
                 fontsize=11, fontweight='bold', y=0.99)
    fig.subplots_adjust(bottom=0.18)

    save_fig(fig, 'figure_exp1_strategy_bar.pdf')


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Exp 3 — Johnson's Rule Ordering Ablation (Horizontal Bar Chart)
# ══════════════════════════════════════════════════════════════════════════

def figure_exp3_orderings():
    """Horizontal bar chart: all 6 permutations — measured makespan.
    Data from Table 4.2 (thesis): 90 tasks, 3 models, prefetch ON, cold start.
    """
    # (label, measured, std, tag)
    orderings = [
        ('0.6B → 8B → 32B', 132.4, 3.5, "Johnson's Rule (optimal)"),
        ('0.6B → 32B → 8B', 132.9, 0.0, ''),
        ('8B → 32B → 0.6B', 173.0, 0.0, ''),
        ('8B → 0.6B → 32B', 175.7, 8.1, ''),
        ('32B → 8B → 0.6B', 223.1, 8.7, 'Size-descending'),
        ('32B → 0.6B → 8B', 227.8, 5.1, 'Alphabetical (default)'),
    ]

    # Best at top → reverse for barh
    orderings_rev = list(reversed(orderings))
    labels = [o[0] for o in orderings_rev]
    vals = [o[1] for o in orderings_rev]
    errs = [o[2] for o in orderings_rev]
    tags = [o[3] for o in orderings_rev]

    fig, ax = plt.subplots(figsize=(8, 3.8))

    bar_colors = []
    for o in orderings_rev:
        if 'optimal' in o[3]:
            bar_colors.append(GREEN)
        elif 'default' in o[3] or 'descending' in o[3]:
            bar_colors.append(MAGENTA)
        else:
            bar_colors.append(BLUE)

    ax.barh(range(len(labels)), vals, xerr=errs, capsize=3,
            color=bar_colors, edgecolor='black', linewidth=0.5,
            height=0.55, error_kw={'linewidth': 1.0},
            label='Measured', zorder=3)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Makespan (s)')
    ax.set_xlim(120, 260)

    jr_val = 132.4
    for i, (val, err, tag) in enumerate(zip(vals, errs, tags)):
        pct = ((val - jr_val) / jr_val) * 100
        if pct > 0:
            ax.text(val + err + 1.5, i, f'+{pct:.1f}%',
                    va='center', fontsize=8, color='gray')
        else:
            ax.text(val + err + 1.5, i, 'optimal',
                    va='center', fontsize=8, fontweight='bold', color=GREEN)
        if tag:
            ax.text(val - 1.5, i + 0.32, tag, va='center', ha='right',
                    fontsize=7, fontstyle='italic', color='gray')

    ax.set_title("Johnson's Rule Ordering Ablation (90 tasks, 3 models: 0.6B, 8B, 32B)",
                 fontsize=11, fontweight='bold')
    ax.text(0.98, 0.02, 'X-axis starts at 120s',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            fontstyle='italic', color='gray')

    fig.tight_layout()
    save_fig(fig, 'figure_exp3_ordering_bar.pdf')


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Exp 4a — Scalability (Dual Y-Axis: Makespan + Speedup)
# ══════════════════════════════════════════════════════════════════════════

def figure_exp4a_scalability():
    """Dual Y-axis line plot: FIFO, Grouping, Shared-Aware + Speedup.
    Calibrated model lines (solid = measured range n≤501, dashed = projected)
    with measured anchor points overlaid as markers.
    Equations from Table 4.3:
      C_FIFO(n) ≈ 90.3n
      C_G(n)    ≈ 218 + 0.021n
      C_SA(n)   ≈ 129 + 0.071n
    Projection capped at n=1000 to stay within reasonable extrapolation range.
    """
    # Measured anchor points
    measured_n     = [99,    501   ]
    fifo_meas      = [8920,  45495 ]
    group_meas     = [220.1, 228.5 ]
    group_meas_std = [5.9,   14.2  ]
    sa_meas        = [136.3, 164.7 ]
    sa_meas_std    = [5.8,   8.5   ]
    speedups_meas  = [65,    276   ]

    # Calibrated model lines
    n_fit  = np.linspace(10,  501,  200)   # solid — measured range
    n_proj = np.linspace(501, 1000, 100)   # dashed — projection

    def c_fifo(n):  return 90.3 * n
    def c_g(n):     return 218 + 0.021 * n
    def c_sa(n):    return 129 + 0.071 * n
    def speedup(n): return c_fifo(n) / c_sa(n)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size (tasks)')
    ax1.set_ylabel('Makespan (s, log scale)')

    # ── FIFO ──
    ax1.plot(n_fit,  c_fifo(n_fit),  '-',  color=MAGENTA, linewidth=1.6,
             label='FIFO (extrapolated)')
    ax1.plot(n_proj, c_fifo(n_proj), '--', color=MAGENTA, linewidth=1.2, alpha=0.6)
    ax1.errorbar(measured_n, fifo_meas, yerr=[0, 3353], fmt='^',
                 color=MAGENTA, markersize=8, capsize=4, zorder=5,
                 markeredgecolor='black', markeredgewidth=0.5)

    # ── Model Grouping ──
    ax1.plot(n_fit,  c_g(n_fit),  '-',  color=ORANGE, linewidth=1.6,
             label='Model Grouping')
    ax1.plot(n_proj, c_g(n_proj), '--', color=ORANGE, linewidth=1.2, alpha=0.6)
    ax1.errorbar(measured_n, group_meas, yerr=group_meas_std, fmt='s',
                 color=ORANGE, markersize=7, capsize=4, zorder=5,
                 markeredgecolor='black', markeredgewidth=0.5)

    # ── Shared-Aware ──
    ax1.plot(n_fit,  c_sa(n_fit),  '-',  color=BLUE, linewidth=1.6,
             label='Shared-Aware')
    ax1.plot(n_proj, c_sa(n_proj), '--', color=BLUE, linewidth=1.2, alpha=0.6)
    ax1.errorbar(measured_n, sa_meas, yerr=sa_meas_std, fmt='o',
                 color=BLUE, markersize=7, capsize=4, zorder=5,
                 markeredgecolor='black', markeredgewidth=0.5)

    # Projected region shading + label
    ax1.axvspan(501, 1000, alpha=0.04, color='gray')
    ax1.text(750, 200, 'projected\n(model)', fontsize=7, color='gray',
             fontstyle='italic', ha='center', va='bottom')

    # Vertical lines at measured n
    for n in measured_n:
        ax1.axvline(n, color='gray', linestyle=':', linewidth=0.6, alpha=0.4)

    ax1.set_xticks([99, 200, 300, 400, 501, 1000])
    ax1.set_xticklabels(['99', '200', '300', '400', '501', '1,000'])
    ax1.set_xlim(50, 1100)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _:
            f'{x/3600:.1f}h' if x >= 3600 else f'{int(x):,}s'))

    # 1-hour reference line
    ax1.axhline(3600, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.text(55, 3600 * 1.15, '1 hour', fontsize=7, color='gray')

    ax1.legend(loc='upper left', framealpha=0.9)

    # ── Right Y-axis: speedup ──
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.plot(n_fit,  speedup(n_fit),  '-',  color=GREEN, linewidth=1.5,
             label='Speedup (FIFO/SA)')
    ax2.plot(n_proj, speedup(n_proj), '--', color=GREEN, linewidth=1.2, alpha=0.6)
    ax2.plot(measured_n, speedups_meas, 'D', color=GREEN, markersize=7,
             zorder=5, markeredgecolor='black', markeredgewidth=0.5)
    ax2.set_ylabel('Speedup (×)', color=GREEN)
    ax2.tick_params(axis='y', labelcolor=GREEN)

    for n, sp in zip(measured_n + [1000],
                     speedups_meas + [int(speedup(1000))]):
        ax2.annotate(f'{sp}×', xy=(n, sp), xytext=(6, 3),
                     textcoords='offset points', fontsize=8,
                     color=GREEN, fontweight='bold')

    ax2.set_ylim(0, 550)
    ax2.legend(loc='center right', framealpha=0.9)

    fig.suptitle('Scalability: Makespan vs Batch Size (3 models: 0.6B, 8B, 32B)',
                 fontsize=11, fontweight='bold', y=0.99)

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.8))

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
    ax1.legend(loc='upper left', fontsize=8)

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
    ax2.set_ylim(25, 52)
    ax2.set_title('(b) End-to-End Makespan (60 tasks, 2 models)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=8)

    # Annotate reduction % above each bar pair
    for i, (np_v, wp_v, red) in enumerate(
            zip(no_prefetch_ms, with_prefetch_ms, reductions)):
        top_y = max(np_v, wp_v) + 0.6
        ax2.annotate(f'savings: −{red}', xy=(i, top_y), ha='center',
                     fontsize=7.5, fontweight='bold', color=GREEN)

    # Note about y-axis
    ax2.text(0.98, 0.02, 'Y-axis starts at 25s',
             transform=ax2.transAxes, fontsize=7, ha='right', va='bottom',
             fontstyle='italic', color='gray')

    fig.suptitle('Storage Tier Impact on Prefetching (8B model, 15.4 GB)',
                 fontsize=12, fontweight='bold')
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
