# src/plotting.py
"""Matplotlib figure builders. Each function returns a Figure; no side effects."""

from __future__ import annotations

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def plot_wip_evolution(wip_trace: list[tuple[float, int]], title: str = "WIP en el tiempo") -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if wip_trace:
        ts, wips = zip(*wip_trace)
        ax.step(ts, wips, where="post", linewidth=1.2)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("WIP (B1 + B2)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_flow_time_histogram(flow_times: list[float], title: str = "Distribución de tiempo de flujo") -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if flow_times:
        ax.hist(flow_times, bins=30, edgecolor="black", alpha=0.75)
        mean = float(np.mean(flow_times))
        ax.axvline(mean, color="red", linestyle="--", label=f"Media={mean:.2f} min")
        ax.legend()
    ax.set_xlabel("Tiempo de flujo (min)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_throughput_bar_with_ci(
    throughputs: list[float], ci: tuple[float, float, float] | None = None
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    indices = np.arange(1, len(throughputs) + 1)
    ax.bar(indices, throughputs, alpha=0.7, edgecolor="black")
    if ci is not None:
        mean, lo, hi = ci
        ax.axhline(mean, color="red", linestyle="-", linewidth=1.5, label=f"Media={mean:.2f}")
        ax.axhspan(lo, hi, color="red", alpha=0.15, label=f"IC 95% [{lo:.2f}, {hi:.2f}]")
        ax.legend()
    ax.set_xlabel("Réplica #")
    ax.set_ylabel("Throughput (piezas/h)")
    ax.set_title("Throughput por réplica")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_heatmap(grid: dict[tuple[int, int], float], metric_name: str = "Throughput") -> Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    k1s = sorted({k for k, _ in grid})
    k2s = sorted({k for _, k in grid})
    matrix = np.array([[grid[(k1, k2)] for k1 in k1s] for k2 in k2s])
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(k1s)), k1s)
    ax.set_yticks(range(len(k2s)), k2s)
    ax.set_xlabel("K1")
    ax.set_ylabel("K2")
    ax.set_title(f"{metric_name} por configuración (K1, K2)")
    for i, k2 in enumerate(k2s):
        for j, k1 in enumerate(k1s):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white", fontsize=7)
    # Mark max
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    ax.plot(max_idx[1], max_idx[0], "o", markersize=18, markerfacecolor="none", markeredgecolor="red", markeredgewidth=2)
    fig.colorbar(im, ax=ax, label=metric_name)
    fig.tight_layout()
    return fig


def plot_comparison_wip(
    trace_a: list[tuple[float, int]],
    trace_b: list[tuple[float, int]],
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if trace_a:
        ta, wa = zip(*trace_a)
        ax.step(ta, wa, where="post", label=label_a, alpha=0.8)
    if trace_b:
        tb, wb = zip(*trace_b)
        ax.step(tb, wb, where="post", label=label_b, alpha=0.8)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("WIP (B1 + B2)")
    ax.set_title("Comparación de WIP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
