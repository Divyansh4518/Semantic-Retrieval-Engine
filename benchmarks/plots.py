"""Automated matplotlib chart generation from benchmark sweep results.

Usage::

    python -m benchmarks.plots

Reads the latest aggregated JSONs from ``outputs/aggregated/`` and
generates PNG charts in ``outputs/figures/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_BENCHMARKS_DIR = Path(__file__).resolve().parent
_AGG_DIR = _BENCHMARKS_DIR / "outputs" / "aggregated"
_FIG_DIR = _BENCHMARKS_DIR / "outputs" / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Color Palette
# ------------------------------------------------------------------
_COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "accent": "#43E8D8",
    "warn": "#FFD93D",
    "muted": "#8B8B9E",
    "hnsw": "#00C896",    # teal-green — reserved for FaissHNSWIndex series
}


# ------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------


def _apply_style() -> None:
    """Apply a consistent dark theme to all charts."""
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#1A1A2E",
            "axes.facecolor": "#16213E",
            "axes.edgecolor": "#3A3A5C",
            "axes.labelcolor": "#E0E0E0",
            "text.color": "#E0E0E0",
            "xtick.color": "#B0B0C0",
            "ytick.color": "#B0B0C0",
            "grid.color": "#2A2A4A",
            "grid.alpha": 0.6,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.dpi": 300,
        }
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _find_latest(prefix: str) -> Path | None:
    """Find the most recently created aggregated JSON for *prefix*."""
    candidates = sorted(_AGG_DIR.glob(f"{prefix}_*.json"))
    return candidates[-1] if candidates else None


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Chart: Recall vs ef_search (Sweep C)
# ------------------------------------------------------------------


def plot_recall_vs_ef_search() -> None:
    """Dual-series line chart -- Recall@10 vs ef_search for Graph and HNSW (Sweep C)."""
    path = _find_latest("sweep_c")
    if not path:
        print("  [SKIP] No sweep_c data found, skipping recall_vs_ef_search")
        return

    data = _load_json(path)
    results = data["results"]

    ef_values    = [r["ef_search"] for r in results]
    graph_recalls = [r["recall_mean"] for r in results]
    hnsw_recalls  = [
        r["faiss_hnsw_recall_mean"]
        if "faiss_hnsw_recall_mean" in r else None
        for r in results
    ]

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        ef_values, graph_recalls,
        marker="o", color=_COLORS["primary"],
        linewidth=2, markersize=7, label="GraphIndex", zorder=3,
    )
    ax.fill_between(ef_values, graph_recalls, alpha=0.10, color=_COLORS["primary"])

    if any(v is not None for v in hnsw_recalls):
        hnsw_vals = [v if v is not None else float("nan") for v in hnsw_recalls]
        ax.plot(
            ef_values, hnsw_vals,
            marker="D", color=_COLORS["hnsw"],
            linewidth=2, markersize=7, label="FaissHNSWIndex", zorder=3,
        )
        ax.fill_between(ef_values, hnsw_vals, alpha=0.10, color=_COLORS["hnsw"])

    ax.set_xlabel("ef_search")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 vs. ef_search -- Graph vs HNSW (Sweep C)")
    ax.set_xscale("log", base=2)
    ax.legend(framealpha=0.3)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    out = _FIG_DIR / "recall_vs_ef_search.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


# ------------------------------------------------------------------
# Chart: Recall vs M (Sweep B)
# ------------------------------------------------------------------


def plot_recall_vs_M() -> None:
    """Dual-series line chart -- Recall@10 vs M for Graph and HNSW (Sweep B)."""
    path = _find_latest("sweep_b")
    if not path:
        print("  [SKIP] No sweep_b data found, skipping recall_vs_M")
        return

    data = _load_json(path)
    results = data["results"]

    m_values     = [r["M"] for r in results]
    graph_recalls = [r["recall_mean"] for r in results]
    hnsw_recalls  = [
        r["faiss_hnsw_recall_mean"]
        if "faiss_hnsw_recall_mean" in r else None
        for r in results
    ]

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        m_values, graph_recalls,
        marker="s", color=_COLORS["secondary"],
        linewidth=2, markersize=7, label="GraphIndex", zorder=3,
    )
    ax.fill_between(m_values, graph_recalls, alpha=0.10, color=_COLORS["secondary"])

    if any(v is not None for v in hnsw_recalls):
        hnsw_vals = [v if v is not None else float("nan") for v in hnsw_recalls]
        ax.plot(
            m_values, hnsw_vals,
            marker="D", color=_COLORS["hnsw"],
            linewidth=2, markersize=7, label="FaissHNSWIndex", zorder=3,
        )
        ax.fill_between(m_values, hnsw_vals, alpha=0.10, color=_COLORS["hnsw"])

    ax.set_xlabel("M (max neighbors per node)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 vs. M -- Graph vs HNSW (Sweep B)")
    ax.legend(framealpha=0.3)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    out = _FIG_DIR / "recall_vs_M.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


# ------------------------------------------------------------------
# Chart: Build Time vs N (Sweep A)
# ------------------------------------------------------------------


def plot_build_time_vs_N() -> None:
    """Log-scale line chart -- Build Time vs N for Graph, Exact, Flat, and HNSW."""
    path = _find_latest("sweep_a")
    if not path:
        print("  [SKIP] No sweep_a data found, skipping build_time_vs_N")
        return

    data = _load_json(path)
    results = data["results"]

    n_values = [r["N"] for r in results]
    graph_times = [r.get("graph_build_time_s", r.get("build_time_s")) for r in results]
    exact_times = [r.get("exact_build_time_s", 0) for r in results]
    faiss_times = [r.get("faiss_build_time_s", 0) for r in results]
    hnsw_times  = [r.get("faiss_hnsw_build_time_s", 0) for r in results]

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        n_values, graph_times,
        marker="^", color=_COLORS["accent"],
        linewidth=2, markersize=7, label="GraphIndex", zorder=3,
    )
    if any(t > 0 for t in exact_times):
        ax.plot(
            n_values, exact_times,
            marker="o", color=_COLORS["primary"],
            linewidth=2, markersize=7, label="ExactIndex", zorder=3,
        )
    if any(t > 0 for t in faiss_times):
        ax.plot(
            n_values, faiss_times,
            marker="s", color=_COLORS["secondary"],
            linewidth=2, markersize=7, label="FaissFlatIndex", zorder=3,
        )
    if any(t > 0 for t in hnsw_times):
        ax.plot(
            n_values, hnsw_times,
            marker="D", color=_COLORS["hnsw"],
            linewidth=2, markersize=7, label="FaissHNSWIndex", zorder=3,
        )
    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("Build Time (seconds)")
    ax.set_title("Build Time vs. Dataset Size (Sweep A)")
    ax.set_yscale("log")
    ax.legend(framealpha=0.3)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = _FIG_DIR / "build_time_vs_N.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


# ------------------------------------------------------------------
# Chart: Query Latency (p50/p95) vs N (Sweep A)
# ------------------------------------------------------------------


def plot_query_latency_vs_N() -> None:
    """Line chart -- p50 query latency per engine vs dataset size N (Sweep A).

    Plots three series side-by-side:
        * GraphIndex p50  (primary)
        * FaissFlatIndex p50  (secondary)
        * FaissHNSWIndex p50  (hnsw)

    Older sweep_a files that pre-date HNSW integration will simply omit
    the HNSW series rather than crashing.
    """
    path = _find_latest("sweep_a")
    if not path:
        print("  [SKIP] No sweep_a data found, skipping query_latency_vs_N")
        return

    data = _load_json(path)
    results = data["results"]

    n_values  = [r["N"] for r in results]
    graph_p50 = [r["query_latency"]["p50"] * 1000 for r in results]
    flat_p50  = [
        r["faiss_query_latency"]["p50"] * 1000
        if "faiss_query_latency" in r else None
        for r in results
    ]
    hnsw_p50  = [
        r["faiss_hnsw_query_latency"]["p50"] * 1000
        if "faiss_hnsw_query_latency" in r else None
        for r in results
    ]

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        n_values, graph_p50,
        marker="^", color=_COLORS["accent"],
        linewidth=2, markersize=7, label="GraphIndex p50", zorder=3,
    )

    if any(v is not None for v in flat_p50):
        flat_vals = [v if v is not None else float("nan") for v in flat_p50]
        ax.plot(
            n_values, flat_vals,
            marker="s", color=_COLORS["secondary"],
            linewidth=2, markersize=7, label="FaissFlatIndex p50", zorder=3,
        )

    if any(v is not None for v in hnsw_p50):
        hnsw_vals = [v if v is not None else float("nan") for v in hnsw_p50]
        ax.plot(
            n_values, hnsw_vals,
            marker="D", color=_COLORS["hnsw"],
            linewidth=2, markersize=7, label="FaissHNSWIndex p50", zorder=3,
        )

    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("p50 Query Latency (ms)")
    ax.set_title("p50 Query Latency vs. Dataset Size — Graph vs Flat vs HNSW (Sweep A)")
    ax.legend(framealpha=0.3)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = _FIG_DIR / "query_latency_vs_N.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


# ------------------------------------------------------------------
# Chart: Connected Components vs N (Sweep A)
# ------------------------------------------------------------------


def plot_components_vs_N() -> None:
    """Bar chart — Connected Components vs dataset size ``N``."""
    path = _find_latest("sweep_a")
    if not path:
        print("  [SKIP] No sweep_a data found, skipping components_vs_N")
        return

    data = _load_json(path)
    results = data["results"]

    n_values = [str(r["N"]) for r in results]
    components = [r["graph_health"]["num_components"] for r in results]

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        n_values, components,
        color=_COLORS["warn"],
        edgecolor=_COLORS["muted"],
        linewidth=0.8, zorder=3,
    )

    for bar, val in zip(bars, components):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("Connected Components")
    ax.set_title("Connected Components vs. Dataset Size (Sweep A)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = _FIG_DIR / "components_vs_N.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


# ------------------------------------------------------------------
# Chart: QPS vs N — Sweep F
# ------------------------------------------------------------------


def plot_sweep_f_qps_vs_N() -> None:
    """Line chart -- QPS of ExactIndex vs FaissFlatIndex as N scales."""
    path = _find_latest("sweep_f")
    if not path:
        print("  [SKIP] No sweep_f data found, skipping sweep_f_qps_vs_N")
        return

    data = _load_json(path)
    results = data["results"]

    n_values = [r["N"] for r in results]
    exact_qps = [r["exact_query_latency"]["qps"] for r in results]
    faiss_qps = [r["faiss_query_latency"]["qps"] for r in results]

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        n_values, exact_qps,
        marker="o", color=_COLORS["primary"],
        linewidth=2, markersize=7, label="ExactIndex (NumPy)", zorder=3,
    )
    ax.plot(
        n_values, faiss_qps,
        marker="s", color=_COLORS["secondary"],
        linewidth=2, markersize=7, label="FaissFlatIndex (C++)", zorder=3,
    )
    ax.fill_between(n_values, exact_qps, faiss_qps, alpha=0.1, color=_COLORS["muted"])
    ax.set_xlabel("Dataset Size (N)")
    ax.set_ylabel("Queries Per Second (QPS)")
    ax.set_title("QPS vs. Dataset Size (Sweep F)")
    ax.legend(framealpha=0.3)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = _FIG_DIR / "sweep_f_qps_vs_N.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


# ------------------------------------------------------------------
# Chart: Batch Throughput — Sweep G
# ------------------------------------------------------------------


def plot_sweep_g_batch_qps() -> None:
    """Grouped bar chart -- Sequential vs Batched QPS for both engines."""
    path = _find_latest("sweep_g")
    if not path:
        print("  [SKIP] No sweep_g data found, skipping sweep_g_batch_qps")
        return

    data = _load_json(path)
    results = data["results"]

    batch_labels = [str(r["batch_size"]) for r in results]
    exact_seq = [r["exact_seq_qps"] for r in results]
    exact_batch = [r["exact_batch_qps"] for r in results]
    faiss_seq = [r["faiss_seq_qps"] for r in results]
    faiss_batch = [r["faiss_batch_qps"] for r in results]

    x = np.arange(len(batch_labels))
    width = 0.18

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 1.5 * width, exact_seq, width,
           label="Exact Sequential", color=_COLORS["primary"], zorder=3)
    ax.bar(x - 0.5 * width, exact_batch, width,
           label="Exact Batched", color=_COLORS["accent"], zorder=3)
    ax.bar(x + 0.5 * width, faiss_seq, width,
           label="FAISS Sequential", color=_COLORS["secondary"], zorder=3)
    ax.bar(x + 1.5 * width, faiss_batch, width,
           label="FAISS Batched", color=_COLORS["warn"], zorder=3)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Queries Per Second (QPS)")
    ax.set_title("Batch Throughput: Sequential vs Batched (Sweep G)")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_labels)
    ax.set_yscale("log")
    ax.legend(framealpha=0.3, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = _FIG_DIR / "sweep_g_batch_qps.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def generate_all() -> None:
    """Generate all available charts from aggregated sweep data."""
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_recall_vs_ef_search()
    plot_recall_vs_M()
    plot_build_time_vs_N()
    plot_query_latency_vs_N()
    plot_components_vs_N()
    plot_sweep_f_qps_vs_N()
    plot_sweep_g_batch_qps()

    print(f"\n[OK] All plots written to {_FIG_DIR}")


if __name__ == "__main__":
    generate_all()
