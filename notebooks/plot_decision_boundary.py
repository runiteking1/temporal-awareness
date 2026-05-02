"""
Plot decision boundary search traces for all 27 Kirby MCQ questions.

Each subplot shows one question: the x-axis is the binary search step,
the y-axis is the delayed reward (LDR) tested, points are colored by
the LLM's response (now / later), and a dashed line marks the boundary.
"""

import json
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_boundaries(data, out_path=None):
    results = data["results"]
    model = data["model"]

    # Organize by k_indiff (columns) × magnitude (rows)
    # 9 unique k values, 3 magnitudes (small/medium/large) = 9×3 grid
    mag_order = {"small": 0, "medium": 1, "large": 2}
    picks = sorted(results, key=lambda r: (r["k_indiff"], mag_order.get(r["magnitude"], 0)))
    unique_ks = sorted(set(r["k_indiff"] for r in picks))

    ncols = len(unique_ks)  # 9
    nrows = 3               # small, medium, large
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows))

    # Build lookup: (k_indiff, magnitude) -> result
    lookup = {(r["k_indiff"], r["magnitude"]): r for r in picks}

    for col, k_val in enumerate(unique_ks):
        for row, mag in enumerate(["small", "medium", "large"]):
            ax = axes[row, col]
            r = lookup.get((k_val, mag))
            if r is None:
                ax.set_visible(False)
                continue
            log = r["search_log"]
            steps = list(range(len(log)))
            # Handle both old format [[ldr, choice], ...] and new format [{"ldr":..., "choice":...}, ...]
            if log and isinstance(log[0], dict):
                ldrs = [entry["ldr"] for entry in log]
                choices = [entry["choice"] for entry in log]
            else:
                ldrs = [entry[0] for entry in log]
                choices = [entry[1] for entry in log]

            colors = ["#e74c3c" if c == "now" else "#2ecc71" for c in choices]

            ax.scatter(steps, ldrs, c=colors, s=40, zorder=3, edgecolors="k", linewidths=0.4)
            ax.plot(steps, ldrs, color="#bbb", linewidth=0.8, zorder=1)

            # Boundary line
            if r["flipped"] and r["boundary_ldr"] is not None:
                ax.axhline(r["boundary_ldr"], color="#3498db", linestyle="--",
                           linewidth=1.5, alpha=0.8)

            # SIR reference line
            ax.axhline(r["sir"], color="#95a5a6", linestyle=":", linewidth=0.8, alpha=0.6)

            # Original LDR marker
            ax.axhline(r["ldr_original"], color="#f39c12", linestyle=":", linewidth=0.8, alpha=0.6)

            # Title: clean, readable (escape $ to avoid LaTeX math mode)
            bnd = f"\\${r['boundary_ldr']}" if r["boundary_ldr"] is not None else "N/A"
            kb = f"{r['boundary_k']}" if r.get("boundary_k") is not None else "N/A"
            ax.set_title(
                f"Q{r['question']}: \\${r['sir']} vs \\${r['ldr_original']} in {r['delay']}d\n"
                f"boundary={bnd}  k_bnd={kb}",
                fontsize=7, linespacing=1.3,
            )
            ax.set_yscale("log")
            ax.tick_params(labelsize=6)

            # Only label axes on edges
            if row == nrows - 1:
                ax.set_xlabel("Step", fontsize=7)
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("LDR ($)", fontsize=7)
            else:
                ax.set_ylabel("")

    # Row labels (magnitude) on the right side
    for row, mag in enumerate(["small", "medium", "large"]):
        axes[row, -1].annotate(
            mag.upper(), xy=(1.15, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", rotation=-90,
            ha="center", va="center",
        )

    # Column labels (k_indiff) on top
    for col, k_val in enumerate(unique_ks):
        axes[0, col].annotate(
            f"k={k_val}", xy=(0.5, 1.35), xycoords="axes fraction",
            fontsize=8, ha="center", va="bottom", fontweight="bold",
        )

    # Shared legend
    now_patch = mpatches.Patch(color="#e74c3c", label="chose NOW")
    later_patch = mpatches.Patch(color="#2ecc71", label="chose LATER")
    boundary_line = plt.Line2D([0], [0], color="#3498db", linestyle="--", label="boundary")
    orig_line = plt.Line2D([0], [0], color="#f39c12", linestyle=":", label="original LDR")
    sir_line = plt.Line2D([0], [0], color="#95a5a6", linestyle=":", label="SIR (now)")
    fig.legend(handles=[now_patch, later_patch, boundary_line, orig_line, sir_line],
               loc="upper center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(f"Decision Boundary Search — {model}  (sorted by k_indiff)",
                 fontsize=14, y=1.035)
    fig.tight_layout(h_pad=1.5)

    if out_path is None:
        # Derive from input path or model name
        safe = model.replace("/", "_")
        persona = data.get("persona", "default")
        out_path = f"results/decision_boundary_{safe}_{persona}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/decision_boundary_Qwen_Qwen3-4B.json"
    data = load_results(path)
    plot_boundaries(data)
