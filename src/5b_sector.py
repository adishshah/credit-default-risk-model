"""
Stage 5b: Sector-cyclicality view of the Stage 5 stress test results.

Investigates a real finding per Section 3.7's own instruction ("if the model doesn't
naturally show this pattern, investigate and report why honestly"):

On IMPLIED PD FOLD-CHANGE (severely_adverse / baseline), cyclical and defensive sectors
look almost indistinguishable (avg log10 fold-increase: cyclical 6.05 vs defensive 5.91).
That looks like the sector-differentiated shock isn't working.

It is working - the metric is just the wrong one. PD = N(-DD), and the normal CDF is
extremely convex in its left tail: for companies with very high baseline DD (very safe,
e.g. AAPL, MSFT), a small absolute change in DD produces an enormous RELATIVE change in
PD, swamping any signal from the cyclical/defensive multiplier. The fold-change metric is
dominated by how far out in the tail a company starts, not by the shock differentiation
actually applied.

On DISTANCE-TO-DEFAULT REDUCTION (a linear measure: baseline DD minus severely-adverse
DD), the intended pattern is clearly present: cyclical companies average a 1.63 standard-
deviation DD reduction vs. 1.20 for defensive companies - the ~1.35x/0.65x multiplier
differentiation is working exactly as designed, it's just invisible on the PD scale for
very safe names.

This is a genuine, reportable methodology finding: PD fold-change is a poor metric for
comparing stress sensitivity across companies with very different baseline risk levels;
DD reduction is the more honest comparison.

Output:
  output/charts/stress_sector_comparison.png (two-panel: DD reduction vs PD fold-change)
  data/processed/stress_sector_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STRESS_CSV = ROOT / "data" / "processed" / "stress_test_results.csv"
OUT_DIR = ROOT / "data" / "processed"
CHART_DIR = ROOT / "output" / "charts"


def main():
    df = pd.read_csv(STRESS_CSV)

    piv_dd = df.pivot_table(index=["ticker", "sector", "cyclicality"],
                             columns="scenario", values="distance_to_default").reset_index()
    piv_pd = df.pivot_table(index=["ticker", "sector", "cyclicality"],
                             columns="scenario", values="implied_PD").reset_index()

    piv_dd["DD_reduction"] = piv_dd["baseline"] - piv_dd["severely_adverse"]
    piv_pd["log10_fold_increase"] = np.log10(piv_pd["severely_adverse"] / piv_pd["baseline"])

    merged = piv_dd[["ticker", "sector", "cyclicality", "DD_reduction"]].merge(
        piv_pd[["ticker", "log10_fold_increase"]], on="ticker"
    )
    merged = merged.sort_values(["cyclicality", "sector", "DD_reduction"], ascending=[True, True, False])
    merged.to_csv(OUT_DIR / "stress_sector_summary.csv", index=False)

    summary = merged.groupby("cyclicality")[["DD_reduction", "log10_fold_increase"]].agg(["mean", "std"])
    print("Summary by cyclicality:")
    print(summary)
    print(f"\nOn DD reduction (linear, undistorted): cyclical companies deteriorate "
          f"{merged[merged.cyclicality=='cyclical']['DD_reduction'].mean() / merged[merged.cyclicality=='defensive']['DD_reduction'].mean():.2f}x "
          f"as much as defensive companies, on average - the intended sector pattern.")
    print(f"On PD fold-change (convex, tail-dominated): the same comparison collapses to "
          f"near 1.0x, because baseline tail position swamps the shock-size signal for "
          f"very safe companies.")

    # --- two-panel chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    colors = {"cyclical": "#d62728", "defensive": "#1f77b4"}

    order = merged.sort_values("DD_reduction", ascending=True)["ticker"].tolist()
    plot_df1 = merged.set_index("ticker").loc[order]
    bar_colors1 = [colors[c] for c in plot_df1["cyclicality"]]
    ax1.barh(plot_df1.index, plot_df1["DD_reduction"], color=bar_colors1)
    ax1.set_xlabel("DD reduction, baseline -> severely adverse (standard deviations)")
    ax1.set_title("Linear measure: DD reduction\n(sector pattern IS visible here)")
    ax1.grid(alpha=0.3, axis="x")

    order2 = merged.sort_values("log10_fold_increase", ascending=True)["ticker"].tolist()
    plot_df2 = merged.set_index("ticker").loc[order2]
    bar_colors2 = [colors[c] for c in plot_df2["cyclicality"]]
    ax2.barh(plot_df2.index, plot_df2["log10_fold_increase"], color=bar_colors2)
    ax2.set_xlabel("log10(PD severely-adverse / PD baseline)")
    ax2.set_title("Convex measure: PD fold-change\n(sector pattern is MASKED by tail convexity)")
    ax2.grid(alpha=0.3, axis="x")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors["cyclical"], label="Cyclical"),
                        Patch(facecolor=colors["defensive"], label="Defensive")]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(CHART_DIR / "stress_sector_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("\nChart written to output/charts/stress_sector_comparison.png")


if __name__ == "__main__":
    main()