"""
Compare Selective Search 'fast' vs 'quality' proposal evaluation JSONs
and generate a single compact CVPR-ready comparison figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fast vs quality Selective Search evaluation JSONs.")
    parser.add_argument("--fast-json", type=Path, required=True)
    parser.add_argument("--quality-json", type=Path, required=True)
    parser.add_argument("--out-fig", type=Path, default=Path("outputs/figures/proposal_fast_vs_quality.png"))
    args = parser.parse_args()

    with open(args.fast_json, "r", encoding="utf-8") as f:
        fast_data = json.load(f)["results_by_k"]
    with open(args.quality_json, "r", encoding="utf-8") as f:
        qual_data = json.load(f)["results_by_k"]

    ks_fast = sorted(int(k) for k in fast_data.keys())
    ks_qual = sorted(int(k) for k in qual_data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # Left: Recall@0.50 and Recall@0.70
    axes[0].plot(
        ks_fast,
        [fast_data[str(k)]["recall@0.50"] * 100 for k in ks_fast],
        "o-",
        color="#2b6cb0",
        linewidth=2.0,
        markersize=4.5,
        label="Fast (IoU ≥ 0.50)",
    )
    axes[0].plot(
        ks_qual,
        [qual_data[str(k)]["recall@0.50"] * 100 for k in ks_qual],
        "s-",
        color="#2f855a",
        linewidth=2.0,
        markersize=4.5,
        label="Quality (IoU ≥ 0.50)",
    )
    axes[0].plot(
        ks_fast,
        [fast_data[str(k)]["recall@0.70"] * 100 for k in ks_fast],
        "o--",
        color="#2b6cb0",
        linewidth=1.6,
        markersize=4.0,
        alpha=0.75,
        label="Fast (IoU ≥ 0.70)",
    )
    axes[0].plot(
        ks_qual,
        [qual_data[str(k)]["recall@0.70"] * 100 for k in ks_qual],
        "s--",
        color="#2f855a",
        linewidth=1.6,
        markersize=4.0,
        alpha=0.75,
        label="Quality (IoU ≥ 0.70)",
    )
    axes[0].set_title("Recall vs. Proposals (Fast vs. Quality)")
    axes[0].set_xlabel("Number of Proposals per Image (K)")
    axes[0].set_ylabel("Recall (%)")
    axes[0].set_ylim(0, 102)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right", fontsize=8.5)

    # Right: MABO and AR
    axes[1].plot(
        ks_fast,
        [fast_data[str(k)]["mabo"] * 100 for k in ks_fast],
        "o-",
        color="#2b6cb0",
        linewidth=2.0,
        markersize=4.5,
        label="Fast MABO",
    )
    axes[1].plot(
        ks_qual,
        [qual_data[str(k)]["mabo"] * 100 for k in ks_qual],
        "s-",
        color="#2f855a",
        linewidth=2.0,
        markersize=4.5,
        label="Quality MABO",
    )
    axes[1].plot(
        ks_fast,
        [fast_data[str(k)]["ar"] * 100 for k in ks_fast],
        "o--",
        color="#2b6cb0",
        linewidth=1.6,
        markersize=4.0,
        alpha=0.75,
        label="Fast AR",
    )
    axes[1].plot(
        ks_qual,
        [qual_data[str(k)]["ar"] * 100 for k in ks_qual],
        "s--",
        color="#2f855a",
        linewidth=1.6,
        markersize=4.0,
        alpha=0.75,
        label="Quality AR",
    )
    axes[1].set_title("MABO & Average Recall (Fast vs. Quality)")
    axes[1].set_xlabel("Number of Proposals per Image (K)")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_ylim(0, 102)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure: {args.out_fig}")


if __name__ == "__main__":
    main()
