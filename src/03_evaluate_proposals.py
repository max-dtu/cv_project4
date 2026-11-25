"""
Project 4.1 - Task 3: Evaluate extracted object proposals on the training set
(Recall@K, MABO@K, AR@K) to determine the required number of proposals K.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import DEFAULT_DATA_DIR, compute_iou_matrix, load_dataset, load_or_create_splits


DEFAULT_K_LIST = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
IOU_THRESHOLDS = [0.5, 0.6, 0.7, 0.75]
AR_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def evaluate_split(
    records: dict,
    proposals_npz,
    image_ids: List[str],
    k_values: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute Recall@IoU, MABO, and AR for each K on the given image_ids."""
    best_ious_per_k: Dict[int, List[float]] = {k: [] for k in k_values}

    for img_id in image_ids:
        gt_boxes = records[img_id]["boxes"]
        if len(gt_boxes) == 0:
            continue

        props = proposals_npz[img_id]
        if len(props) == 0:
            for k in k_values:
                best_ious_per_k[k].extend([0.0] * len(gt_boxes))
            continue

        max_k = min(max(k_values), len(props))
        iou_mat = compute_iou_matrix(props[:max_k], gt_boxes)  # [P, G]

        for k in k_values:
            sub = iou_mat[: min(k, len(iou_mat))]
            best_ious = sub.max(axis=0) if len(sub) > 0 else np.zeros(len(gt_boxes), dtype=np.float32)
            best_ious_per_k[k].extend(best_ious.tolist())

    results: Dict[int, Dict[str, float]] = {}
    for k in k_values:
        arr = np.asarray(best_ious_per_k[k], dtype=np.float32)
        entry: Dict[str, float] = {
            "num_gt": int(len(arr)),
            "mabo": float(np.mean(arr)) if len(arr) > 0 else 0.0,
            "ar": float(np.mean([np.mean(arr >= thr) for thr in AR_THRESHOLDS])) if len(arr) > 0 else 0.0,
        }
        for thr in IOU_THRESHOLDS:
            entry[f"recall@{thr:.2f}"] = float(np.mean(arr >= thr)) if len(arr) > 0 else 0.0
        results[k] = entry

    return results


def plot_recall_curves(
    results: Dict[int, Dict[str, float]],
    out_path: Path,
    title_suffix: str = "",
) -> None:
    ks = sorted(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    colors = ["#1a365d", "#2b6cb0", "#38a169", "#c05621"]
    for thr, color in zip(IOU_THRESHOLDS, colors):
        ys = [results[k][f"recall@{thr:.2f}"] * 100.0 for k in ks]
        axes[0].plot(ks, ys, marker="o", linewidth=2.0, markersize=4.5, color=color, label=f"IoU ≥ {thr:.2f}")

    axes[0].set_title(f"Proposal Recall vs. Number of Proposals (K){title_suffix}")
    axes[0].set_xlabel("Number of Proposals per Image (K)")
    axes[0].set_ylabel("Recall (%)")
    axes[0].set_ylim(0, 102)
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(loc="lower right")

    mabo = [results[k]["mabo"] * 100.0 for k in ks]
    ar = [results[k]["ar"] * 100.0 for k in ks]
    axes[1].plot(ks, mabo, marker="s", linewidth=2.0, markersize=4.5, color="#805ad5", label="MABO (%)")
    axes[1].plot(ks, ar, marker="^", linewidth=2.0, markersize=4.5, color="#d69e2e", label="AR@[0.50:0.95] (%)")
    axes[1].set_title(f"MABO & Average Recall vs. K{title_suffix}")
    axes[1].set_xlabel("Number of Proposals per Image (K)")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_ylim(0, 102)
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate object proposals on the training set.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits-json", type=Path, default=None)
    parser.add_argument("--proposals", type=Path, required=True, help="Path to proposals_*.npz file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--split", choices=["train", "val", "trainval"], default="train")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_LIST)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_dataset(args.data_dir)
    splits = load_or_create_splits(
        records,
        data_dir=args.data_dir,
        splits_json=args.splits_json,
        output_splits_path=args.output_dir / "splits_used.json",
        seed=args.seed,
    )

    if args.split == "trainval":
        eval_ids = splits["train"] + splits["val"]
    else:
        eval_ids = splits[args.split]

    proposals_npz = np.load(args.proposals)
    results = evaluate_split(records, proposals_npz, eval_ids, sorted(args.k_values))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"proposal_eval_{args.proposals.stem}_{args.split}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "proposals_file": str(args.proposals),
                "split": args.split,
                "num_images": len(eval_ids),
                "results_by_k": {str(k): v for k, v in results.items()},
            },
            f,
            indent=2,
        )

    out_fig = args.output_dir / "figures" / f"proposal_recall_{args.proposals.stem}_{args.split}.png"
    plot_recall_curves(results, out_fig, title_suffix=f" [{args.split}]")

    print(f"=== Proposal Evaluation ({args.proposals.name}, split={args.split}, {len(eval_ids)} imgs) ===")
    header = f"{'K':>6} | {'Rec@0.50':>9} | {'Rec@0.60':>9} | {'Rec@0.70':>9} | {'MABO':>8} | {'AR':>8}"
    print(header)
    print("-" * len(header))
    for k in sorted(results.keys()):
        r = results[k]
        print(
            f"{k:>6d} | {r['recall@0.50']*100:>8.2f}% | {r['recall@0.60']*100:>8.2f}% | "
            f"{r['recall@0.70']*100:>8.2f}% | {r['mabo']*100:>7.2f}% | {r['ar']*100:>7.2f}%"
        )

    # Suggest elbow K: smallest K achieving within 2.5 percentage points of max Recall@0.50
    max_rec50 = max(r["recall@0.50"] for r in results.values())
    suggested_k = next(k for k in sorted(results.keys()) if results[k]["recall@0.50"] >= max_rec50 - 0.025)
    print(
        f"\nSuggested K (within 2.5% of max Recall@0.50={max_rec50*100:.2f}%): "
        f"K={suggested_k} (Recall@0.50={results[suggested_k]['recall@0.50']*100:.2f}%, "
        f"MABO={results[suggested_k]['mabo']*100:.2f}%)"
    )
    print(f"Saved JSON  : {out_json}")
    print(f"Saved curve : {out_fig}")


if __name__ == "__main__":
    main()
