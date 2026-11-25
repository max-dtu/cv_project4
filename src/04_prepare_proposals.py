"""
Project 4.1 - Task 4: Assign class (pothole=1) / background (=0) / ignore (=-1) labels
to the top-K object proposals for detector training (Part 4.2) and evaluation (Part 4.3).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import DEFAULT_DATA_DIR, compute_iou_matrix, load_dataset, load_or_create_splits


def label_split(
    records: dict,
    proposals_npz,
    image_ids: List[str],
    num_proposals: int,
    pos_iou: float,
    neg_iou_low: float,
    neg_iou_high: float,
    include_gt: bool,
) -> tuple[List[dict], Dict[str, float], np.ndarray]:
    """Label top-K proposals for all images in a split."""
    entries: List[dict] = []
    all_max_ious: List[np.ndarray] = []

    n_pos = 0
    n_neg = 0
    n_ign = 0
    n_gt_added = 0

    for img_id in image_ids:
        rec = records[img_id]
        gt_boxes = rec["boxes"]  # [G, 4] float32
        props = proposals_npz[img_id][:num_proposals].astype(np.int32)  # [P, 4]

        if include_gt and len(gt_boxes) > 0:
            gt_int = np.round(gt_boxes).astype(np.int32)
            if len(props) > 0:
                props = np.vstack([gt_int, props])
                _, uniq_idx = np.unique(props, axis=0, return_index=True)
                uniq_idx.sort()
                props = props[uniq_idx]
            else:
                props = gt_int
            n_gt_added += len(gt_int)

        n_p = len(props)
        if n_p == 0:
            labels = np.zeros((0,), dtype=np.int8)
            max_ious = np.zeros((0,), dtype=np.float32)
            gt_indices = np.zeros((0,), dtype=np.int32)
        elif len(gt_boxes) == 0:
            labels = np.zeros((n_p,), dtype=np.int8)  # all background
            max_ious = np.zeros((n_p,), dtype=np.float32)
            gt_indices = np.full((n_p,), -1, dtype=np.int32)
        else:
            iou_mat = compute_iou_matrix(props, gt_boxes)  # [P, G]
            gt_indices = np.argmax(iou_mat, axis=1).astype(np.int32)
            max_ious = iou_mat[np.arange(n_p), gt_indices].astype(np.float32)

            labels = np.full((n_p,), -1, dtype=np.int8)  # ignore zone by default
            labels[max_ious >= pos_iou] = 1
            labels[(max_ious >= neg_iou_low) & (max_ious < neg_iou_high)] = 0

        n_pos += int(np.sum(labels == 1))
        n_neg += int(np.sum(labels == 0))
        n_ign += int(np.sum(labels == -1))
        if len(max_ious) > 0:
            all_max_ious.append(max_ious)

        entries.append(
            {
                "id": img_id,
                "filename": rec["filename"],
                "image_path": rec["image_path"],
                "width": rec["width"],
                "height": rec["height"],
                "gt_boxes": gt_boxes,
                "proposals": props,
                "labels": labels,
                "max_ious": max_ious,
                "gt_indices": gt_indices,
            }
        )

    total = n_pos + n_neg + n_ign
    summary = {
        "num_images": len(entries),
        "total_proposals": total,
        "positives": n_pos,
        "negatives": n_neg,
        "ignored": n_ign,
        "gt_boxes_appended": n_gt_added,
        "pos_fraction_of_labeled": round(n_pos / max(n_pos + n_neg, 1), 4),
        "neg_to_pos_ratio": round(n_neg / max(n_pos, 1), 2),
    }
    concat_ious = np.concatenate(all_max_ious) if all_max_ious else np.zeros((0,), dtype=np.float32)
    return entries, summary, concat_ious


def plot_label_distribution(
    train_ious: np.ndarray,
    summaries: Dict[str, Dict[str, float]],
    pos_iou: float,
    neg_iou_high: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # Left: IoU histogram on train proposals
    axes[0].hist(train_ious, bins=40, range=(0.0, 1.0), color="#4a5568", edgecolor="white")
    axes[0].axvline(neg_iou_high, color="#d69e2e", linestyle="--", linewidth=2, label=f"Neg upper ({neg_iou_high:.2f})")
    axes[0].axvline(pos_iou, color="#38a169", linestyle="--", linewidth=2, label=f"Pos thresh ({pos_iou:.2f})")
    axes[0].set_yscale("log")
    axes[0].set_title("Training Proposal Max-IoU Distribution (log scale)")
    axes[0].set_xlabel("Max IoU with Ground Truth")
    axes[0].set_ylabel("# Proposals (log)")
    axes[0].legend()

    # Right: Positive vs Negative vs Ignored counts across splits
    split_names = ["train", "val", "test"]
    x = np.arange(len(split_names))
    w = 0.25
    pos_vals = [summaries[s]["positives"] for s in split_names]
    neg_vals = [summaries[s]["negatives"] for s in split_names]
    ign_vals = [summaries[s]["ignored"] for s in split_names]

    axes[1].bar(x - w, pos_vals, width=w, label="Positive (pothole)", color="#38a169")
    axes[1].bar(x, neg_vals, width=w, label="Negative (background)", color="#3182ce")
    axes[1].bar(x + w, ign_vals, width=w, label="Ignored", color="#a0aec0")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(split_names)
    axes[1].set_title("Labeled Proposal Counts per Split (log scale)")
    axes[1].set_ylabel("# Proposals (log)")
    axes[1].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label top-K proposals for detector training.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits-json", type=Path, default=None)
    parser.add_argument("--proposals", type=Path, required=True, help="Path to proposals_*.npz file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-proposals", type=int, default=1000, help="Number of proposals K per image.")
    parser.add_argument("--pos-iou", type=float, default=0.5, help="IoU >= pos_iou -> pothole (1).")
    parser.add_argument("--neg-iou-low", type=float, default=0.0, help="Lower IoU bound for background (0).")
    parser.add_argument("--neg-iou-high", type=float, default=0.3, help="Upper IoU bound for background (0); [neg_iou_high, pos_iou) is ignored (-1).")
    parser.add_argument("--no-include-gt-train", action="store_true", help="Do not append GT boxes to train proposals.")
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

    proposals_npz = np.load(args.proposals)

    prepared_splits: Dict[str, List[dict]] = {}
    summaries: Dict[str, Dict[str, float]] = {}
    train_ious = np.zeros((0,), dtype=np.float32)

    for split_name in ["train", "val", "test"]:
        include_gt = (split_name == "train") and (not args.no_include_gt_train)
        entries, summary, ious = label_split(
            records=records,
            proposals_npz=proposals_npz,
            image_ids=splits[split_name],
            num_proposals=args.num_proposals,
            pos_iou=args.pos_iou,
            neg_iou_low=args.neg_iou_low,
            neg_iou_high=args.neg_iou_high,
            include_gt=include_gt,
        )
        prepared_splits[split_name] = entries
        summaries[split_name] = summary
        if split_name == "train":
            train_ious = ious

    payload = {
        "meta": {
            "proposals_file": str(args.proposals),
            "num_proposals": args.num_proposals,
            "pos_iou": args.pos_iou,
            "neg_iou_low": args.neg_iou_low,
            "neg_iou_high": args.neg_iou_high,
            "include_gt_train": not args.no_include_gt_train,
            "class_map": {0: "background", 1: "pothole", -1: "ignore"},
            "summaries": summaries,
        },
        "splits": prepared_splits,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = args.output_dir / f"prepared_proposals_K{args.num_proposals}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    out_summary = args.output_dir / f"prepared_proposals_K{args.num_proposals}_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(payload["meta"], f, indent=2)

    out_fig = args.output_dir / "figures" / f"label_distribution_K{args.num_proposals}.png"
    plot_label_distribution(train_ious, summaries, args.pos_iou, args.neg_iou_high, out_fig)

    print(f"=== Prepared Proposals (K={args.num_proposals}, pos>={args.pos_iou}, neg=[{args.neg_iou_low}, {args.neg_iou_high})) ===")
    for split_name in ["train", "val", "test"]:
        s = summaries[split_name]
        print(
            f"{split_name:>5} | imgs={s['num_images']:>3} | pos={s['positives']:>6} | "
            f"neg={s['negatives']:>7} | ign={s['ignored']:>6} | neg:pos={s['neg_to_pos_ratio']:.1f}:1"
        )
    print(f"Saved dataset : {out_pkl}")
    print(f"Saved summary : {out_summary}")
    print(f"Saved figure  : {out_fig}")


if __name__ == "__main__":
    main()
