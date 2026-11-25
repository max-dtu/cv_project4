"""
Project 4.1 - Task 1: Explore the Potholes dataset and visualize ground-truth boxes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from utils import DEFAULT_DATA_DIR, load_dataset, load_or_create_splits


def compute_stats(records: dict, splits: dict) -> dict:
    widths, heights = [], []
    boxes_per_img = []
    box_widths, box_heights, box_areas, box_rel_areas, box_aspects = [], [], [], [], []
    class_counts: dict[str, int] = {}

    for rec in records.values():
        w, h = rec["width"], rec["height"]
        if w <= 0 or h <= 0:
            img = cv2.imread(rec["image_path"])
            if img is not None:
                h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

        boxes = rec["boxes"]
        boxes_per_img.append(len(boxes))
        for lbl in rec["labels"]:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        if len(boxes) > 0:
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            area = bw * bh
            img_area = max(float(w * h), 1.0)
            box_widths.extend(bw.tolist())
            box_heights.extend(bh.tolist())
            box_areas.extend(area.tolist())
            box_rel_areas.extend((area / img_area).tolist())
            box_aspects.extend((bw / np.maximum(bh, 1e-6)).tolist())

    def summarize(arr):
        a = np.asarray(arr, dtype=np.float64)
        return {
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
        }

    split_box_counts = {
        k: int(sum(len(records[i]["boxes"]) for i in ids))
        for k, ids in splits.items()
    }

    return {
        "num_images_total": len(records),
        "split_image_counts": {k: len(v) for k, v in splits.items()},
        "split_box_counts": split_box_counts,
        "num_boxes_total": int(sum(boxes_per_img)),
        "num_empty_images": int(sum(1 for c in boxes_per_img if c == 0)),
        "class_counts": class_counts,
        "image_width": summarize(widths),
        "image_height": summarize(heights),
        "boxes_per_image": summarize(boxes_per_img),
        "box_width": summarize(box_widths),
        "box_height": summarize(box_heights),
        "box_area": summarize(box_areas),
        "box_rel_area": summarize(box_rel_areas),
        "box_aspect_ratio": summarize(box_aspects),
        "_raw": {
            "boxes_per_img": boxes_per_img,
            "box_rel_areas": box_rel_areas,
            "box_aspects": box_aspects,
        },
    }


def plot_dataset_stats(raw: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    axes[0].hist(raw["boxes_per_img"], bins=range(0, max(raw["boxes_per_img"]) + 2), color="#2b6cb0", edgecolor="white")
    axes[0].set_title("Ground-Truth Boxes per Image")
    axes[0].set_xlabel("# Potholes")
    axes[0].set_ylabel("# Images")

    rel_pct = np.asarray(raw["box_rel_areas"]) * 100.0
    axes[1].hist(np.clip(rel_pct, 0, 40), bins=30, color="#2f855a", edgecolor="white")
    axes[1].set_title("GT Box Relative Area (% of Image)")
    axes[1].set_xlabel("Area (% of image, clipped at 40%)")
    axes[1].set_ylabel("# Boxes")

    aspects = np.clip(np.asarray(raw["box_aspects"]), 0, 6)
    axes[2].hist(aspects, bins=30, color="#c05621", edgecolor="white")
    axes[2].set_title("GT Box Aspect Ratio (W / H)")
    axes[2].set_xlabel("Width / Height (clipped at 6)")
    axes[2].set_ylabel("# Boxes")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_gt_examples(records: dict, train_ids: list[str], out_path: Path, n_samples: int = 6) -> None:
    # Pick diverse examples spanning low, medium, and high pothole counts
    sorted_ids = sorted(train_ids, key=lambda i: len(records[i]["boxes"]))
    indices = np.linspace(0, len(sorted_ids) - 1, n_samples, dtype=int)
    chosen = [sorted_ids[idx] for idx in indices]

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7.5))
    for ax, img_id in zip(axes.flat, chosen):
        rec = records[img_id]
        img_bgr = cv2.imread(rec["image_path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        for box in rec["boxes"]:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2.0,
                edgecolor="#e53e3e",
                facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_title(
            f"{rec['filename']} ({rec['width']}x{rec['height']}, {len(rec['boxes'])} GT)",
            fontsize=10,
        )
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Potholes dataset and plot GT examples.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
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

    stats = compute_stats(records, splits)
    raw = stats.pop("_raw")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    plot_dataset_stats(raw, args.output_dir / "figures" / "dataset_stats.png")
    plot_gt_examples(records, splits["train"], args.output_dir / "figures" / "gt_examples.png")

    print("=== Potholes Dataset Summary ===")
    print(f"Total images : {stats['num_images_total']} (empty: {stats['num_empty_images']})")
    print(f"Splits (imgs): {stats['split_image_counts']}")
    print(f"Splits (GTs) : {stats['split_box_counts']}")
    print(f"Classes      : {stats['class_counts']}")
    print(
        f"Image WxH    : W={stats['image_width']['min']:.0f}..{stats['image_width']['max']:.0f} "
        f"(med {stats['image_width']['median']:.0f}), "
        f"H={stats['image_height']['min']:.0f}..{stats['image_height']['max']:.0f} "
        f"(med {stats['image_height']['median']:.0f})"
    )
    print(
        f"Boxes/image  : min={stats['boxes_per_image']['min']:.0f}, "
        f"med={stats['boxes_per_image']['median']:.1f}, "
        f"mean={stats['boxes_per_image']['mean']:.2f}, "
        f"max={stats['boxes_per_image']['max']:.0f}"
    )
    print(
        f"Box size (px): W med={stats['box_width']['median']:.1f} (min {stats['box_width']['min']:.0f}), "
        f"H med={stats['box_height']['median']:.1f} (min {stats['box_height']['min']:.0f})"
    )
    print(f"Saved stats  : {stats_path}")
    print(f"Saved plots  : {args.output_dir / 'figures' / 'dataset_stats.png'}")
    print(f"               {args.output_dir / 'figures' / 'gt_examples.png'}")


if __name__ == "__main__":
    main()
