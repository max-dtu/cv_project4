"""
Project 4.1 - Task 2: Extract Selective Search object proposals for all images and cache to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import (
    DEFAULT_DATA_DIR,
    clip_and_filter_boxes,
    compute_iou_matrix,
    load_dataset,
    load_or_create_splits,
)


def _extract_single_image(task: Tuple[str, str, str, int, int, int]) -> Tuple[str, np.ndarray, float]:
    """Worker function to run Selective Search on a single image."""
    img_id, image_path, mode, max_dim, max_proposals, min_size = task
    cv2.setNumThreads(1)

    t0 = time.perf_counter()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    scale = 1.0
    if max_dim > 0 and max(orig_h, orig_w) > max_dim:
        scale = max_dim / float(max(orig_h, orig_w))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(resized)
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    elif mode == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        raise ValueError(f"Unsupported Selective Search mode: {mode}")

    rects = ss.process()  # [N, 4] in (x, y, w, h)
    if len(rects) == 0:
        boxes = np.zeros((0, 4), dtype=np.int32)
    else:
        rects = np.asarray(rects, dtype=np.float32)
        # Convert (x, y, w, h) -> (xmin, ymin, xmax, ymax) in original coordinates
        boxes = np.empty_like(rects)
        boxes[:, 0] = rects[:, 0] / scale
        boxes[:, 1] = rects[:, 1] / scale
        boxes[:, 2] = (rects[:, 0] + rects[:, 2]) / scale
        boxes[:, 3] = (rects[:, 1] + rects[:, 3]) / scale
        boxes = clip_and_filter_boxes(boxes, orig_w, orig_h, min_size=min_size)
        if max_proposals > 0 and len(boxes) > max_proposals:
            boxes = boxes[:max_proposals]

    elapsed = time.perf_counter() - t0
    return img_id, boxes, elapsed


def plot_proposal_examples(
    records: dict,
    proposals_by_id: dict[str, np.ndarray],
    train_ids: list[str],
    out_path: Path,
    n_samples: int = 3,
    show_top_k: int = 100,
) -> None:
    """Visualize raw proposals (left) and best-matching proposals vs GT (right)."""
    non_empty = [i for i in train_ids if len(records[i]["boxes"]) > 0]
    non_empty.sort(key=lambda i: len(records[i]["boxes"]))
    indices = np.linspace(len(non_empty) // 4, len(non_empty) - 1, n_samples, dtype=int)
    chosen = [non_empty[idx] for idx in indices]

    fig, axes = plt.subplots(n_samples, 2, figsize=(11, 3.6 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, img_id in enumerate(chosen):
        rec = records[img_id]
        img_rgb = cv2.cvtColor(cv2.imread(rec["image_path"]), cv2.COLOR_BGR2RGB)
        props = proposals_by_id[img_id]
        gt = rec["boxes"]

        # Left: top-K proposals
        ax_l = axes[row_idx, 0]
        ax_l.imshow(img_rgb)
        for box in props[:show_top_k]:
            x1, y1, x2, y2 = box
            ax_l.add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=0.8, edgecolor="#3182ce", facecolor="none", alpha=0.55
                )
            )
        ax_l.set_title(f"{rec['filename']} — First {min(show_top_k, len(props))} of {len(props)} proposals", fontsize=9.5)
        ax_l.axis("off")

        # Right: GT (red) + best-matching proposal per GT (green)
        ax_r = axes[row_idx, 1]
        ax_r.imshow(img_rgb)
        ious = compute_iou_matrix(props, gt) if len(props) > 0 else np.zeros((0, len(gt)))
        for g_idx, gbox in enumerate(gt):
            x1, y1, x2, y2 = gbox
            ax_r.add_patch(
                patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.0, edgecolor="#e53e3e", facecolor="none")
            )
            if len(props) > 0:
                best_p = int(np.argmax(ious[:, g_idx]))
                best_iou = float(ious[best_p, g_idx])
                px1, py1, px2, py2 = props[best_p]
                ax_r.add_patch(
                    patches.Rectangle(
                        (px1, py1),
                        px2 - px1,
                        py2 - py1,
                        linewidth=1.8,
                        linestyle="--",
                        edgecolor="#38a169",
                        facecolor="none",
                    )
                )
                ax_r.text(
                    x1,
                    max(0, y1 - 3),
                    f"{best_iou:.2f}",
                    color="white",
                    fontsize=7.5,
                    bbox=dict(facecolor="#2f855a", alpha=0.75, pad=1, edgecolor="none"),
                )
        ax_r.set_title(f"{rec['filename']} — GT (red) vs Best Proposal (green, IoU)", fontsize=9.5)
        ax_r.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Selective Search proposals for all Potholes images.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--mode", choices=["fast", "quality"], default="fast")
    parser.add_argument("--max-dim", type=int, default=500, help="Resize longer side to at most max-dim (0 = no resize).")
    parser.add_argument("--max-proposals", type=int, default=2000, help="Maximum proposals to store per image.")
    parser.add_argument("--min-size", type=int, default=6, help="Minimum box width/height in pixels.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError(
            "cv2.ximgproc is not available. Install 'opencv-contrib-python-headless'."
        )

    records = load_dataset(args.data_dir)
    splits = load_or_create_splits(
        records,
        data_dir=args.data_dir,
        splits_json=args.splits_json,
        output_splits_path=args.output_dir / "splits_used.json",
        seed=args.seed,
    )

    tasks = [
        (img_id, rec["image_path"], args.mode, args.max_dim, args.max_proposals, args.min_size)
        for img_id, rec in sorted(records.items())
    ]

    proposals_by_id: dict[str, np.ndarray] = {}
    times: list[float] = []
    wall_start = time.perf_counter()

    if args.workers <= 1:
        for task in tqdm(tasks, desc=f"Selective Search ({args.mode})"):
            img_id, boxes, elapsed = _extract_single_image(task)
            proposals_by_id[img_id] = boxes
            times.append(elapsed)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_extract_single_image, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Selective Search ({args.mode}, {args.workers}w)"):
                img_id, boxes, elapsed = fut.result()
                proposals_by_id[img_id] = boxes
                times.append(elapsed)

    wall_elapsed = time.perf_counter() - wall_start
    counts = np.array([len(b) for b in proposals_by_id.values()], dtype=np.int32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"proposals_{args.mode}_maxdim{args.max_dim}"
    npz_path = args.output_dir / f"{stem}.npz"
    meta_path = args.output_dir / f"{stem}_meta.json"

    np.savez_compressed(npz_path, **proposals_by_id)
    meta = {
        "mode": args.mode,
        "max_dim": args.max_dim,
        "max_proposals": args.max_proposals,
        "min_size": args.min_size,
        "num_images": len(proposals_by_id),
        "wall_time_sec": round(wall_elapsed, 2),
        "mean_per_image_sec": round(float(np.mean(times)), 4),
        "proposals_per_image": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "mean": round(float(counts.mean()), 2),
            "max": int(counts.max()),
        },
        "npz_path": str(npz_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    fig_path = args.output_dir / "figures" / f"proposal_examples_{args.mode}.png"
    plot_proposal_examples(records, proposals_by_id, splits["train"], fig_path)

    print("=== Proposal Extraction Complete ===")
    print(f"Mode / max_dim    : {args.mode} / {args.max_dim}")
    print(f"Wall time         : {wall_elapsed:.1f}s (mean {np.mean(times):.3f}s/img)")
    print(f"Proposals / image : min={counts.min()}, med={np.median(counts):.0f}, mean={counts.mean():.1f}, max={counts.max()}")
    print(f"Saved proposals   : {npz_path}")
    print(f"Saved metadata    : {meta_path}")
    print(f"Saved figure      : {fig_path}")


if __name__ == "__main__":
    main()
