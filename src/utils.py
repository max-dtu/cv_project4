"""Shared dataset parsing, split management, and bounding-box utilities for Project 4."""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_DATA_DIR = Path("/dtu/datasets1/02516/potholes")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_voc_xml(xml_path: Path, images_dir: Path) -> Dict:
    """Parse a PascalVOC-style XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if not filename:
        filename = xml_path.stem + ".png"

    image_path = images_dir / filename
    if not image_path.exists():
        for ext in IMAGE_EXTS:
            candidate = images_dir / (xml_path.stem + ext)
            if candidate.exists():
                image_path = candidate
                filename = candidate.name
                break

    size_node = root.find("size")
    width = int(float(size_node.findtext("width", "0"))) if size_node is not None else 0
    height = int(float(size_node.findtext("height", "0"))) if size_node is not None else 0

    boxes: List[List[float]] = []
    labels: List[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "pothole").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)

    boxes_arr = (
        np.array(boxes, dtype=np.float32)
        if boxes
        else np.zeros((0, 4), dtype=np.float32)
    )

    return {
        "id": xml_path.stem,
        "filename": filename,
        "image_path": str(image_path),
        "xml_path": str(xml_path),
        "width": width,
        "height": height,
        "boxes": boxes_arr,
        "labels": labels,
    }


def load_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> Dict[str, Dict]:
    """Load all annotations in the Potholes dataset, keyed by image stem (e.g. 'potholes0')."""
    data_dir = Path(data_dir)
    ann_dir = data_dir / "annotations"
    img_dir = data_dir / "images"

    if not ann_dir.is_dir() or not img_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'annotations/' and 'images/' inside {data_dir}"
        )

    xml_files = sorted(ann_dir.glob("*.xml"))
    records: Dict[str, Dict] = {}
    for xml_path in xml_files:
        rec = parse_voc_xml(xml_path, img_dir)
        records[rec["id"]] = rec
    return records


def _normalize_id(entry: str) -> str:
    """Convert a filename or path like 'potholes0.xml' or 'images/potholes0.png' to stem 'potholes0'."""
    return Path(entry).stem


def load_or_create_splits(
    records: Dict[str, Dict],
    data_dir: Path = DEFAULT_DATA_DIR,
    splits_json: Optional[Path] = None,
    output_splits_path: Path = Path("outputs/splits_used.json"),
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Load train/test splits from splits.json if available, otherwise create deterministic
    train/val/test splits and persist them to outputs/splits_used.json.

    Returns a dict with keys 'train', 'val', 'test' mapping to lists of image IDs (stems).
    """
    all_ids = sorted(records.keys())
    all_id_set = set(all_ids)

    candidate_paths: List[Path] = []
    if splits_json is not None:
        candidate_paths.append(Path(splits_json))
    candidate_paths.extend(
        [
            Path(data_dir) / "splits.json",
            Path(data_dir).parent / "splits.json",
            Path("splits.json"),
        ]
    )

    found_splits_file = next((p for p in candidate_paths if p.is_file()), None)

    rng = random.Random(seed)

    if found_splits_file is not None:
        with open(found_splits_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        train_full = [
            _normalize_id(x) for x in raw.get("train", []) if _normalize_id(x) in all_id_set
        ]
        test_ids = [
            _normalize_id(x) for x in raw.get("test", []) if _normalize_id(x) in all_id_set
        ]

        if "val" in raw and raw["val"]:
            val_ids = [
                _normalize_id(x) for x in raw["val"] if _normalize_id(x) in all_id_set
            ]
            train_ids = train_full
        else:
            shuffled = list(train_full)
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_ratio)))
            val_ids = sorted(shuffled[:n_val])
            train_ids = sorted(shuffled[n_val:])

        source = str(found_splits_file)
    elif output_splits_path.is_file():
        with open(output_splits_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return {
            "train": saved["train"],
            "val": saved["val"],
            "test": saved["test"],
        }
    else:
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_ratio)))
        test_ids = sorted(shuffled[:n_test])
        trainval = shuffled[n_test:]
        n_val = max(1, int(round(len(trainval) * val_ratio)))
        val_ids = sorted(trainval[:n_val])
        train_ids = sorted(trainval[n_val:])
        source = f"generated(seed={seed}, test_ratio={test_ratio}, val_ratio={val_ratio})"

    splits = {
        "source": source,
        "seed": seed,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    output_splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between boxes_a [N, 4] and boxes_b [M, 4].
    Boxes are in [xmin, ymin, xmax, ymax] format.
    Returns float32 array of shape [N, M].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0.0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def clip_and_filter_boxes(
    boxes: np.ndarray,
    width: int,
    height: int,
    min_size: int = 8,
) -> np.ndarray:
    """Clip boxes [N, 4] to image bounds and drop degenerate/tiny or duplicate boxes."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.int32)

    b = np.asarray(boxes, dtype=np.float32).copy()
    b[:, 0] = np.clip(b[:, 0], 0, width - 1)
    b[:, 1] = np.clip(b[:, 1], 0, height - 1)
    b[:, 2] = np.clip(b[:, 2], 1, width)
    b[:, 3] = np.clip(b[:, 3], 1, height)

    b = np.round(b).astype(np.int32)
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    b = b[keep]

    if len(b) == 0:
        return b

    # Deduplicate while preserving original proposal ranking order
    _, unique_idx = np.unique(b, axis=0, return_index=True)
    unique_idx.sort()
    return b[unique_idx]
