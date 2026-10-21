"""Record local MAT schemas, split counts, and SHA-256 hashes without publishing data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


ROOT = Path(__file__).resolve().parents[1]


def inspect(path):
    schema = whosmat(path)
    keys = {entry[0] for entry in schema}
    fields = [key for key in ["label", "Label", "train_mask", "val_mask", "test_mask"] if key in keys]
    values = loadmat(path, variable_names=fields)
    labels = np.asarray(values.get("label", values.get("Label"))).ravel()
    masks = {key: np.asarray(values[key]).ravel().astype(bool) for key in fields if key.endswith("_mask")}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    info = {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "nodes": int(labels.size),
        "positives": int((labels == 1).sum()),
        "fields": [{"name": key, "shape": list(shape), "type": kind} for key, shape, kind in schema],
        "split_counts": {key: int(mask.sum()) for key, mask in masks.items()},
    }
    if "train_mask" in masks:
        info["training_positives"] = int((labels[masks["train_mask"]] == 1).sum())
    if len(masks) == 3:
        usage = sum(mask.astype(np.uint8) for mask in masks.values())
        info["overlapping_split_nodes"] = int((usage > 1).sum())
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "inventory.json")
    args = parser.parse_args()
    paths = sorted((ROOT / "data" / "raw").glob("*.mat"))
    paths += sorted((ROOT / "data" / "generated").glob("*.mat"))
    result = {
        "description": "Inventory of local files; data and graph caches are excluded from Git.",
        "files": [inspect(path) for path in paths],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Recorded {len(paths)} MAT files in {args.output}")


if __name__ == "__main__":
    main()
