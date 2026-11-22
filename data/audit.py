"""Inventario y partición reproducible de las imágenes originales."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageOps

CLASSES = ("cyst", "normal", "stone", "tumor")
SPLITS = {"train": 0.70, "test": 0.25, "val": 0.05}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DISTANCE = 4
MAX_RMSE = 0.035
SEED = 27


def difference_hash(image):
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(gray)
    bits = np.packbits(values[:, 1:] > values[:, :-1])
    return int.from_bytes(bits, "big")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inspect(root):
    rows, damaged, auxiliary, thumbnails = [], [], [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        label = path.parent.name.lower()
        if path.suffix.lower() not in EXTENSIONS or label not in CLASSES:
            auxiliary.append(str(relative))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                pixels = image.convert("RGB")
                width, height = pixels.size
                rows.append({
                    "path": str(relative), "class": label, "format": image.format,
                    "mode": image.mode, "width": width, "height": height,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                    "pixel_sha256": sha256(f"{width}x{height}:".encode() + pixels.tobytes()).hexdigest(),
                    "exif_tags": len(image.getexif()),
                    "dhash": f"{difference_hash(pixels):016x}",
                })
                thumb = ImageOps.grayscale(pixels).resize((32, 32), Image.Resampling.BOX)
                thumbnails.append(np.asarray(thumb, dtype=np.float32) / 255)
        except (OSError, ValueError) as error:
            damaged.append({"path": str(relative), "error": str(error)})
    if {row["class"] for row in rows} != set(CLASSES):
        raise ValueError("Se requieren imágenes decodificables de las cuatro clases.")
    return rows, damaged, auxiliary, np.asarray(thumbnails)


def cluster(rows, thumbnails):
    parent = list(range(len(rows)))

    def root(index):
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    buckets = defaultdict(list)
    representatives = {}
    hashes = [int(row["dhash"], 16) for row in rows]
    near_edges = []
    for index, (row, fingerprint) in enumerate(zip(rows, hashes)):
        digest = row["pixel_sha256"]
        representative = representatives.setdefault(digest, index)
        if representative != index:
            parent[index] = root(representative)
            continue
        # Si difieren en cuatro bits o menos, al menos una de las cinco bandas coincide.
        bands = [(band, (fingerprint >> (band * 13)) & 8191) for band in range(5)]
        candidates = {other for band in bands for other in buckets[band]}
        for other in sorted(candidates):
            distance = (fingerprint ^ hashes[other]).bit_count()
            if distance > DISTANCE:
                continue
            rmse = float(np.sqrt(np.mean((thumbnails[index] - thumbnails[other]) ** 2)))
            if rmse <= MAX_RMSE:
                left, right = root(index), root(other)
                parent[max(left, right)] = min(left, right)
                near_edges.append({"path_a": rows[other]["path"], "path_b": row["path"],
                                   "distance": distance, "rmse": round(rmse, 7), "same_class": row["class"] == rows[other]["class"]})
        for band in bands:
            buckets[band].append(index)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        group = root(index)
        row["group"] = f"g{group:05}"
        groups[group].append(row)
    return list(groups.values()), near_edges


def partition(groups):
    rng = Random(SEED)
    accepted = defaultdict(list)
    for group in groups:
        labels = {row["class"] for row in group}
        if len(labels) > 1:
            for row in group:
                row["split"] = "review_conflict"
            continue
        unique = {}
        for row in group:
            if row["pixel_sha256"] in unique:
                row["split"] = "duplicate"
            else:
                unique[row["pixel_sha256"]] = row
        accepted[group[0]["class"]].append(list(unique.values()))

    for label in CLASSES:
        blocks = accepted[label]
        if len(blocks) < 3:
            raise ValueError(f"{label}: hacen falta al menos tres grupos sin conflicto.")
        rng.shuffle(blocks)
        blocks.sort(key=len, reverse=True)
        total = sum(map(len, blocks))
        targets = {split: total * fraction for split, fraction in SPLITS.items()}
        counts = dict.fromkeys(SPLITS, 0)
        for index, block in enumerate(blocks):
            missing = [split for split, count in counts.items() if not count]
            choices = missing if len(blocks) - index == len(missing) else SPLITS
            split = min(choices, key=lambda name: (
                (counts[name] + len(block) - targets[name]) ** 2 -
                (counts[name] - targets[name]) ** 2
            ) / targets[name])
            counts[split] += len(block)
            for row in block:
                row["split"] = split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows, damaged, auxiliary, thumbnails = inspect(args.root)
    groups, edges = cluster(rows, thumbnails)
    partition(groups)
    for group in groups:
        assert len({row["split"] for row in group} & SPLITS.keys()) <= 1

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "manifest.csv", rows, list(rows[0]))
    write_csv(out / "damaged.csv", damaged, ["path", "error"])
    write_csv(out / "similar_pairs.csv", edges, ["path_a", "path_b", "distance", "rmse", "same_class"])
    (out / "auxiliary.json").write_text(json.dumps(auxiliary, indent=2), encoding="utf-8")
    digest_counts = Counter(row["pixel_sha256"] for row in rows)
    csv_path = args.root / "dataset" / "datos.csv"
    metadata = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            metadata["columns"] = reader.fieldnames
            labels = list(reader)
        expected = {row["path"]: row["class"] for row in rows}
        metadata["rows"] = len(labels)
        metadata["unique_paths"] = len({row["imagen"] for row in labels})
        metadata["label_mismatches"] = sum(
            [label for label in CLASSES if row[label] == "1"] != [expected.get(row["imagen"])]
            for row in labels
        )
    summary = {
        "valid_images": len(rows), "damaged_images": len(damaged), "auxiliary_files": len(auxiliary),
        "classes": dict(Counter(row["class"] for row in rows)),
        "formats": dict(Counter(row["format"] for row in rows)),
        "dimensions": dict(Counter(f'{row["width"]}x{row["height"]}' for row in rows)),
        "modes": dict(Counter(row["mode"] for row in rows)),
        "exact_extra_files": len(rows) - len({row["sha256"] for row in rows}),
        "pixel_duplicate_groups": sum(count > 1 for count in digest_counts.values()),
        "pixel_extra_images": len(rows) - len(digest_counts),
        "similar_pairs_distinct_pixels": len(edges),
        "similar_pairs_cross_class": sum(not edge["same_class"] for edge in edges),
        "groups": len(groups), "largest_group": max(map(len, groups)),
        "seed": SEED, "dhash_threshold": DISTANCE, "thumbnail_rmse_threshold": MAX_RMSE,
        "images_with_exif_tags": sum(bool(row["exif_tags"]) for row in rows),
        "dataset_csv": metadata,
        "split_by_class": {split: dict(Counter(row["class"] for row in rows if row["split"] == split))
                           for split in (*SPLITS, "duplicate", "review_conflict")},
        "manifest_sha256": sha256((out / "manifest.csv").read_bytes()).hexdigest(),
    }
    (out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
