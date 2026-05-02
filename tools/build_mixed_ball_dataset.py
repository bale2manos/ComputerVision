from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple YOLO ball datasets into one single-class dataset.")
    parser.add_argument("--sources", nargs="+", required=True, help="Dataset roots to merge, e.g. datasets/ball datasets/roboflow_ball")
    parser.add_argument("--output", required=True, help="Output dataset root, e.g. datasets/ball_mixed")
    parser.add_argument("--reset", action="store_true", help="Delete output dataset first if it exists.")
    parser.add_argument("--val-ratio", type=float, default=0.18, help="Used only for sources without an explicit val/valid split.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--class-id", type=int, default=0, help="Final single class id. Defaults to 0 = ball.")
    parser.add_argument("--class-name", default="ball")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output = Path(args.output)
    if args.reset and output.exists():
        shutil.rmtree(output)

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for source_value in args.sources:
        source = Path(source_value)
        if not source.exists():
            print(f"[WARN] Source not found: {source}")
            continue
        split_dirs = find_split_dirs(source)
        if split_dirs:
            for image_dir, label_dir, split in split_dirs:
                copied = copy_split(source, image_dir, label_dir, output, split, args.class_id)
                counts[split] += copied
                print(f"[OK] {source} {split}: {copied} images")
        else:
            copied_counts = copy_flat_source(source, output, args.val_ratio, args.class_id)
            counts["train"] += copied_counts["train"]
            counts["val"] += copied_counts["val"]
            print(f"[OK] {source} flat: {copied_counts}")

    data_yaml = (
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"  0: {args.class_name}\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    print(f"\nMerged dataset: {output}")
    print(f"Counts: {counts}")
    print(f"data.yaml: {output / 'data.yaml'}")


def find_split_dirs(root: Path) -> list[tuple[Path, Path, str]]:
    dirs: list[tuple[Path, Path, str]] = []

    # Ultralytics layout: root/images/train + root/labels/train.
    if (root / "images").exists() and (root / "labels").exists():
        for source_split, out_split in (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "val")):
            image_dir = root / "images" / source_split
            label_dir = root / "labels" / source_split
            if image_dir.exists() and label_dir.exists():
                dirs.append((image_dir, label_dir, out_split))

    # Roboflow layout: root/train/images + root/train/labels.
    for source_split, out_split in (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "val")):
        image_dir = root / source_split / "images"
        label_dir = root / source_split / "labels"
        if image_dir.exists() and label_dir.exists():
            dirs.append((image_dir, label_dir, out_split))

    return dedupe_dirs(dirs)


def dedupe_dirs(dirs: list[tuple[Path, Path, str]]) -> list[tuple[Path, Path, str]]:
    seen = set()
    output = []
    for image_dir, label_dir, split in dirs:
        key = (image_dir.resolve(), label_dir.resolve(), split)
        if key in seen:
            continue
        seen.add(key)
        output.append((image_dir, label_dir, split))
    return output


def copy_split(source: Path, image_dir: Path, label_dir: Path, output: Path, split: str, class_id: int) -> int:
    count = 0
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        prefix = safe_prefix(source.name, split)
        dst_image = unique_path(output / "images" / split / f"{prefix}_{image_path.name}")
        dst_label = output / "labels" / split / f"{dst_image.stem}.txt"
        shutil.copy2(image_path, dst_image)
        normalize_label(label_path, dst_label, class_id)
        count += 1
    return count


def copy_flat_source(source: Path, output: Path, val_ratio: float, class_id: int) -> dict[str, int]:
    images = [p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS]
    random.shuffle(images)
    counts = {"train": 0, "val": 0}
    for image_path in images:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            candidate = source / "labels" / f"{image_path.stem}.txt"
            if candidate.exists():
                label_path = candidate
            else:
                continue
        split = "val" if random.random() < val_ratio else "train"
        prefix = safe_prefix(source.name, split)
        dst_image = unique_path(output / "images" / split / f"{prefix}_{image_path.name}")
        dst_label = output / "labels" / split / f"{dst_image.stem}.txt"
        shutil.copy2(image_path, dst_image)
        normalize_label(label_path, dst_label, class_id)
        counts[split] += 1
    return counts


def normalize_label(src: Path, dst: Path, class_id: int) -> None:
    lines_out = []
    for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        parts[0] = str(class_id)
        lines_out.append(" ".join(parts[:5]))
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def safe_prefix(name: str, split: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    return f"{clean}_{split}"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


if __name__ == "__main__":
    main()
