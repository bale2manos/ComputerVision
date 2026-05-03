from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIGITS = "0123456789"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a reviewed jersey OCR dataset for future OCR fine-tuning. "
            "Input is the labels.txt produced by review_jersey_identities.py: images/file.jpg<TAB>11. "
            "Output includes PaddleOCR-style train/val label lists plus a digit dictionary."
        )
    )
    parser.add_argument("--input-root", default="datasets/jersey_ocr_labeled")
    parser.add_argument("--output-root", default="datasets/jersey_ocr_paddle")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-label-count", type=int, default=1, help="Drop labels with fewer examples than this.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--copy-images", action="store_true", default=True)
    parser.add_argument("--no-copy-images", action="store_false", dest="copy_images")
    parser.add_argument("--dedupe", action="store_true", default=True)
    parser.add_argument("--no-dedupe", action="store_false", dest="dedupe")
    parser.add_argument("--max-per-label", type=int, default=0, help="Optional cap per jersey number to reduce imbalance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    labels_path = input_root / "labels.txt"
    if not labels_path.exists():
        raise SystemExit(f"Missing labels file: {labels_path}")

    items = load_items(input_root, labels_path)
    if args.dedupe:
        items = dedupe_items(items)
    items = filter_and_balance(items, args.min_label_count, args.max_per_label)
    if not items:
        raise SystemExit("No valid OCR samples after filtering.")

    train, val = stratified_split(items, args.val_ratio, args.seed)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        train = copy_items(train, output_root / "images" / "train", output_root)
        val = copy_items(val, output_root / "images" / "val", output_root)
    else:
        train = absolutize_items(train)
        val = absolutize_items(val)

    write_label_list(output_root / "train_list.txt", train)
    write_label_list(output_root / "val_list.txt", val)
    write_dict(output_root / "digit_dict.txt")
    write_manifest(output_root / "manifest.csv", train, val)
    summary = build_summary(train, val, input_root, output_root)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nNext step: use train_list.txt, val_list.txt and digit_dict.txt from", output_root)


def load_items(input_root: Path, labels_path: Path) -> list[dict[str, Any]]:
    items = []
    for line_no, line in enumerate(labels_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or "\t" not in line:
            continue
        rel, label = line.split("\t", 1)
        label = normalize_label(label)
        if not label:
            continue
        image_path = input_root / rel
        if image_path.suffix.lower() not in IMG_EXTS or not image_path.exists():
            continue
        items.append({"source": image_path, "label": label, "line_no": line_no})
    return items


def normalize_label(label: str) -> str:
    digits = "".join(ch for ch in str(label).strip() if ch.isdigit())
    if len(digits) == 2 and digits.startswith("0"):
        digits = digits[1:]
    if not digits or len(digits) > 2:
        return ""
    return digits


def dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for item in items:
        key = (str(item["source"].resolve()), item["label"])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def filter_and_balance(items: list[dict[str, Any]], min_label_count: int, max_per_label: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_label.setdefault(item["label"], []).append(item)
    output = []
    for label, label_items in sorted(by_label.items(), key=lambda kv: int(kv[0])):
        if len(label_items) < min_label_count:
            continue
        random.shuffle(label_items)
        if max_per_label > 0:
            label_items = label_items[:max_per_label]
        output.extend(label_items)
    random.shuffle(output)
    return output


def stratified_split(items: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_label.setdefault(item["label"], []).append(item)
    train, val = [], []
    for label_items in by_label.values():
        rng.shuffle(label_items)
        if len(label_items) <= 2:
            train.extend(label_items)
            continue
        val_count = max(1, int(round(len(label_items) * val_ratio)))
        val.extend(label_items[:val_count])
        train.extend(label_items[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def copy_items(items: list[dict[str, Any]], dst_dir: Path, output_root: Path) -> list[dict[str, Any]]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for idx, item in enumerate(items):
        src: Path = item["source"]
        label = item["label"]
        dst_name = safe_name(f"{label}_{idx:06d}_{src.name}")
        dst = dst_dir / dst_name
        shutil.copy2(src, dst)
        copied.append({"path": dst.relative_to(output_root).as_posix(), "label": label, "source": str(src)})
    return copied


def absolutize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"path": str(item["source"].resolve()), "label": item["label"], "source": str(item["source"])} for item in items]


def write_label_list(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text("".join(f"{item['path']}\t{item['label']}\n" for item in items), encoding="utf-8")


def write_dict(path: Path) -> None:
    path.write_text("\n".join(DIGITS) + "\n", encoding="utf-8")


def write_manifest(path: Path, train: list[dict[str, Any]], val: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "file", "label", "source"])
        writer.writeheader()
        for split, items in (("train", train), ("val", val)):
            for item in items:
                writer.writerow({"split": split, "file": item["path"], "label": item["label"], "source": item.get("source")})


def build_summary(train: list[dict[str, Any]], val: list[dict[str, Any]], input_root: Path, output_root: Path) -> dict[str, Any]:
    train_counts = Counter(item["label"] for item in train)
    val_counts = Counter(item["label"] for item in val)
    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "train_total": len(train),
        "val_total": len(val),
        "labels_total": len(set(train_counts) | set(val_counts)),
        "train_counts": dict(sorted(train_counts.items(), key=lambda kv: int(kv[0]))),
        "val_counts": dict(sorted(val_counts.items(), key=lambda kv: int(kv[0]))),
        "files": {
            "train_list": str(output_root / "train_list.txt"),
            "val_list": str(output_root / "val_list.txt"),
            "dict": str(output_root / "digit_dict.txt"),
            "manifest": str(output_root / "manifest.csv"),
        },
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


if __name__ == "__main__":
    main()
