from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize reviewed jersey OCR labels and estimate whether it is worth fine-tuning an adapted OCR."
    )
    parser.add_argument("--input-root", default="datasets/jersey_ocr_labeled")
    parser.add_argument("--min-useful", type=int, default=300)
    parser.add_argument("--good", type=int, default=1000)
    parser.add_argument("--ideal", type=int, default=3000)
    parser.add_argument("--min-per-label", type=int, default=20)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    summary = summarize(root, args)
    print_human_summary(summary)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def summarize(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    labels_path = root / "labels.txt"
    items = []
    missing = 0
    invalid = 0
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or "\t" not in line:
                continue
            rel, raw_label = line.split("\t", 1)
            label = normalize_label(raw_label)
            if not label:
                invalid += 1
                continue
            image = root / rel
            if not image.exists() or image.suffix.lower() not in IMG_EXTS:
                missing += 1
                continue
            items.append((image, label))
    counts = Counter(label for _image, label in items)
    label_count = len(counts)
    total = len(items)
    labels_below_target = {label: n for label, n in sorted(counts.items(), key=lambda kv: int(kv[0])) if n < args.min_per_label}
    readiness = readiness_level(total, label_count, counts, args)
    recommendation = recommendation_text(readiness, total, label_count, labels_below_target, args)
    return {
        "input_root": str(root),
        "labels_file": str(labels_path),
        "total_valid_samples": total,
        "labels_count": label_count,
        "counts_by_label": dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
        "missing_image_rows": missing,
        "invalid_label_rows": invalid,
        "thresholds": {
            "min_useful": args.min_useful,
            "good": args.good,
            "ideal": args.ideal,
            "min_per_label": args.min_per_label,
        },
        "labels_below_min_per_label": labels_below_target,
        "readiness": readiness,
        "recommendation": recommendation,
        "next_prepare_command": (
            f"python tools\\prepare_jersey_ocr_dataset.py --input-root \"{root}\" "
            f"--output-root \"datasets\\jersey_ocr_paddle\" --val-ratio 0.15"
        ),
    }


def normalize_label(label: str) -> str:
    digits = "".join(ch for ch in str(label).strip() if ch.isdigit())
    if len(digits) == 2 and digits.startswith("0"):
        digits = digits[1:]
    if not digits or len(digits) > 2:
        return ""
    return digits


def readiness_level(total: int, label_count: int, counts: Counter[str], args: argparse.Namespace) -> str:
    if total < args.min_useful:
        return "too_small"
    if total < args.good:
        return "minimum_experiment"
    if total < args.ideal:
        return "good_for_finetuning"
    low_labels = [label for label, n in counts.items() if n < args.min_per_label]
    if low_labels and label_count > 4:
        return "good_but_imbalanced"
    return "strong"


def recommendation_text(
    readiness: str,
    total: int,
    label_count: int,
    low_labels: dict[str, int],
    args: argparse.Namespace,
) -> str:
    if readiness == "too_small":
        return (
            f"Todavía no compensa entrenar OCR adaptado. Hay {total} crops válidos; intenta llegar al menos a "
            f"{args.min_useful}, mejor {args.good}+ con varios vídeos."
        )
    if readiness == "minimum_experiment":
        return (
            f"Ya se puede hacer una prueba pequeña de fine-tuning, pero será frágil. Hay {total} crops en "
            f"{label_count} dorsales; úsalo como experimento, no como modelo final."
        )
    if readiness == "good_for_finetuning":
        return (
            f"Ya compensa preparar dataset y probar fine-tuning OCR. Hay {total} crops; revisa balance por dorsal "
            f"y añade más ejemplos de los números flojos."
        )
    if readiness == "good_but_imbalanced":
        lows = ", ".join(f"{label}:{n}" for label, n in low_labels.items())
        return (
            f"Cantidad suficiente, pero el dataset está desbalanceado. Antes de entrenar en serio, añade ejemplos de: {lows}."
        )
    return f"Dataset fuerte para intentar fine-tuning OCR adaptado: {total} crops válidos y {label_count} dorsales."


def print_human_summary(summary: dict[str, Any]) -> None:
    print("\n=== Jersey OCR dataset readiness ===")
    print(f"Dataset: {summary['input_root']}")
    print(f"Valid samples: {summary['total_valid_samples']}")
    print(f"Distinct jersey labels: {summary['labels_count']}")
    print(f"Readiness: {summary['readiness']}")
    print("\nCounts by label:")
    if summary["counts_by_label"]:
        for label, count in summary["counts_by_label"].items():
            print(f"  {label:>2}: {count}")
    else:
        print("  <empty>")
    if summary["missing_image_rows"] or summary["invalid_label_rows"]:
        print("\nWarnings:")
        print(f"  missing image rows: {summary['missing_image_rows']}")
        print(f"  invalid label rows: {summary['invalid_label_rows']}")
    print("\nRecommendation:")
    print("  " + summary["recommendation"])
    print("\nPrepare command when ready:")
    print("  " + summary["next_prepare_command"])


if __name__ == "__main__":
    main()
