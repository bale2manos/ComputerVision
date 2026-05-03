from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "Jersey crop reviewer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review jersey OCR crops interactively and build a YOLO classification dataset. "
            "Press Enter to accept the proposed label, type digits to correct it, or skip bad crops."
        )
    )
    parser.add_argument("--crops-dir", required=True, help="Directory containing saved jersey crops from extract_jersey_numbers.py.")
    parser.add_argument("--jersey-report", default=None, help="Optional jersey_numbers.json to prefill predictions by player_id.")
    parser.add_argument("--output-dir", default="datasets/jersey_cls", help="Classification dataset root.")
    parser.add_argument("--split", choices=["train", "val", "auto"], default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--unknown-label", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    crops_dir = resolve_path(args.crops_dir)
    if not crops_dir.exists():
        raise SystemExit(f"Crops directory not found: {crops_dir}")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "jersey_review_manifest.csv"
    manifest_exists = manifest_path.exists()

    images = discover_images(crops_dir)
    if args.shuffle:
        random.shuffle(images)
    if args.max_crops > 0:
        images = images[: args.max_crops]
    if args.start_index > 0:
        images = images[args.start_index :]
    if not images:
        raise SystemExit("No crop images found.")

    predictions = load_predictions(resolve_path(args.jersey_report) if args.jersey_report else None)
    state: dict[str, Any] = {
        "args": args,
        "images": images,
        "index": 0,
        "output_dir": output_dir,
        "crops_dir": crops_dir,
        "predictions": predictions,
        "typed": "",
        "last_saved": [],
        "manifest_writer": None,
    }

    print_controls()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "source", "split", "label", "accepted_prediction", "player_id", "variant"],
        )
        if not manifest_exists:
            writer.writeheader()
        state["manifest_writer"] = writer
        while 0 <= state["index"] < len(images):
            draw_current(state)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("n"), ord(" ")):
                state["index"] += 1
                state["typed"] = ""
                continue
            if key == ord("p"):
                state["index"] = max(0, state["index"] - 1)
                state["typed"] = ""
                continue
            if key == ord("j"):
                state["index"] = max(0, state["index"] - 25)
                state["typed"] = ""
                continue
            if key == ord("l"):
                state["index"] = min(len(images) - 1, state["index"] + 25)
                state["typed"] = ""
                continue
            if key == ord("u"):
                undo_last(state)
                continue
            if key == ord("x"):
                state["typed"] = args.unknown_label
                save_current(state, accepted_prediction=False)
                state["index"] += 1
                state["typed"] = ""
                continue
            if key in (8, 127):
                state["typed"] = state["typed"][:-1]
                continue
            if key in (13, 10):
                label = state["typed"] or prediction_for_current(state) or ""
                if not is_valid_label(label, args.unknown_label):
                    print(f"[warn] Invalid label: {label!r}. Type 0-99, x for unknown, or n to skip.")
                    continue
                state["typed"] = normalize_label(label, args.unknown_label)
                save_current(state, accepted_prediction=not bool(state["typed"] == args.unknown_label) and state["typed"] == prediction_for_current(state))
                state["index"] += 1
                state["typed"] = ""
                continue
            if ord("0") <= key <= ord("9"):
                if len(state["typed"]) < 2:
                    state["typed"] += chr(key)
                continue

    cv2.destroyWindow(WINDOW)
    print(f"Dataset: {output_dir}")
    print(f"Manifest: {manifest_path}")


def discover_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMG_EXTS)


def load_predictions(report_path: Path | None) -> dict[str, str]:
    if report_path is None or not report_path.exists():
        return {}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    by_player: dict[str, str] = {}
    for player in data.get("players", []):
        number = player.get("jersey_number")
        if number is not None and player.get("player_id") is not None:
            by_player[str(player["player_id"])] = str(number)
    return by_player


def prediction_for_current(state: dict[str, Any]) -> str | None:
    path = state["images"][state["index"]]
    player_id = parse_player_id(path)
    if player_id is None:
        return None
    return state["predictions"].get(str(player_id))


def parse_player_id(path: Path) -> str | None:
    # Handles directories like player_03 and filenames containing _p3_ or p3.
    for part in reversed(path.parts):
        if part.startswith("player_"):
            value = part.replace("player_", "").lstrip("0") or "0"
            return value if value.isdigit() else None
    stem = path.stem
    for token in stem.replace("-", "_").split("_"):
        if token.startswith("p") and token[1:].isdigit():
            return token[1:]
    return None


def save_current(state: dict[str, Any], accepted_prediction: bool) -> None:
    src = state["images"][state["index"]]
    label = normalize_label(state["typed"], state["args"].unknown_label)
    split = choose_split(state, src, label)
    out_dir = state["output_dir"] / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / make_unique_name(src, state["crops_dir"])
    shutil.copy2(src, dst)
    state["last_saved"] = [dst]
    player_id = parse_player_id(src)
    state["manifest_writer"].writerow(
        {
            "file": str(dst.relative_to(state["output_dir"])),
            "source": str(src),
            "split": split,
            "label": label,
            "accepted_prediction": bool(accepted_prediction),
            "player_id": player_id,
            "variant": guess_variant(src),
        }
    )
    print(f"[saved] {src.name} -> {split}/{label}")


def undo_last(state: dict[str, Any]) -> None:
    removed = 0
    for path in state.get("last_saved", []):
        try:
            Path(path).unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    state["last_saved"] = []
    print(f"[undo] removed {removed} files. Manifest is append-only; ignore undone rows if needed.")


def draw_current(state: dict[str, Any]) -> None:
    path = state["images"][state["index"]]
    image = cv2.imread(str(path))
    if image is None:
        state["index"] += 1
        return
    canvas = make_canvas(image)
    pred = prediction_for_current(state)
    typed = state["typed"]
    player_id = parse_player_id(path)
    lines = [
        f"{state['index'] + 1}/{len(state['images'])} | {path.name}",
        f"player_id: {player_id or '?'} | EasyOCR prediction: {pred or '-'} | typed: {typed or '<enter=accept pred>'}",
        "Type 0-99 then Enter. Enter accepts pred. x=unknown/bad, n=skip, p=prev, j/l jump, u=undo, q=quit.",
    ]
    draw_panel(canvas, lines)
    cv2.imshow(WINDOW, canvas)


def make_canvas(image: np.ndarray) -> np.ndarray:
    max_side = 680
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 5.0)
    if scale > 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        image = image.copy()
    pad_top = 115
    canvas = np.zeros((image.shape[0] + pad_top, max(image.shape[1], 900), 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)
    canvas[pad_top : pad_top + image.shape[0], 0 : image.shape[1]] = image
    return canvas


def draw_panel(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28
    for line in lines:
        cv2.putText(frame, line, (14, y), font, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
        y += 30


def choose_split(state: dict[str, Any], src: Path, label: str) -> str:
    args = state["args"]
    if args.split != "auto":
        return args.split
    key = f"{src}-{label}"
    return "val" if stable_ratio(key) < float(args.val_ratio) else "train"


def stable_ratio(value: str) -> float:
    return (abs(hash(value)) % 10000) / 10000.0


def make_unique_name(src: Path, root: Path) -> str:
    try:
        rel = src.relative_to(root)
        prefix = "__".join(rel.parts[:-1])
    except ValueError:
        prefix = src.parent.name
    base = f"{prefix}__{src.name}" if prefix else src.name
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)


def guess_variant(src: Path) -> str | None:
    stem = src.stem
    for name in ("raw", "rgb", "gray", "otsu", "inv", "adaptive"):
        if stem.endswith("_" + name) or f"_{name}_" in stem:
            return name
    return None


def normalize_label(value: str, unknown_label: str) -> str:
    value = str(value).strip().lower()
    if value in {"", "?", "unknown", "unk", "bad", "none"}:
        return unknown_label
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) == 2 and digits.startswith("0"):
        digits = digits[1:]
    return digits


def is_valid_label(value: str, unknown_label: str) -> bool:
    value = normalize_label(value, unknown_label)
    if value == unknown_label:
        return True
    if not value.isdigit() or len(value) > 2:
        return False
    number = int(value)
    return 0 <= number <= 99


def print_controls() -> None:
    print("Controls:")
    print("  digits + Enter: save corrected jersey number")
    print("  Enter: accept EasyOCR prediction when available")
    print("  x: save as unknown/bad crop")
    print("  n or space: skip")
    print("  p: previous")
    print("  j/l: jump -/+25 crops")
    print("  u: undo last copied file")
    print("  q/esc: quit")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
