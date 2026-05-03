from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "Jersey identity reviewer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Human-in-the-loop jersey identity review. Shows the best jersey crops for each "
            "player/track, lets you accept or correct the OCR prediction, writes identity_overrides.json "
            "for this clip, and saves corrected crop labels for future OCR fine-tuning."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--jersey-report", default=None, help="jersey_numbers.json from extract_jersey_numbers.py")
    parser.add_argument("--crops-dir", default=None, help="Crops directory from extract_jersey_numbers.py --save-crops")
    parser.add_argument("--output-overrides", required=True, help="identity_overrides.json to write")
    parser.add_argument("--labeled-crops-dir", default="datasets/jersey_ocr_labeled", help="PaddleOCR-style labeled crop dataset")
    parser.add_argument("--group-by", choices=["player", "track"], default="player")
    parser.add_argument("--samples-per-group", type=int, default=12)
    parser.add_argument("--min-box-height", type=float, default=90.0)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--include-unknown-team", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--append", action="store_true", help="Append to an existing overrides/review dataset instead of replacing in memory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = resolve_path(args.video)
    tracks_path = resolve_path(args.tracks)
    output_overrides = resolve_path(args.output_overrides)
    labeled_root = resolve_path(args.labeled_crops_dir)
    crops_dir = resolve_path(args.crops_dir) if args.crops_dir else None
    report_path = resolve_path(args.jersey_report) if args.jersey_report else None

    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    jersey_report = load_json(report_path) if report_path and report_path.exists() else {}

    groups = build_review_groups(records, jersey_report, crops_dir, args)
    if args.start_index:
        groups = groups[args.start_index :]
    if args.max_groups > 0:
        groups = groups[: args.max_groups]
    if not groups:
        raise SystemExit("No reviewable jersey groups found.")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    state: dict[str, Any] = {
        "args": args,
        "video": video,
        "records": records,
        "groups": groups,
        "index": 0,
        "typed": "",
        "cap": cap,
        "output_overrides": output_overrides,
        "labeled_root": labeled_root,
        "overrides": load_existing_overrides(output_overrides) if args.append else {"track_overrides": [], "reviewed_groups": []},
        "last_action": None,
    }

    prepare_labeled_dataset(labeled_root)
    print_controls()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    while 0 <= state["index"] < len(groups):
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
            state["index"] = max(0, state["index"] - 10)
            state["typed"] = ""
            continue
        if key == ord("l"):
            state["index"] = min(len(groups) - 1, state["index"] + 10)
            state["typed"] = ""
            continue
        if key == ord("u"):
            undo_last(state)
            continue
        if key in (8, 127):
            state["typed"] = state["typed"][:-1]
            continue
        if ord("0") <= key <= ord("9"):
            if len(state["typed"]) < 2:
                state["typed"] += chr(key)
            continue
        if key == ord("x"):
            save_review(state, label=None, status="unknown_or_bad")
            state["index"] += 1
            state["typed"] = ""
            continue
        if key == ord("r"):
            save_review(state, label=None, status="review_later")
            state["index"] += 1
            state["typed"] = ""
            continue
        if key in (13, 10):
            group = groups[state["index"]]
            label = state["typed"] or group.get("prediction") or ""
            if not is_valid_number(label):
                print(f"[warn] Invalid dorsal {label!r}. Type 0-99, press Enter, or x for unknown/bad.")
                continue
            save_review(state, label=normalize_number(label), status="confirmed")
            state["index"] += 1
            state["typed"] = ""
            continue

    write_overrides(state)
    cap.release()
    cv2.destroyWindow(WINDOW)
    print(f"Overrides: {output_overrides}")
    print(f"Labeled OCR crops: {labeled_root}")


def build_review_groups(
    records: list[dict[str, Any]],
    jersey_report: dict[str, Any],
    crops_dir: Path | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person":
            continue
        if rec.get("role") == "referee" or rec.get("team") == "referee":
            continue
        if not args.include_unknown_team and rec.get("team") in (None, "unknown"):
            continue
        if rec.get("on_court") is False or rec.get("bottom_truncated"):
            continue
        if float(rec.get("confidence") or 0.0) < args.min_confidence:
            continue
        bbox = rec.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if float(bbox[3]) - float(bbox[1]) < args.min_box_height:
            continue
        key = group_key(rec, args.group_by)
        if key is None:
            continue
        grouped_records[key].append(rec)

    player_predictions = report_predictions_by_player(jersey_report)
    segment_predictions = report_predictions_by_track(jersey_report)
    crop_index = index_crop_paths(crops_dir) if crops_dir else {"player": defaultdict(list), "track": defaultdict(list)}

    groups = []
    for key, group_records in grouped_records.items():
        group_records = sorted(group_records, key=lambda rec: int(rec.get("frame_index", 0)))
        team = majority_value(rec.get("team") for rec in group_records)
        track_ids = sorted({int(rec["track_id"]) for rec in group_records if rec.get("track_id") is not None})
        player_ids = sorted({int(rec["player_id"]) for rec in group_records if rec.get("player_id") is not None})
        first_frame = int(group_records[0].get("frame_index", 0))
        last_frame = int(group_records[-1].get("frame_index", 0))
        prediction = None
        if args.group_by == "player" and player_ids:
            prediction = player_predictions.get(str(player_ids[0]))
        if prediction is None and track_ids:
            prediction = segment_predictions.get(str(track_ids[0]))
        crop_paths = []
        if args.group_by == "player":
            for player_id in player_ids:
                crop_paths.extend(crop_index["player"].get(str(player_id), []))
        else:
            for track_id in track_ids:
                crop_paths.extend(crop_index["track"].get(str(track_id), []))
        groups.append(
            {
                "key": key,
                "group_by": args.group_by,
                "team": team,
                "track_ids": track_ids,
                "player_ids": player_ids,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "records": group_records,
                "prediction": prediction,
                "crop_paths": unique_paths(crop_paths),
                "vote_summary": vote_summary_for_group(jersey_report, player_ids, track_ids),
            }
        )

    def sort_key(group: dict[str, Any]) -> tuple[int, int, str]:
        pred_missing = 0 if group.get("prediction") else 1
        return (pred_missing, int(group.get("first_frame") or 0), str(group.get("key")))

    return sorted(groups, key=sort_key)


def group_key(rec: dict[str, Any], group_by: str) -> str | None:
    if group_by == "track":
        return f"track_{rec.get('track_id')}" if rec.get("track_id") is not None else None
    return f"player_{rec.get('player_id')}" if rec.get("player_id") is not None else None


def report_predictions_by_player(report: dict[str, Any]) -> dict[str, str]:
    output = {}
    for player in report.get("players", []) or []:
        if player.get("player_id") is not None and player.get("jersey_number") is not None:
            output[str(player["player_id"])] = str(player["jersey_number"])
    return output


def report_predictions_by_track(report: dict[str, Any]) -> dict[str, str]:
    output = {}
    for segment in report.get("identity_segments", []) or []:
        if segment.get("track_id") is not None and segment.get("jersey_number") is not None:
            output[str(segment["track_id"])] = str(segment["jersey_number"])
    return output


def vote_summary_for_group(report: dict[str, Any], player_ids: list[int], track_ids: list[int]) -> dict[str, Any]:
    summary = {"players": {}, "tracks": {}}
    for player in report.get("players", []) or []:
        if player.get("player_id") is not None and int(player["player_id"]) in player_ids:
            summary["players"][str(player["player_id"])] = {
                "jersey_number": player.get("jersey_number"),
                "frame_votes": player.get("frame_votes", {}),
                "score_by_number": player.get("score_by_number", {}),
            }
    for segment in report.get("identity_segments", []) or []:
        if segment.get("track_id") is not None and int(segment["track_id"]) in track_ids:
            summary["tracks"][str(segment["track_id"])] = {
                "jersey_number": segment.get("jersey_number"),
                "frame_votes": segment.get("frame_votes", {}),
                "score_by_number": segment.get("score_by_number", {}),
            }
    return summary


def index_crop_paths(crops_dir: Path) -> dict[str, defaultdict[str, list[Path]]]:
    index = {"player": defaultdict(list), "track": defaultdict(list)}
    for path in sorted(p for p in crops_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
        player_id = parse_player_id(path)
        track_id = parse_track_id(path)
        if player_id is not None:
            index["player"][str(player_id)].append(path)
        if track_id is not None:
            index["track"][str(track_id)].append(path)
    return index


def parse_player_id(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part.startswith("player_"):
            value = part.replace("player_", "").lstrip("0") or "0"
            return value if value.isdigit() else None
    for token in path.stem.replace("-", "_").split("_"):
        if token.startswith("p") and token[1:].isdigit():
            return token[1:]
    return None


def parse_track_id(path: Path) -> str | None:
    for token in path.stem.replace("-", "_").split("_"):
        if token.startswith("t") and token[1:].isdigit():
            return token[1:]
    return None


def draw_current(state: dict[str, Any]) -> None:
    group = state["groups"][state["index"]]
    thumbs = collect_thumbnails(group, state)
    canvas = make_montage(thumbs)
    typed = state["typed"]
    pred = group.get("prediction")
    vote_text = compact_vote_text(group.get("vote_summary", {}))
    lines = [
        f"{state['index'] + 1}/{len(state['groups'])} | {group['key']} | team={group.get('team')} | frames={group.get('first_frame')}-{group.get('last_frame')}",
        f"tracks={group.get('track_ids')} | players={group.get('player_ids')} | OCR pred={pred or '-'} | typed={typed or '<enter=accept pred>'}",
        f"votes: {vote_text}",
        "digits+Enter=confirm/correct | Enter=accept OCR pred | x=unknown/bad | n=skip | p=prev | j/l jump | u=undo | q=quit",
    ]
    draw_panel(canvas, lines)
    cv2.imshow(WINDOW, canvas)


def collect_thumbnails(group: dict[str, Any], state: dict[str, Any]) -> list[np.ndarray]:
    crop_paths = group.get("crop_paths", [])[: state["args"].samples_per_group]
    thumbs = []
    for path in crop_paths:
        img = cv2.imread(str(path))
        if img is not None:
            thumbs.append(img)
    if len(thumbs) >= state["args"].samples_per_group:
        return thumbs[: state["args"].samples_per_group]

    needed = state["args"].samples_per_group - len(thumbs)
    generated = generate_crops_from_records(group.get("records", []), state["cap"], needed)
    thumbs.extend(generated)
    return thumbs


def generate_crops_from_records(records: list[dict[str, Any]], cap: cv2.VideoCapture, max_crops: int) -> list[np.ndarray]:
    if max_crops <= 0:
        return []
    ordered = sorted(
        records,
        key=lambda rec: (float(rec.get("bbox_area") or 0.0), float(rec.get("confidence") or 0.0)),
        reverse=True,
    )
    thumbs = []
    used_frames: list[int] = []
    for rec in ordered:
        frame = int(rec.get("frame_index", 0))
        if any(abs(frame - prev) < 8 for prev in used_frames):
            continue
        crop = read_jersey_crop(cap, frame, rec.get("bbox"))
        if crop is not None:
            thumbs.append(crop)
            used_frames.append(frame)
        if len(thumbs) >= max_crops:
            break
    return thumbs


def read_jersey_crop(cap: cv2.VideoCapture, frame_index: int, bbox: Any) -> np.ndarray | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    box_w = x2 - x1
    box_h = y2 - y1
    cx1 = x1 + int(0.12 * box_w)
    cx2 = x2 - int(0.12 * box_w)
    cy1 = y1 + int(0.08 * box_h)
    cy2 = y1 + int(0.68 * box_h)
    crop = frame[max(0, cy1) : min(h, cy2), max(0, cx1) : min(w, cx2)]
    return crop if crop.size else None


def make_montage(images: list[np.ndarray]) -> np.ndarray:
    thumb_w, thumb_h = 190, 230
    cols = 4
    rows = max(1, int(np.ceil(max(len(images), 1) / cols)))
    header_h = 130
    canvas = np.zeros((header_h + rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)
    if not images:
        cv2.putText(canvas, "No jersey crops available for this group", (20, header_h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        return canvas
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = c * thumb_w
        y = header_h + r * thumb_h
        thumb = fit_image(img, thumb_w - 12, thumb_h - 28)
        canvas[y + 24 : y + 24 + thumb.shape[0], x + 6 : x + 6 + thumb.shape[1]] = thumb
        cv2.putText(canvas, str(idx + 1), (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
    return canvas


def fit_image(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((max_h, max_w, 3), dtype=np.uint8)
    scale = min(max_w / w, max_h / h)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    out = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    out[:] = (45, 45, 45)
    y = (max_h - resized.shape[0]) // 2
    x = (max_w - resized.shape[1]) // 2
    out[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return out


def draw_panel(canvas: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
        y += 29


def save_review(state: dict[str, Any], label: str | None, status: str) -> None:
    group = state["groups"][state["index"]]
    output = state["overrides"]
    action = {"override_added": None, "copied": [], "review_index": len(output.get("reviewed_groups", []))}

    review = {
        "key": group["key"],
        "group_by": group["group_by"],
        "status": status,
        "confirmed_jersey_number": label,
        "prediction": group.get("prediction"),
        "team": group.get("team"),
        "track_ids": group.get("track_ids", []),
        "player_ids": group.get("player_ids", []),
        "first_frame": group.get("first_frame"),
        "last_frame": group.get("last_frame"),
    }
    output.setdefault("reviewed_groups", []).append(review)

    if label is not None:
        override = {
            "track_ids": group.get("track_ids", []),
            "first_frame": int(group.get("first_frame", 0)),
            "last_frame": int(group.get("last_frame", 1_000_000)),
            "team": group.get("team"),
            "jersey_number": str(label),
            "jersey_identity": f"{group.get('team')}_{label}" if group.get("team") else None,
            "canonical_player_id": group.get("player_ids", [None])[0] if group.get("player_ids") else None,
            "reason": f"manual_jersey_review_{group['group_by']}",
        }
        output.setdefault("track_overrides", []).append(override)
        action["override_added"] = override
        action["copied"] = save_labeled_crops(state, group, str(label))

    state["last_action"] = action
    write_overrides(state)
    print(f"[review] {group['key']} -> {label if label is not None else status}")


def save_labeled_crops(state: dict[str, Any], group: dict[str, Any], label: str) -> list[Path]:
    labeled_root: Path = state["labeled_root"]
    images_dir = labeled_root / "images"
    labels_file = labeled_root / "labels.txt"
    manifest_file = labeled_root / "manifest.csv"
    images_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = group.get("crop_paths", [])[: state["args"].samples_per_group]
    copied: list[Path] = []
    label_lines = []
    manifest_rows = []
    if crop_paths:
        for idx, src in enumerate(crop_paths):
            dst = images_dir / safe_filename(f"{group['key']}_{idx:02d}_{Path(src).name}")
            shutil.copy2(src, dst)
            rel = dst.relative_to(labeled_root).as_posix()
            label_lines.append(f"{rel}\t{label}\n")
            manifest_rows.append((rel, str(src), label, group["key"], group.get("team")))
            copied.append(dst)
    else:
        generated = generate_crops_from_records(group.get("records", []), state["cap"], state["args"].samples_per_group)
        for idx, img in enumerate(generated):
            dst = images_dir / safe_filename(f"{group['key']}_{idx:02d}.jpg")
            cv2.imwrite(str(dst), img)
            rel = dst.relative_to(labeled_root).as_posix()
            label_lines.append(f"{rel}\t{label}\n")
            manifest_rows.append((rel, "generated_from_video", label, group["key"], group.get("team")))
            copied.append(dst)

    with labels_file.open("a", encoding="utf-8") as f:
        f.writelines(label_lines)
    manifest_exists = manifest_file.exists()
    with manifest_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "label", "group", "team"])
        if not manifest_exists:
            writer.writeheader()
        for rel, src, lbl, key, team in manifest_rows:
            writer.writerow({"file": rel, "source": src, "label": lbl, "group": key, "team": team})
    return copied


def undo_last(state: dict[str, Any]) -> None:
    action = state.get("last_action")
    if not action:
        print("[undo] nothing to undo")
        return
    reviewed = state["overrides"].get("reviewed_groups", [])
    if reviewed and action.get("review_index") == len(reviewed) - 1:
        reviewed.pop()
    override = action.get("override_added")
    if override is not None:
        overrides = state["overrides"].get("track_overrides", [])
        if overrides and overrides[-1] == override:
            overrides.pop()
    removed = 0
    for path in action.get("copied", []):
        try:
            Path(path).unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    state["last_action"] = None
    write_overrides(state)
    print(f"[undo] removed last review, deleted {removed} copied crops. labels.txt is append-only; ignore undone rows if needed.")


def write_overrides(state: dict[str, Any]) -> None:
    output_path: Path = state["output_overrides"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state["overrides"])
    payload["description"] = "Manual jersey identity corrections generated by tools/review_jersey_identities.py"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_existing_overrides(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"track_overrides": [], "reviewed_groups": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("track_overrides", [])
    data.setdefault("reviewed_groups", [])
    return data


def prepare_labeled_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels.txt").touch(exist_ok=True)


def compact_vote_text(vote_summary: dict[str, Any]) -> str:
    chunks = []
    for group_name in ("players", "tracks"):
        for key, item in (vote_summary.get(group_name, {}) or {}).items():
            votes = item.get("frame_votes", {}) or {}
            top = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:3]
            chunks.append(f"{group_name[:-1]} {key}: pred={item.get('jersey_number')} votes={top}")
    return " | ".join(chunks) if chunks else "-"


def majority_value(values: Any) -> Any:
    counter = Counter(v for v in values if v is not None)
    return counter.most_common(1)[0][0] if counter else None


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    output = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        output.append(path)
    return output


def normalize_number(value: str) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) == 2 and digits.startswith("0"):
        digits = digits[1:]
    return digits


def is_valid_number(value: str) -> bool:
    digits = normalize_number(value)
    return bool(digits) and digits.isdigit() and len(digits) <= 2 and 0 <= int(digits) <= 99


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def load_json(path: Path | None) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}


def print_controls() -> None:
    print("Controls:")
    print("  digits + Enter: confirm/correct jersey number for this player/track group")
    print("  Enter: accept OCR prediction")
    print("  x: unknown/bad/no visible number; no override")
    print("  r: review later; no override")
    print("  n or space: skip")
    print("  p: previous group")
    print("  j/l: jump -/+10 groups")
    print("  u: undo last review")
    print("  q/esc: quit")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
