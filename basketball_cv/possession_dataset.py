from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


BALL_STATE_CLASSES = ("owned", "air", "loose")
OWNER_STATE_CLASSES = ("control", "dribble", "shot", "contested")
QUALITY_FLAGS = {
    "uncertain",
    "occluded",
    "ball_not_visible_cleanly",
    "candidate_missing",
    "reviewed_in_cvat",
}


def build_manifest_row(
    *,
    video: str,
    frame_index: int,
    split_hint: str | None,
    ball_state: str,
    ball_bbox: list[float] | list[int],
    candidate_players: list[dict[str, Any]],
    owner_player_id: Any | None = None,
    owner_track_id: Any | None = None,
    owner_team: str | None = None,
    owner_state: str | None = None,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    row = {
        "video": str(video),
        "frame_index": int(frame_index),
        "split_hint": split_hint,
        "ball_state": ball_state,
        "owner_player_id": owner_player_id,
        "owner_track_id": owner_track_id,
        "owner_team": owner_team,
        "owner_state": owner_state,
        "ball_bbox": [float(v) for v in ball_bbox],
        "candidate_players": candidate_players,
        "flags": list(flags or []),
    }
    validate_manifest_row(row)
    return row


def validate_manifest_row(row: dict[str, Any]) -> None:
    ball_state = str(row.get("ball_state") or "")
    if ball_state not in BALL_STATE_CLASSES:
        raise ValueError(f"ball_state must be one of {BALL_STATE_CLASSES}")
    flags = row.get("flags") or []
    unknown_flags = [flag for flag in flags if flag not in QUALITY_FLAGS]
    if unknown_flags:
        raise ValueError(f"unknown flags: {unknown_flags}")

    if not _valid_bbox(row.get("ball_bbox")):
        raise ValueError("ball_bbox must be a 4-value bbox")

    candidates = row.get("candidate_players") or []
    for candidate in candidates:
        if not _valid_bbox(candidate.get("bbox")):
            raise ValueError("candidate bbox must be a 4-value bbox")

    owner_values = (
        row.get("owner_player_id"),
        row.get("owner_track_id"),
        row.get("owner_team"),
        row.get("owner_state"),
    )
    has_owner_data = any(value is not None for value in owner_values)
    if ball_state != "owned":
        if has_owner_data:
            raise ValueError("owner fields must be empty unless ball_state is owned")
        return

    owner_state = row.get("owner_state")
    if owner_state not in OWNER_STATE_CLASSES:
        raise ValueError(f"owner_state must be one of {OWNER_STATE_CLASSES}")

    owner_player_id = row.get("owner_player_id")
    owner_track_id = row.get("owner_track_id")
    if owner_player_id is None and owner_track_id is None:
        raise ValueError("owned rows must reference an owner candidate")

    if not _candidate_matches_owner(candidates, owner_player_id, owner_track_id):
        raise ValueError("owner must match one of the candidate players")


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_manifest_row(row)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("source_manifest_row", idx)
            rows.append(row)
    return rows


def assign_manifest_split(row: dict[str, Any], val_video_stems: set[str], val_ratio: float) -> str:
    split_hint = row.get("split_hint")
    if split_hint in {"train", "val"}:
        return str(split_hint)
    video_stem = Path(str(row.get("video", ""))).stem
    if video_stem in val_video_stems:
        return "val"
    if val_video_stems:
        return "train"
    key = video_stem or f"row-{row.get('source_manifest_row', 0)}"
    return "val" if _stable_ratio(key) < float(val_ratio) else "train"


def export_manifest_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    task: str,
    max_negatives_per_frame: int = 4,
    ball_margin: float = 1.2,
    owner_margin: float = 0.35,
    val_video_stems: set[str] | None = None,
    val_ratio: float = 0.15,
    exclude_flags: set[str] | None = None,
) -> dict[str, Any]:
    if task not in {"ball-state", "owner-state"}:
        raise ValueError("task must be ball-state or owner-state")

    rows = load_manifest_rows(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_manifest_path = output_dir / "manifest.jsonl"
    if export_manifest_path.exists():
        export_manifest_path.unlink()

    exported = 0
    captures: dict[str, cv2.VideoCapture] = {}
    resolved_val_stems = set(val_video_stems or set())
    blocked_flags = set(exclude_flags or set())
    class_counts: dict[str, int] = {}
    skipped_flagged = 0

    try:
        for row in rows:
            validate_manifest_row(row)
            row_flags = set(row.get("flags") or [])
            if blocked_flags and row_flags.intersection(blocked_flags):
                skipped_flagged += 1
                continue
            frame = _read_frame(captures, str(row["video"]), int(row["frame_index"]))
            if frame is None:
                continue
            split = assign_manifest_split(row, resolved_val_stems, val_ratio)
            if task == "ball-state":
                crop = make_ball_context_crop(frame, row["ball_bbox"], margin=ball_margin)
                if crop is None:
                    continue
                label = str(row["ball_state"])
                out_path = _write_export_image(output_dir, split, label, row, crop, suffix="ball")
                _append_jsonl(
                    export_manifest_path,
                    {
                        "video": str(row["video"]),
                        "frame_index": int(row["frame_index"]),
                        "split": split,
                        "ball_state": str(row["ball_state"]),
                        "ball_bbox": row["ball_bbox"],
                        "flags": list(row.get("flags") or []),
                        "file": str(out_path.relative_to(output_dir)),
                        "task": task,
                        "label": label,
                        "source_manifest_row": int(row["source_manifest_row"]),
                    },
                )
                exported += 1
                class_counts[label] = class_counts.get(label, 0) + 1
                continue

            owner_examples = iter_owner_examples(row, max_negatives_per_frame=max_negatives_per_frame)
            for example in owner_examples:
                crop = make_owner_ball_crop(frame, example["candidate"]["bbox"], row["ball_bbox"], margin=owner_margin)
                if crop is None:
                    continue
                label = str(example["label"])
                out_path = _write_export_image(
                    output_dir,
                    split,
                    label,
                    row,
                    crop,
                    suffix=f"p{example['candidate'].get('player_id')}_r{example['candidate'].get('rank', 0)}",
                )
                _append_jsonl(
                    export_manifest_path,
                    {
                        "video": str(row["video"]),
                        "frame_index": int(row["frame_index"]),
                        "split": split,
                        "ball_state": str(row["ball_state"]),
                        "ball_bbox": row["ball_bbox"],
                        "owner_player_id": example["candidate"].get("player_id"),
                        "owner_track_id": example["candidate"].get("track_id"),
                        "owner_team": example["candidate"].get("team"),
                        "owner_state": label if label != "no_control" else None,
                        "flags": list(row.get("flags") or []),
                        "file": str(out_path.relative_to(output_dir)),
                        "task": task,
                        "label": label,
                        "jersey_number": example["candidate"].get("jersey_number"),
                        "source_manifest_row": int(row["source_manifest_row"]),
                    },
                )
                exported += 1
                class_counts[label] = class_counts.get(label, 0) + 1
    finally:
        for capture in captures.values():
            capture.release()

    return {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "task": task,
        "exported": exported,
        "class_counts": class_counts,
        "skipped_flagged": skipped_flagged,
    }


def iter_owner_examples(row: dict[str, Any], *, max_negatives_per_frame: int) -> list[dict[str, Any]]:
    if row.get("ball_state") != "owned":
        return []
    candidates = list(row.get("candidate_players") or [])
    owner_player_id = row.get("owner_player_id")
    owner_track_id = row.get("owner_track_id")
    owner_state = str(row.get("owner_state"))
    owner_candidate = None
    negatives: list[dict[str, Any]] = []
    for candidate in candidates:
        if _candidate_matches_owner([candidate], owner_player_id, owner_track_id):
            owner_candidate = candidate
        else:
            negatives.append(candidate)
    if owner_candidate is None:
        return []
    examples = [{"label": owner_state, "candidate": owner_candidate}]
    for candidate in negatives[: max(0, int(max_negatives_per_frame))]:
        examples.append({"label": "no_control", "candidate": candidate})
    return examples


def make_ball_context_crop(frame: Any, ball_bbox: list[float] | list[int], *, margin: float) -> Any | None:
    if not _valid_bbox(ball_bbox):
        return None
    return _crop_from_boxes(frame, [ball_bbox], margin=margin)


def make_owner_ball_crop(frame: Any, player_bbox: list[float] | list[int], ball_bbox: list[float] | list[int], *, margin: float) -> Any | None:
    if not _valid_bbox(player_bbox) or not _valid_bbox(ball_bbox):
        return None
    return _crop_from_boxes(frame, [player_bbox, ball_bbox], margin=margin)


def _read_frame(captures: dict[str, cv2.VideoCapture], video: str, frame_index: int) -> Any | None:
    capture = captures.get(video)
    if capture is None:
        capture = cv2.VideoCapture(video)
        captures[video] = capture
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return frame if ok else None


def _write_export_image(output_dir: Path, split: str, label: str, row: dict[str, Any], crop: Any, *, suffix: str) -> Path:
    out_dir = output_dir / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(str(row["video"])).stem
    filename = f"{video_stem}_f{int(row['frame_index']):06d}_{suffix}.jpg"
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), crop)
    return out_path


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _crop_from_boxes(frame: Any, boxes: list[list[float] | list[int]], *, margin: float) -> Any | None:
    valid = [[float(v) for v in box] for box in boxes if _valid_bbox(box)]
    if not valid:
        return None
    x1 = min(box[0] for box in valid)
    y1 = min(box[1] for box in valid)
    x2 = max(box[2] for box in valid)
    y2 = max(box[3] for box in valid)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad = float(margin) * max(width, height)
    left = max(0, int(round(x1 - pad)))
    top = max(0, int(round(y1 - pad)))
    right = min(frame.shape[1], int(round(x2 + pad)))
    bottom = min(frame.shape[0], int(round(y2 + pad)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right].copy()


def _candidate_matches_owner(candidates: list[dict[str, Any]], owner_player_id: Any | None, owner_track_id: Any | None) -> bool:
    for candidate in candidates:
        if owner_player_id is not None and str(candidate.get("player_id")) == str(owner_player_id):
            return True
        if owner_track_id is not None and str(candidate.get("track_id")) == str(owner_track_id):
            return True
    return False


def _valid_bbox(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4


def _stable_ratio(value: str) -> float:
    return (abs(hash(value)) % 10000) / 10000.0
