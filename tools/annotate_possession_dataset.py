from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.possession_model import PossessionModelConfig, choose_active_ball, make_player_ball_crop, select_candidates

WINDOW = "Possession dataset annotator"
LABEL_KEYS = {
    ord("c"): "control",
    ord("d"): "dribble",
    ord("s"): "shot",
    ord("t"): "contested",
}
CLASSES = ["control", "dribble", "shot", "contested", "no_control"]
IMG_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively label player-ball possession crops across many videos. "
            "The tool draws numbered candidates near the detected ball. You can press 1-9 "
            "or click a numbered candidate. The selected candidate is saved as the current "
            "owner label and the other nearby candidates are saved as no_control."
        )
    )
    parser.add_argument("--videos", nargs="*", default=None, help="Video paths. If omitted, scans --video-root.")
    parser.add_argument("--video-root", default=".", help="Root scanned for videos when --videos is omitted.")
    parser.add_argument("--glob", default="*.mp4", help="Video glob under --video-root.")
    parser.add_argument("--runs-root", default="runs/pipeline", help="Pipeline runs root used to find tracks.json.")
    parser.add_argument("--tracks", default=None, help="Single tracks.json. Only valid when labeling one video.")
    parser.add_argument("--output-dir", default="datasets/possession_cls", help="YOLO classification dataset root.")
    parser.add_argument("--split", choices=["train", "val", "auto"], default="auto")
    parser.add_argument("--val-video-stems", default="", help="Comma-separated video stems forced to val split.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Used only with --split auto and no val-video-stems match.")
    parser.add_argument("--sample-step", type=int, default=12, help="Frame step. At 60fps, 12 means 5 samples/sec.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=7)
    parser.add_argument("--candidate-radius-m", type=float, default=8.0)
    parser.add_argument("--candidate-min-contact", type=float, default=0.03)
    parser.add_argument("--image-candidate-radius-px", type=float, default=280.0)
    parser.add_argument("--crop-margin", type=float, default=0.42)
    parser.add_argument("--save-negative-candidates", action="store_true", default=True)
    parser.add_argument("--no-save-negative-candidates", action="store_false", dest="save_negative_candidates")
    parser.add_argument("--max-negatives-per-frame", type=int, default=4)
    parser.add_argument("--draw-all-people", action="store_true", default=True)
    parser.add_argument("--no-draw-all-people", action="store_false", dest="draw_all_people")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_root = Path(args.output_dir)
    ensure_dataset_dirs(out_root)
    manifest_path = out_root / "annotations_manifest.csv"
    manifest_exists = manifest_path.exists()

    videos = discover_videos(args)
    if not videos:
        raise SystemExit("No videos found.")

    with manifest_path.open("a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "file",
                "video",
                "split",
                "label",
                "frame_index",
                "candidate_rank",
                "track_id",
                "player_id",
                "team",
                "ball_bbox",
                "player_bbox",
                "kind",
            ],
        )
        if not manifest_exists:
            writer.writeheader()

        for video in videos:
            tracks_path = resolve_tracks_for_video(video, args)
            if tracks_path is None or not tracks_path.exists():
                print(f"[skip] Missing tracks.json for {video.name}. Run the game pipeline first.")
                continue
            annotate_video(video, tracks_path, args, out_root, writer)

    print(f"Manifest: {manifest_path}")


def annotate_video(video: Path, tracks_path: Path, args: argparse.Namespace, out_root: Path, writer: csv.DictWriter) -> None:
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_frame.setdefault(int(rec.get("frame_index", 0)), []).append(rec)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[skip] Could not open {video}")
        return
    fps = float(tracks.get("fps", cap.get(cv2.CAP_PROP_FPS) or 60.0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or max(by_frame, default=0) + 1)

    config = PossessionModelConfig(
        crop_margin=args.crop_margin,
        max_candidates=args.max_candidates,
        candidate_radius_m=args.candidate_radius_m,
        candidate_min_contact=args.candidate_min_contact,
        image_candidate_radius_px=args.image_candidate_radius_px,
        suppress_non_active_balls=False,
    )

    frame_indices = [idx for idx in sorted(by_frame) if idx >= args.start_frame and idx % max(args.sample_step, 1) == 0]
    if args.max_frames_per_video > 0:
        frame_indices = frame_indices[: args.max_frames_per_video]

    state = {
        "video": video,
        "tracks_path": tracks_path,
        "fps": fps,
        "total_frames": total_frames,
        "frame_indices": frame_indices,
        "index": 0,
        "cap": cap,
        "by_frame": by_frame,
        "config": config,
        "current_label": "control",
        "out_root": out_root,
        "args": args,
        "writer": writer,
        "undo_stack": [],
        "last_frame": None,
        "last_candidates": [],
        "last_ball": None,
        "last_image": None,
        "redraw_requested": False,
    }

    print(f"\n[video] {video.name} | tracks={tracks_path} | samples={len(frame_indices)} | fps={fps:.2f}")
    print_controls()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    while 0 <= state["index"] < len(frame_indices):
        draw_current(state)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("n") or key == ord(" "):
            state["index"] += 1
            continue
        if key == ord("p"):
            state["index"] = max(0, state["index"] - 1)
            continue
        if key == ord("j"):
            state["index"] = max(0, state["index"] - 10)
            continue
        if key == ord("l"):
            state["index"] = min(len(frame_indices) - 1, state["index"] + 10)
            continue
        if key == ord("u"):
            undo_last(state)
            continue
        if key in LABEL_KEYS:
            state["current_label"] = LABEL_KEYS[key]
            print(f"[label] current owner label: {state['current_label']}")
            continue
        if key == ord("x"):
            save_loose_frame(state)
            state["index"] += 1
            continue
        if ord("1") <= key <= ord("9"):
            candidate_idx = key - ord("1")
            save_owner_frame(state, candidate_idx)
            state["index"] += 1
            continue

    cap.release()
    cv2.destroyWindow(WINDOW)


def on_mouse(event: int, x: int, y: int, _flags: int, state: dict[str, Any]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    candidates = state.get("last_candidates") or []
    if not candidates:
        print("[click] No numbered candidates in this frame. Press n to skip or x if it is loose/no owner.")
        return
    idx = nearest_candidate_index(candidates[:9], x, y)
    if idx is None:
        print("[click] Click closer to a numbered candidate box, or press its 1-9 key.")
        return
    save_owner_frame(state, idx)
    state["index"] += 1
    if 0 <= state["index"] < len(state["frame_indices"]):
        draw_current(state)


def nearest_candidate_index(candidates: list[dict[str, Any]], x: int, y: int) -> int | None:
    best_idx = None
    best_score = 1e9
    for idx, player in enumerate(candidates):
        bbox = player.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        inside = x1 <= x <= x2 and y1 <= y <= y2
        cx = min(max(float(x), x1), x2)
        cy = min(max(float(y), y1), y2)
        dist = float(np.hypot(float(x) - cx, float(y) - cy))
        score = -1.0 if inside else dist
        if score < best_score and (inside or dist <= 90.0):
            best_score = score
            best_idx = idx
    return best_idx


def draw_current(state: dict[str, Any]) -> None:
    frame_idx = state["frame_indices"][state["index"]]
    cap: cv2.VideoCapture = state["cap"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        state["index"] += 1
        return
    frame_records = state["by_frame"].get(frame_idx, [])
    ball = choose_active_ball(frame_records, state["config"])
    candidates = select_candidates(ball, frame_records, state["config"]) if ball else []
    state["last_frame"] = frame_idx
    state["last_candidates"] = candidates
    state["last_ball"] = ball
    state["last_image"] = frame

    out = frame.copy()
    if state["args"].draw_all_people:
        for person in [rec for rec in frame_records if rec.get("class_name") == "person"]:
            draw_person_hint(out, person)
    if ball is not None:
        draw_box(out, ball.get("bbox"), (0, 190, 255), "BALL", thickness=3)
    for i, player in enumerate(candidates[:9], start=1):
        color = (255, 180, 0) if i % 2 else (0, 220, 255)
        label = f"{i}: T{player.get('track_id')} P{player.get('player_id')} {player.get('team')}"
        draw_candidate_box(out, player.get("bbox"), color, label, i)

    panel = [
        f"{state['video'].name} | frame {frame_idx} | {state['index'] + 1}/{len(state['frame_indices'])} | t={frame_idx / max(state['fps'], 1e-6):.2f}s",
        f"Current owner label: {state['current_label'].upper()}  | numbered candidates: {len(candidates)} | ball: {'yes' if ball else 'no'}",
        "Use 1-9 OR CLICK a numbered box. c/d/s/t changes label. x = loose/no owner. n = skip.",
        "Grey boxes are all detected people; ONLY yellow/cyan numbered boxes are selectable candidates.",
    ]
    draw_panel(out, panel)
    cv2.imshow(WINDOW, out)


def save_owner_frame(state: dict[str, Any], candidate_idx: int) -> None:
    candidates = state.get("last_candidates") or []
    ball = state.get("last_ball")
    frame = state.get("last_image")
    frame_idx = state.get("last_frame")
    if ball is None or frame is None or frame_idx is None:
        print("[warn] No ball/frame available.")
        return
    if candidate_idx < 0 or candidate_idx >= len(candidates):
        print(f"[warn] Candidate {candidate_idx + 1} not available.")
        return

    label = state["current_label"]
    written = []
    owner = candidates[candidate_idx]
    written.extend(write_crop(state, frame, ball, owner, label, candidate_idx, "owner"))

    if state["args"].save_negative_candidates:
        negs = [item for idx, item in enumerate(candidates[:9]) if idx != candidate_idx]
        negs = negs[: max(0, int(state["args"].max_negatives_per_frame))]
        for idx, player in enumerate(negs):
            rank = candidates.index(player) if player in candidates else idx
            written.extend(write_crop(state, frame, ball, player, "no_control", rank, "auto_negative"))

    state["undo_stack"].append(written)
    print(f"[saved] frame={frame_idx} owner_candidate={candidate_idx + 1} label={label} files={len(written)}")


def save_loose_frame(state: dict[str, Any]) -> None:
    candidates = state.get("last_candidates") or []
    ball = state.get("last_ball")
    frame = state.get("last_image")
    frame_idx = state.get("last_frame")
    if ball is None or frame is None or frame_idx is None:
        print("[warn] No ball/frame available.")
        return
    written = []
    for idx, player in enumerate(candidates[: max(1, int(state["args"].max_negatives_per_frame))]):
        written.extend(write_crop(state, frame, ball, player, "no_control", idx, "loose_negative"))
    state["undo_stack"].append(written)
    print(f"[saved] frame={frame_idx} loose/no owner files={len(written)}")


def write_crop(
    state: dict[str, Any],
    frame: np.ndarray,
    ball: dict[str, Any],
    player: dict[str, Any],
    label: str,
    rank: int,
    kind: str,
) -> list[Path]:
    crop = make_player_ball_crop(frame, player, ball, state["args"].crop_margin)
    if crop is None or crop.size == 0:
        return []
    split = choose_split(state, label)
    out_dir = Path(state["out_root"]) / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = int(state["last_frame"])
    video_stem = safe_stem(Path(state["video"]).stem)
    track_id = player.get("track_id")
    player_id = player.get("player_id")
    filename = f"{video_stem}_f{frame_idx:06d}_r{rank}_t{track_id}_p{player_id}_{kind}.jpg"
    path = out_dir / filename
    cv2.imwrite(str(path), crop)

    rel = path.relative_to(state["out_root"])
    state["writer"].writerow(
        {
            "file": str(rel),
            "video": str(state["video"]),
            "split": split,
            "label": label,
            "frame_index": frame_idx,
            "candidate_rank": rank,
            "track_id": track_id,
            "player_id": player_id,
            "team": player.get("team"),
            "ball_bbox": ball.get("bbox"),
            "player_bbox": player.get("bbox"),
            "kind": kind,
        }
    )
    return [path]


def choose_split(state: dict[str, Any], label: str) -> str:
    args = state["args"]
    if args.split != "auto":
        return args.split
    val_stems = {item.strip() for item in args.val_video_stems.split(",") if item.strip()}
    video_stem = Path(state["video"]).stem
    if video_stem in val_stems:
        return "val"
    if val_stems:
        return "train"
    key = f"{video_stem}-{state['last_frame']}-{label}"
    return "val" if stable_ratio(key) < float(args.val_ratio) else "train"


def undo_last(state: dict[str, Any]) -> None:
    if not state["undo_stack"]:
        print("[undo] nothing to undo")
        return
    paths = state["undo_stack"].pop()
    removed = 0
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    print(f"[undo] removed {removed} files. Manifest rows are append-only; ignore latest undone rows if needed.")


def discover_videos(args: argparse.Namespace) -> list[Path]:
    if args.videos:
        return [resolve_path(value) for value in args.videos if resolve_path(value).exists()]
    root = resolve_path(args.video_root)
    return sorted([path for path in root.glob(args.glob) if path.suffix.lower() in IMG_EXTS])


def resolve_tracks_for_video(video: Path, args: argparse.Namespace) -> Path | None:
    if args.tracks:
        return resolve_path(args.tracks)
    runs_root = resolve_path(args.runs_root)
    candidates = []
    for path in runs_root.glob("*/tracks.json"):
        run_name = path.parent.name.lower()
        if video.stem.lower() in run_name:
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def ensure_dataset_dirs(root: Path) -> None:
    for split in ("train", "val"):
        for label in CLASSES:
            (root / split / label).mkdir(parents=True, exist_ok=True)


def draw_person_hint(frame: np.ndarray, person: dict[str, Any]) -> None:
    bbox = person.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    color = (135, 135, 135)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    label = f"T{person.get('track_id')} P{person.get('player_id')} {person.get('team')}"
    cv2.putText(frame, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_candidate_box(frame: np.ndarray, bbox: Any, color: tuple[int, int, int], label: str, number: int) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
    cx = int(round((x1 + x2) / 2))
    cy = int(round(max(y1 + 22, y1 + 0.18 * (y2 - y1))))
    cv2.circle(frame, (cx, cy), 22, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 20, color, -1, cv2.LINE_AA)
    cv2.putText(frame, str(number), (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, str(number), (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_box(frame: np.ndarray, bbox: Any, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 25
    width = min(frame.shape[1] - 20, max(cv2.getTextSize(line, font, 0.58, 2)[0][0] for line in lines) + 24)
    height = 16 + line_h * len(lines)
    x0, y0 = 12, 12
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 220, 255), 2)
    for idx, line in enumerate(lines):
        y = y0 + 26 + idx * line_h
        cv2.putText(frame, line, (x0 + 12, y), font, 0.58, (245, 245, 245), 2, cv2.LINE_AA)


def print_controls() -> None:
    print("Controls:")
    print("  c/d/s/t: set owner label control/dribble/shot/contested")
    print("  1-9: choose numbered owner candidate")
    print("  mouse click: choose nearest numbered candidate")
    print("  x: loose/no owner; candidates become no_control")
    print("  n or space: skip")
    print("  p: previous sample")
    print("  j/l: jump -/+ 10 samples")
    print("  u: undo files from last save")
    print("  q/esc: quit current video")


def stable_ratio(value: str) -> float:
    return (abs(hash(value)) % 10000) / 10000.0


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value).strip("_") or "video"


if __name__ == "__main__":
    main()
