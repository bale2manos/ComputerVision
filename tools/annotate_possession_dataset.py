from __future__ import annotations

import argparse
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

from basketball_cv.possession_dataset import append_manifest_row, build_manifest_row
from basketball_cv.possession_model import PossessionModelConfig, choose_active_ball, select_candidates

WINDOW = "Possession dataset annotator"
OWNER_STATE_KEYS = {
    ord("c"): "control",
    ord("d"): "dribble",
    ord("s"): "shot",
    ord("t"): "contested",
}
FLAG_KEYS = {
    ord("f"): "uncertain",
    ord("g"): "occluded",
    ord("b"): "ball_not_visible_cleanly",
    ord("m"): "candidate_missing",
}
IMG_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively label basketball possession in two steps: first choose the ball "
            "state (owned / air / loose), then if owned choose the player candidate and "
            "owner subtype. The tool saves canonical manifest rows instead of class folders."
        )
    )
    parser.add_argument("--videos", nargs="*", default=None, help="Video paths. If omitted, scans --video-root.")
    parser.add_argument("--video-root", default=".", help="Root scanned for videos when --videos is omitted.")
    parser.add_argument("--glob", default="*.mp4", help="Video glob under --video-root.")
    parser.add_argument("--runs-root", default="runs/pipeline", help="Pipeline runs root used to find tracks.json.")
    parser.add_argument("--tracks", default=None, help="Single tracks.json. Only valid when labeling one video.")
    parser.add_argument("--manifest", default="datasets/possession_labels/manifest.jsonl", help="Canonical JSONL manifest output.")
    parser.add_argument("--split-hint", choices=["train", "val", "auto"], default="auto", help="Optional split hint stored in manifest rows.")
    parser.add_argument("--val-video-stems", default="", help="Comma-separated video stems that should store split_hint=val.")
    parser.add_argument("--sample-step", type=int, default=12, help="Frame step. At 60fps, 12 means 5 samples/sec.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=7)
    parser.add_argument("--candidate-radius-m", type=float, default=8.0)
    parser.add_argument("--candidate-min-contact", type=float, default=0.03)
    parser.add_argument("--image-candidate-radius-px", type=float, default=280.0)
    parser.add_argument("--draw-all-people", action="store_true", default=True)
    parser.add_argument("--no-draw-all-people", action="store_false", dest="draw_all_people")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    manifest_path = resolve_path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(args)
    if not videos:
        raise SystemExit("No videos found.")

    for video in videos:
        tracks_path = resolve_tracks_for_video(video, args)
        if tracks_path is None or not tracks_path.exists():
            print(f"[skip] Missing tracks.json for {video.name}. Run the game pipeline first.")
            continue
        annotate_video(video, tracks_path, args, manifest_path)

    print(f"Manifest: {manifest_path}")


def annotate_video(video: Path, tracks_path: Path, args: argparse.Namespace, manifest_path: Path) -> None:
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    records = tracks.get("records", []) if isinstance(tracks, dict) else tracks
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_frame.setdefault(int(rec.get("frame_index", 0)), []).append(rec)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[skip] Could not open {video}")
        return
    fps = float(tracks.get("fps", cap.get(cv2.CAP_PROP_FPS) or 60.0)) if isinstance(tracks, dict) else float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    frame_indices = [idx for idx in sorted(by_frame) if idx >= args.start_frame and idx % max(args.sample_step, 1) == 0]
    if args.max_frames_per_video > 0:
        frame_indices = frame_indices[: args.max_frames_per_video]

    state = {
        "video": video,
        "tracks_path": tracks_path,
        "fps": fps,
        "frame_indices": frame_indices,
        "index": 0,
        "cap": cap,
        "by_frame": by_frame,
        "config": PossessionModelConfig(
            max_candidates=args.max_candidates,
            candidate_radius_m=args.candidate_radius_m,
            candidate_min_contact=args.candidate_min_contact,
            image_candidate_radius_px=args.image_candidate_radius_px,
            suppress_non_active_balls=False,
        ),
        "args": args,
        "manifest_path": manifest_path,
        "mode": "await_ball_state",
        "pending_owner_index": None,
        "pending_flags": set(),
        "undo_stack": [],
        "last_frame": None,
        "last_candidates": [],
        "last_ball": None,
        "last_image": None,
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
        if key in (ord("n"), ord(" ")):
            reset_pending_state(state)
            state["index"] += 1
            continue
        if key == ord("p"):
            reset_pending_state(state)
            state["index"] = max(0, state["index"] - 1)
            continue
        if key == ord("j"):
            reset_pending_state(state)
            state["index"] = max(0, state["index"] - 10)
            continue
        if key == ord("k"):
            reset_pending_state(state)
            state["index"] = min(len(state["frame_indices"]) - 1, state["index"] + 10)
            continue
        if key == ord("u"):
            undo_last(state)
            continue
        if key in FLAG_KEYS:
            toggle_flag(state, FLAG_KEYS[key])
            continue
        if key == ord("o"):
            if state.get("last_ball") is None:
                print("[warn] No active ball available in this frame.")
                continue
            state["mode"] = "await_owner_candidate"
            state["pending_owner_index"] = None
            continue
        if key == ord("a"):
            save_ball_state_row(state, "air")
            state["index"] += 1
            continue
        if key == ord("l"):
            save_ball_state_row(state, "loose")
            state["index"] += 1
            continue
        if ord("1") <= key <= ord("9"):
            if state["mode"] != "await_owner_candidate":
                print("[hint] Press o first if this is an owned-ball frame.")
                continue
            choose_owner_candidate(state, key - ord("1"))
            continue
        if key in OWNER_STATE_KEYS:
            if state["mode"] != "await_owner_state":
                print("[hint] Select owner candidate first.")
                continue
            save_owned_row(state, OWNER_STATE_KEYS[key])
            state["index"] += 1
            continue

    cap.release()
    cv2.destroyWindow(WINDOW)


def on_mouse(event: int, x: int, y: int, _flags: int, state: dict[str, Any]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN or state.get("mode") != "await_owner_candidate":
        return
    candidates = state.get("last_candidates") or []
    if not candidates:
        print("[click] No selectable candidates in this frame.")
        return
    idx = nearest_candidate_index(candidates[:9], x, y)
    if idx is None:
        print("[click] Click closer to a numbered candidate box, or use the 1-9 keys.")
        return
    choose_owner_candidate(state, idx)
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


def choose_owner_candidate(state: dict[str, Any], candidate_idx: int) -> None:
    candidates = state.get("last_candidates") or []
    if candidate_idx < 0 or candidate_idx >= len(candidates):
        print(f"[warn] Candidate {candidate_idx + 1} not available.")
        return
    state["pending_owner_index"] = candidate_idx
    state["mode"] = "await_owner_state"
    player = candidates[candidate_idx]
    print(f"[owner] selected candidate {candidate_idx + 1}: P{player.get('player_id')} T{player.get('track_id')} {player.get('team')}. Choose c/d/s/t.")


def save_ball_state_row(state: dict[str, Any], ball_state: str) -> None:
    ball = state.get("last_ball")
    frame_idx = state.get("last_frame")
    if ball is None or frame_idx is None:
        print("[warn] No active ball/frame available.")
        return
    row = build_manifest_row(
        video=str(state["video"]),
        frame_index=int(frame_idx),
        split_hint=choose_split_hint(state),
        ball_state=ball_state,
        ball_bbox=ball.get("bbox"),
        candidate_players=serialize_candidates(state.get("last_candidates") or []),
        flags=sorted(state.get("pending_flags") or set()),
    )
    append_manifest_row(state["manifest_path"], row)
    state["undo_stack"].append({"frame_index": int(frame_idx), "kind": ball_state})
    print(f"[saved] frame={frame_idx} ball_state={ball_state}")
    reset_pending_state(state)


def save_owned_row(state: dict[str, Any], owner_state: str) -> None:
    candidates = state.get("last_candidates") or []
    ball = state.get("last_ball")
    frame_idx = state.get("last_frame")
    owner_idx = state.get("pending_owner_index")
    if ball is None or frame_idx is None or owner_idx is None:
        print("[warn] Missing frame, ball, or owner selection.")
        return
    if owner_idx < 0 or owner_idx >= len(candidates):
        print("[warn] Selected owner candidate is not available anymore.")
        return
    player = candidates[owner_idx]
    row = build_manifest_row(
        video=str(state["video"]),
        frame_index=int(frame_idx),
        split_hint=choose_split_hint(state),
        ball_state="owned",
        ball_bbox=ball.get("bbox"),
        candidate_players=serialize_candidates(candidates),
        owner_player_id=player.get("player_id"),
        owner_track_id=player.get("track_id"),
        owner_team=player.get("team"),
        owner_state=owner_state,
        flags=sorted(state.get("pending_flags") or set()),
    )
    append_manifest_row(state["manifest_path"], row)
    state["undo_stack"].append({"frame_index": int(frame_idx), "kind": owner_state})
    print(f"[saved] frame={frame_idx} owner_candidate={owner_idx + 1} owner_state={owner_state}")
    reset_pending_state(state)


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

    highlight_idx = state.get("pending_owner_index")
    for i, player in enumerate(candidates[:9], start=1):
        color = (255, 180, 0) if i % 2 else (0, 220, 255)
        if highlight_idx is not None and (i - 1) == highlight_idx:
            color = (60, 255, 80)
        label = f"{i}: T{player.get('track_id')} P{player.get('player_id')} {player.get('team')}"
        draw_candidate_box(out, player.get("bbox"), color, label, i)

    panel = build_panel_lines(state, frame_idx, ball, candidates)
    draw_panel(out, panel)
    cv2.imshow(WINDOW, out)


def build_panel_lines(state: dict[str, Any], frame_idx: int, ball: dict[str, Any] | None, candidates: list[dict[str, Any]]) -> list[str]:
    mode = state.get("mode")
    pending_flags = ",".join(sorted(state.get("pending_flags") or set())) or "-"
    base = [
        f"{state['video'].name} | frame {frame_idx} | {state['index'] + 1}/{len(state['frame_indices'])} | t={frame_idx / max(state['fps'], 1e-6):.2f}s",
        f"mode={mode} | candidates={len(candidates)} | ball={'yes' if ball else 'no'} | flags={pending_flags}",
    ]
    if mode == "await_ball_state":
        base += [
            "Step A: o = owned, a = air, l = loose, n = skip.",
            "Flags: f uncertain, g occluded, b ball_not_visible_cleanly, m candidate_missing.",
            "Grey boxes are all people; yellow/cyan boxes become selectable only after pressing o.",
        ]
        return base
    if mode == "await_owner_candidate":
        base += [
            "Step B.1: choose the owner candidate with 1-9 or mouse click.",
            "Press c/d/s/t only after a candidate is selected.",
            "Press n to skip this frame or p to go back.",
        ]
        return base
    if mode == "await_owner_state":
        owner_idx = state.get("pending_owner_index")
        owner_label = f"candidate={owner_idx + 1}" if owner_idx is not None else "candidate=-"
        base += [
            f"Step B.2: selected {owner_label}. Finalize with c = control, d = dribble, s = shot, t = contested.",
            "Press o again to re-pick the owner candidate if needed.",
            "Press n to skip this frame or p to go back.",
        ]
        return base
    return base


def toggle_flag(state: dict[str, Any], flag: str) -> None:
    flags: set[str] = state["pending_flags"]
    if flag in flags:
        flags.remove(flag)
    else:
        flags.add(flag)
    print(f"[flags] {','.join(sorted(flags)) or '-'}")


def reset_pending_state(state: dict[str, Any]) -> None:
    state["mode"] = "await_ball_state"
    state["pending_owner_index"] = None
    state["pending_flags"] = set()


def choose_split_hint(state: dict[str, Any]) -> str:
    args = state["args"]
    if args.split_hint in {"train", "val"}:
        return args.split_hint
    val_stems = {item.strip() for item in args.val_video_stems.split(",") if item.strip()}
    video_stem = Path(state["video"]).stem
    if video_stem in val_stems:
        return "val"
    if val_stems:
        return "train"
    return "auto"


def serialize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for rank, player in enumerate(candidates[:9]):
        bbox = player.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        serialized.append(
            {
                "player_id": player.get("player_id"),
                "track_id": player.get("track_id"),
                "team": player.get("team"),
                "jersey_number": player.get("jersey_number"),
                "bbox": [float(v) for v in bbox],
                "rank": rank,
            }
        )
    return serialized


def undo_last(state: dict[str, Any]) -> None:
    if not state["undo_stack"]:
        print("[undo] nothing to undo")
        return
    info = state["undo_stack"].pop()
    removed = pop_last_manifest_row(state["manifest_path"])
    if removed:
        print(f"[undo] removed last manifest row for frame={info['frame_index']} ({info['kind']})")
    else:
        print("[undo] manifest row could not be removed")


def pop_last_manifest_row(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return False
    lines = lines[:-1]
    text = ("\n".join(lines) + ("\n" if lines else ""))
    path.write_text(text, encoding="utf-8")
    return True


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
    print("  o / a / l: mark owned / air / loose")
    print("  1-9 or mouse click: choose numbered owner candidate after pressing o")
    print("  c / d / s / t: finalize owner state control / dribble / shot / contested")
    print("  f / g / b / m: toggle uncertain / occluded / ball_not_visible_cleanly / candidate_missing")
    print("  n or space: skip")
    print("  p: previous sample")
    print("  j / k: jump -/+ 10 samples")
    print("  u: undo last manifest row")
    print("  q / esc: quit current video")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
