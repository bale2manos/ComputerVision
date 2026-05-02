from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_video import color_for_record

WINDOW = "Team calibration"
ROLES = ["team_a", "team_b", "referee"]
ROLE_INSTRUCTIONS = {
    "team_a": "Click 2 players from TEAM A, then press N",
    "team_b": "Click 2 players from TEAM B, then press N",
    "referee": "Click 1 referee, then press S to save",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click players/referee to create a manual team calibration JSON.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--frame", type=int, default=0, help="Initial frame to display.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-click-distance", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        if rec.get("class_name") == "person":
            by_frame.setdefault(int(rec.get("frame_index", 0)), []).append(rec)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or max(by_frame, default=0))
    state = {
        "frame": int(args.frame),
        "role_index": 0,
        "selected": {"team_a": [], "team_b": [], "referee": []},
        "records": by_frame,
        "frame_img": None,
        "display": None,
        "max_click_distance": float(args.max_click_distance),
        "video": args.video,
        "fps": float(tracks.get("fps", cap.get(cv2.CAP_PROP_FPS) or 30.0)),
    }

    def load_frame(frame_index: int) -> None:
        frame_index = max(0, min(int(frame_index), max_frame - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            return
        state["frame"] = frame_index
        state["frame_img"] = frame
        redraw(state)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse, state)
    load_frame(state["frame"])

    print("Controls:")
    print("  click: select nearest person bbox for current role")
    print("  n: next role")
    print("  b: previous role")
    print("  a/d: previous/next frame")
    print("  j/l: -/+ 30 frames")
    print("  u: undo last selection for current role")
    print("  s: save")
    print("  q/esc: quit without saving")

    while True:
        if state["display"] is not None:
            cv2.imshow(WINDOW, state["display"])
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("a"):
            load_frame(state["frame"] - 1)
        elif key == ord("d"):
            load_frame(state["frame"] + 1)
        elif key == ord("j"):
            load_frame(state["frame"] - 30)
        elif key == ord("l"):
            load_frame(state["frame"] + 30)
        elif key == ord("n"):
            state["role_index"] = min(len(ROLES) - 1, state["role_index"] + 1)
            redraw(state)
        elif key == ord("b"):
            state["role_index"] = max(0, state["role_index"] - 1)
            redraw(state)
        elif key == ord("u"):
            role = ROLES[state["role_index"]]
            if state["selected"][role]:
                state["selected"][role].pop()
            redraw(state)
        elif key == ord("s"):
            save_calibration(args.output, state)
            break

    cap.release()
    cv2.destroyAllWindows()


def on_mouse(event: int, x: int, y: int, _flags: int, state: dict[str, Any]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    role = ROLES[state["role_index"]]
    frame = int(state["frame"])
    records = state["records"].get(frame, [])
    rec = nearest_record(records, x, y, state["max_click_distance"])
    if rec is None:
        print(f"No person close to click ({x}, {y})")
        return
    item = {
        "frame_index": frame,
        "track_id": rec.get("track_id"),
        "player_id": rec.get("player_id"),
        "bbox": rec.get("bbox"),
    }
    state["selected"][role].append(item)
    print(f"Selected {role}: track={item['track_id']} player={item['player_id']} frame={frame}")
    redraw(state)


def nearest_record(records: list[dict[str, Any]], x: int, y: int, max_dist: float) -> dict[str, Any] | None:
    best = None
    best_score = 1e9
    for rec in records:
        bbox = rec.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        inside = x1 <= x <= x2 and y1 <= y <= y2
        cx = min(max(float(x), x1), x2)
        cy = min(max(float(y), y1), y2)
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        score = -1.0 if inside else dist
        if score < best_score and (inside or dist <= max_dist):
            best = rec
            best_score = score
    return best


def redraw(state: dict[str, Any]) -> None:
    frame = state.get("frame_img")
    if frame is None:
        return
    out = frame.copy()
    frame_index = int(state["frame"])
    role = ROLES[state["role_index"]]
    records = state["records"].get(frame_index, [])
    for rec in records:
        bbox = rec.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        color = color_for_record(rec)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"T{rec.get('track_id')} P{rec.get('player_id')} {rec.get('team')}"
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    overlay_lines = [
        f"Frame {frame_index}",
        ROLE_INSTRUCTIONS[role],
        f"team_a: {len(state['selected']['team_a'])} | team_b: {len(state['selected']['team_b'])} | referee: {len(state['selected']['referee'])}",
        "Keys: n next, b prev, a/d frame, j/l 30 frames, u undo, s save, q quit",
    ]
    draw_panel(out, overlay_lines)
    state["display"] = out


def draw_panel(frame, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = 24
    width = min(frame.shape[1] - 20, max(cv2.getTextSize(line, font, 0.62, 2)[0][0] for line in lines) + 24)
    height = 16 + line_h * len(lines)
    x0, y0 = 12, 12
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 220, 255), 2)
    for i, line in enumerate(lines):
        y = y0 + 26 + i * line_h
        cv2.putText(frame, line, (x0 + 12, y), font, 0.62, (245, 245, 245), 2, cv2.LINE_AA)


def save_calibration(output: str, state: dict[str, Any]) -> None:
    payload = {
        "video": state["video"],
        "fps": state["fps"],
        "description": "Manual team calibration. team_a/team_b are generic labels for this game.",
        "team_a": state["selected"]["team_a"],
        "team_b": state["selected"]["team_b"],
        "referee": state["selected"]["referee"],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
