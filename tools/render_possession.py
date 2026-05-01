from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.court import CourtSpec, court_to_canvas, draw_topdown_court, load_calibration
from basketball_cv.events import densify_ball_track_for_render, detect_passes, interpolate_ball_gaps
from basketball_cv.possession import (
    assign_enhanced_ball_ownership,
    best_possession_for_frame,
    build_possession_by_frame,
    build_possession_timeline,
)
from tools.analyze_video import (
    build_ball_trails_by_frame,
    build_pass_active_by_frame,
    build_toasts_by_frame,
    color_for_record,
    draw_ball_marker,
    draw_ball_trail,
    draw_toasts,
    make_writer,
    write_json,
)
from tools.render_tracks import (
    apply_identity_overrides,
    apply_jersey_numbers,
    backfill_jersey_identities_across_fragments,
    interpolate_jersey_identity_gaps,
    load_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tracks with persistent possession HUD and debug timeline.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--tracks", required=True, help="tracks.json produced by analyze_video.py.")
    parser.add_argument("--calibration", required=True, help="Court calibration JSON.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--jersey-numbers", default=None, help="Optional jersey_numbers.json.")
    parser.add_argument("--identity-overrides", default=None, help="Optional manual identity override JSON.")
    parser.add_argument("--events", default=None, help="Optional existing events.json.")
    parser.add_argument("--output-events", default=None, help="Optional output events path after recomputed passes.")
    parser.add_argument("--output-possession", default=None, help="Optional possession_timeline.json path.")
    parser.add_argument("--no-detect-passes", action="store_true", help="Do not recompute passes after enhanced possession.")
    parser.add_argument("--no-dense-ball-track", action="store_true", help="Disable visual dense ball track.")
    parser.add_argument("--dense-ball-max-gap", type=float, default=3.0, help="Max seconds between ball detections to visually connect.")
    parser.add_argument("--no-interpolate-jersey-gaps", action="store_true", help="Disable short jersey identity interpolation.")
    parser.add_argument("--max-interpolation-gap", type=int, default=75, help="Max jersey interpolation gap in frames.")
    parser.add_argument("--debug-possession", action="store_true", help="Show owner candidates and scores in the HUD.")
    parser.add_argument("--no-possession-hud", action="store_true", help="Disable the persistent possession HUD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    fps = float(tracks.get("fps", 60.0))

    if args.jersey_numbers:
        apply_jersey_numbers(records, Path(args.jersey_numbers))
    if args.identity_overrides:
        apply_identity_overrides(records, Path(args.identity_overrides))
    if args.jersey_numbers or args.identity_overrides:
        backfill_jersey_identities_across_fragments(records, fps)
        if not args.no_interpolate_jersey_gaps:
            interpolate_jersey_identity_gaps(records, fps, args.max_interpolation_gap)

    events = load_events(Path(args.events)) if args.events else []
    interpolate_ball_gaps(records, fps)
    ownership_report = assign_enhanced_ball_ownership(records, fps)

    if not args.no_detect_passes:
        recomputed_passes = detect_passes(records, fps)
        non_pass_events = [event for event in events if event.get("type") != "pass"]
        events = sorted(non_pass_events + recomputed_passes, key=lambda event: int(event.get("start_frame", 0)))
        if args.output_events:
            write_json(Path(args.output_events), {"video": args.video, "fps": fps, "events": events})

    timeline = build_possession_timeline(records, fps)
    if args.output_possession:
        write_json(Path(args.output_possession), {"video": args.video, "fps": fps, "summary": ownership_report, "timeline": timeline})

    if not args.no_dense_ball_track:
        frame_count = int(tracks.get("frame_count") or max((int(rec.get("frame_index", 0)) for rec in records), default=0) + 1)
        densify_ball_track_for_render(records, fps, frame_count=frame_count, max_linear_gap_s=args.dense_ball_max_gap)

    calibration = load_calibration(args.calibration)
    court_spec = CourtSpec(**calibration["court"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_annotated_video_with_possession(
        args.video,
        output_path,
        records,
        fps,
        court_spec,
        events=events,
        show_possession_hud=not args.no_possession_hud,
        debug_possession=args.debug_possession,
    )
    print(output_path.resolve())


def render_annotated_video_with_possession(
    video_path: str,
    output_path: Path,
    records: list[dict[str, Any]],
    fps: float,
    court_spec: CourtSpec,
    events: list[dict[str, Any]] | None = None,
    show_possession_hud: bool = True,
    debug_possession: bool = False,
) -> None:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_frame.setdefault(int(rec["frame_index"]), []).append(rec)
    if not by_frame:
        return

    toasts_by_frame = build_toasts_by_frame(events or [], fps)
    pass_active_by_frame = build_pass_active_by_frame(events or [])
    ball_trails_by_frame = build_ball_trails_by_frame(records, trail_frames=24)
    possession_by_frame = build_possession_by_frame(records)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video for rendering: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = make_writer(output_path, fps, width, height)
    max_frame = max(by_frame)
    frame_index = 0
    while frame_index <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame = draw_frame_with_possession(
            frame,
            by_frame.get(frame_index, []),
            court_spec,
            toasts=toasts_by_frame.get(frame_index, []),
            pass_active=pass_active_by_frame.get(frame_index, False),
            ball_trail=ball_trails_by_frame.get(frame_index, []),
            possession=possession_by_frame.get(frame_index),
            show_possession_hud=show_possession_hud,
            debug_possession=debug_possession,
        )
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()


def draw_frame_with_possession(
    frame: np.ndarray,
    records: list[dict[str, Any]],
    court_spec: CourtSpec,
    toasts: list[str] | None = None,
    pass_active: bool = False,
    ball_trail: list[dict[str, Any]] | None = None,
    possession: dict[str, Any] | None = None,
    show_possession_hud: bool = True,
    debug_possession: bool = False,
) -> np.ndarray:
    has_projection = any(rec.get("court_x") is not None and rec.get("court_y") is not None for rec in records)
    minimap = draw_topdown_court(court_spec, pixels_per_meter=24, margin_px=18) if has_projection else None
    possession = possession or best_possession_for_frame(records)

    if ball_trail:
        draw_ball_trail(frame, ball_trail)
    draw_possession_link(frame, possession)

    for rec in records:
        color = color_for_record(rec)
        x1, y1, x2, y2 = [int(round(v)) for v in rec["bbox"]]
        has_ball = bool(rec.get("has_ball")) and rec.get("class_name") == "person"
        thickness = 4 if has_ball else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if has_ball:
            draw_possession_halo(frame, rec)

        if rec["class_name"] == "sports ball":
            draw_ball_marker(frame, rec, color)
            label = "ball est" if rec.get("dense_ball_estimate") else "ball"
            if rec.get("ball_state"):
                label += f" {rec['ball_state']}"
            if rec.get("ball_owner_jersey_number"):
                label += f" -> {rec['ball_owner_jersey_number']}"
            elif rec.get("ball_owner_player_id"):
                label += f" -> P{rec['ball_owner_player_id']}"
        else:
            label = rec["class_name"]
            if rec.get("jersey_number"):
                label += f" #{rec['jersey_number']}"
            elif rec.get("player_id") is not None:
                label += f" P{rec['player_id']}"
            elif rec.get("track_id") is not None:
                label += f" #{rec['track_id']}"
            if rec.get("team"):
                label += f" {rec['team']}"
            if has_ball:
                label += " BALL"

        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        if minimap is not None and rec.get("court_x") is not None and rec.get("court_y") is not None:
            pt = court_to_canvas(np.asarray([[rec["court_x"], rec["court_y"]]], dtype=np.float32), court_spec, pixels_per_meter=24, margin_px=18)[0]
            radius = 8 if has_ball else (6 if rec["class_name"] == "person" else 4)
            if has_ball:
                cv2.circle(minimap, (int(pt[0]), int(pt[1])), radius + 3, (0, 210, 255), 2, cv2.LINE_AA)
            cv2.circle(minimap, (int(pt[0]), int(pt[1])), radius, color, -1)
            label_id = rec.get("jersey_number") or (rec.get("player_id") if rec.get("player_id") is not None else rec.get("track_id"))
            if label_id is not None:
                cv2.putText(minimap, str(label_id), (int(pt[0]) + 7, int(pt[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if toasts:
        draw_toasts(frame, toasts)
    if show_possession_hud:
        draw_possession_hud(frame, possession, debug=debug_possession)

    if minimap is None:
        return frame

    mh, mw = minimap.shape[:2]
    scale = min(0.38 * frame.shape[1] / mw, 0.32 * frame.shape[0] / mh)
    minimap = cv2.resize(minimap, (int(mw * scale), int(mh * scale)), interpolation=cv2.INTER_AREA)
    mh, mw = minimap.shape[:2]
    x0 = frame.shape[1] - mw - 20
    y0 = frame.shape[0] - mh - 20
    overlay = frame[y0 : y0 + mh, x0 : x0 + mw].copy()
    blended = cv2.addWeighted(overlay, 0.25, minimap, 0.75, 0)
    frame[y0 : y0 + mh, x0 : x0 + mw] = blended
    cv2.rectangle(frame, (x0, y0), (x0 + mw, y0 + mh), (40, 40, 40), 1)
    return frame


def draw_possession_halo(frame: np.ndarray, rec: dict[str, Any]) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in rec["bbox"]]
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round(y2))
    radius = max(18, int(round(max(x2 - x1, y2 - y1) * 0.38)))
    cv2.circle(frame, (cx, cy), radius + 4, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, (0, 220, 255), 4, cv2.LINE_AA)
    y_text = min(frame.shape[0] - 10, y2 + 24)
    cv2.putText(frame, "BALL", (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, "BALL", (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)


def draw_possession_link(frame: np.ndarray, possession: dict[str, Any] | None) -> None:
    if not possession:
        return
    owner = possession.get("owner")
    ball = possession.get("ball")
    if not isinstance(owner, dict) or not isinstance(ball, dict):
        return
    owner_anchor = owner.get("anchor_px")
    ball_anchor = ball.get("anchor_px")
    if not isinstance(owner_anchor, list) or not isinstance(ball_anchor, list):
        return
    p1 = (int(round(float(ball_anchor[0]))), int(round(float(ball_anchor[1]))))
    p2 = (int(round(float(owner_anchor[0]))), int(round(float(owner_anchor[1]))))
    cv2.line(frame, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, p1, 4, (0, 220, 255), -1, cv2.LINE_AA)


def draw_possession_hud(frame: np.ndarray, possession: dict[str, Any] | None, debug: bool = False) -> None:
    possession = possession or {"state": "undetected"}
    state = str(possession.get("state") or "undetected")

    if state == "owned":
        line2 = hud_owner_label(possession)
    elif state == "undetected_hold":
        missing = possession.get("missing_frames")
        suffix = f" oculto {missing}f" if missing else " oculto"
        line2 = f"{hud_owner_label(possession)} ({suffix})"
    elif state == "flight":
        line2 = "Balon en vuelo"
    elif state == "loose":
        line2 = "Balon suelto"
    else:
        line2 = "Balon no detectado"

    lines = ["POSESION", line2]
    if possession.get("team"):
        lines.append(f"Equipo: {possession['team']}")

    detail = []
    if possession.get("confidence") is not None:
        detail.append(f"conf {float(possession['confidence']):.2f}")
    if possession.get("distance_m") is not None:
        detail.append(f"dist {float(possession['distance_m']):.2f}m")
    if possession.get("source"):
        detail.append(str(possession["source"]))
    if detail:
        lines.append(" | ".join(detail))

    if debug:
        candidates = possession.get("candidates") or []
        if candidates:
            lines.append("Candidatos:")
            for cand in candidates[:3]:
                label = cand.get("jersey_number") or cand.get("identity") or f"P{cand.get('player_id')}"
                lines.append(f"{label} {cand.get('team', '')} d={cand.get('distance_m')} s={cand.get('score')} c={cand.get('confidence')}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.56
    thickness = 2
    sizes = [cv2.getTextSize(line, font, 0.62 if i == 0 else font_scale, thickness)[0] for i, line in enumerate(lines)]
    box_w = min(max(w for w, _ in sizes) + 28, int(frame.shape[1] * 0.58))
    line_h = 24
    box_h = 18 + line_h * len(lines)
    x0 = max(12, frame.shape[1] - box_w - 18)
    y0 = 18

    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (18, 18, 18), -1)
    border = (0, 220, 255) if state in {"owned", "undetected_hold"} else (180, 180, 180)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), border, 2)

    for i, line in enumerate(lines):
        scale = 0.62 if i == 0 else font_scale
        color = (0, 220, 255) if i == 0 else (245, 245, 245)
        y = y0 + 24 + i * line_h
        cv2.putText(frame, line, (x0 + 14, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x0 + 14, y), font, scale, color, thickness, cv2.LINE_AA)


def hud_owner_label(possession: dict[str, Any]) -> str:
    if possession.get("jersey_number"):
        return f"Jugador #{possession['jersey_number']}"
    if possession.get("identity"):
        return f"Jugador {possession['identity']}"
    if possession.get("player_id") is not None:
        return f"Jugador P{possession['player_id']}"
    return "Jugador desconocido"


if __name__ == "__main__":
    main()
