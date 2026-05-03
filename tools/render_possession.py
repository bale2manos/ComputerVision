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
    best_possession_for_frame,
    build_possession_by_frame,
    build_possession_timeline,
)
from basketball_cv.possession_balanced import assign_balanced_ball_ownership
from tools.analyze_video import (
    build_ball_trails_by_frame,
    build_pass_active_by_frame,
    build_toasts_by_frame,
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

TEAM_PALETTE = {
    "team_a": (20, 190, 235),
    "team_b": (235, 130, 35),
    "red": (40, 40, 230),
    "dark": (40, 40, 40),
    "blue": (220, 95, 35),
    "light": (235, 235, 235),
    "referee": (190, 80, 220),
    "unknown": (160, 160, 160),
    None: (160, 160, 160),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tracks with cleaner visual HUD, OCR labels and minimap.")
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
    parser.add_argument("--no-possession-hud", action="store_true", help="Disable the persistent top-right possession HUD.")
    parser.add_argument("--no-minimap", action="store_true", help="Disable the bottom-right minimap.")
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
    ownership_report = assign_balanced_ball_ownership(records, fps)

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
        show_minimap=not args.no_minimap,
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
    show_minimap: bool = True,
    debug_possession: bool = False,
) -> None:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_frame.setdefault(int(rec["frame_index"]), []).append(rec)
    if not by_frame:
        return

    toasts_by_frame = build_toasts_by_frame(events or [], fps)
    pass_active_by_frame = build_pass_active_by_frame(events or [])
    ball_trails_by_frame = build_ball_trails_by_frame(records, trail_frames=18)
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
            show_minimap=show_minimap,
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
    show_minimap: bool = True,
    debug_possession: bool = False,
) -> np.ndarray:
    has_projection = any(rec.get("court_x") is not None and rec.get("court_y") is not None for rec in records)
    minimap = draw_topdown_court(court_spec, pixels_per_meter=24, margin_px=18, show_labels=False) if has_projection and show_minimap else None
    possession = possession or best_possession_for_frame(records)

    if ball_trail:
        draw_ball_trail(frame, ball_trail)
    draw_possession_link(frame, possession)
    draw_possession_ball_ring(frame, possession)

    # Draw people first, then balls on top.
    people = [rec for rec in records if rec.get("class_name") == "person"]
    balls = [rec for rec in records if rec.get("class_name") == "sports ball"]
    for rec in sorted(people, key=lambda r: bool(r.get("has_ball"))):
        draw_person_record(frame, rec)
        if minimap is not None:
            draw_record_on_minimap(minimap, rec, court_spec, team_color(rec), possession)
    for rec in balls:
        draw_ball_record(frame, rec)
        if minimap is not None:
            draw_record_on_minimap(minimap, rec, court_spec, (0, 210, 255), possession)

    if toasts:
        draw_toasts(frame, toasts)
    if show_possession_hud:
        draw_possession_hud(frame, possession, debug=debug_possession)
    if minimap is not None:
        paste_minimap(frame, minimap)
    draw_corner_watermark(frame)
    return frame


def draw_person_record(frame: np.ndarray, rec: dict[str, Any]) -> None:
    bbox = rec.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    color = team_color(rec)
    has_ball = bool(rec.get("has_ball"))
    role = str(rec.get("role") or "")
    thickness = 4 if has_ball else 2

    # Subtle translucent bbox fill for the ball owner only.
    if has_ball:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame[:] = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
        cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 220, 255), 3, cv2.LINE_AA)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    label = person_label(rec)
    sub = identity_source_label(rec)
    draw_tag(frame, x1, y1 - 9, label, color, sub=sub)
    if has_ball:
        draw_small_badge(frame, x1, min(frame.shape[0] - 10, y2 + 21), "BALL", (0, 220, 255))
    elif role == "referee" or rec.get("team") == "referee":
        draw_small_badge(frame, x1, min(frame.shape[0] - 10, y2 + 21), "REF", color)


def draw_ball_record(frame: np.ndarray, rec: dict[str, Any]) -> None:
    color = (0, 210, 255)
    draw_ball_marker(frame, rec, color)
    bbox = rec.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    label = "ball est" if rec.get("dense_ball_estimate") else "ball"
    if rec.get("ball_owner_jersey_number"):
        label += f" -> #{rec['ball_owner_jersey_number']}"
    elif rec.get("ball_owner_player_id"):
        label += f" -> P{rec['ball_owner_player_id']}"
    draw_tag(frame, x1, y1 - 7, label, color)


def person_label(rec: dict[str, Any]) -> str:
    team = short_team(rec.get("team"))
    if rec.get("jersey_number"):
        base = f"#{rec['jersey_number']}"
    elif rec.get("player_id") is not None:
        base = f"P{rec['player_id']}"
    elif rec.get("track_id") is not None:
        base = f"T{rec['track_id']}"
    else:
        base = "player"
    if rec.get("identity_override"):
        base += " ✓"
    return f"{base} {team}".strip()


def identity_source_label(rec: dict[str, Any]) -> str | None:
    if rec.get("identity_override"):
        return "manual"
    source = rec.get("identity_source")
    if source == "ocr_track_segment":
        return "ocr"
    if source == "jersey_backfill_motion":
        return "ocr+track"
    if rec.get("jersey_number"):
        return "ocr"
    return None


def short_team(team: Any) -> str:
    if team == "team_a":
        return "A"
    if team == "team_b":
        return "B"
    if team == "referee":
        return "REF"
    if team in (None, "unknown"):
        return ""
    return str(team)


def team_color(rec: dict[str, Any]) -> tuple[int, int, int]:
    team = rec.get("team")
    if rec.get("role") == "referee":
        team = "referee"
    return TEAM_PALETTE.get(team, TEAM_PALETTE.get("unknown"))


def draw_tag(frame: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int], sub: str | None = None) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(text)
    scale = 0.62
    th = 2
    (tw, th_text), _ = cv2.getTextSize(text, font, scale, th)
    sub_w = 0
    if sub:
        (sub_w, _), _ = cv2.getTextSize(sub, font, 0.42, 1)
    w = max(tw, sub_w) + 16
    h = 28 + (16 if sub else 0)
    x = max(4, min(x, frame.shape[1] - w - 4))
    y = max(h + 4, y)
    y0 = y - h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y0), (x + w, y), (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.rectangle(frame, (x, y0), (x + w, y), color, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 8, y0 + 21), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (x + 8, y0 + 38), font, 0.42, (210, 210, 210), 1, cv2.LINE_AA)


def draw_small_badge(frame: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
    w, h = tw + 14, 24
    x = max(4, min(x, frame.shape[1] - w - 4))
    cv2.rectangle(frame, (x, y - h), (x + w, y), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y - h), (x + w, y), color, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x + 7, y - 7), font, 0.55, color, 2, cv2.LINE_AA)


def paste_minimap(frame: np.ndarray, minimap: np.ndarray) -> None:
    mh, mw = minimap.shape[:2]
    scale = min(0.31 * frame.shape[1] / mw, 0.27 * frame.shape[0] / mh)
    minimap = cv2.resize(minimap, (int(mw * scale), int(mh * scale)), interpolation=cv2.INTER_AREA)
    mh, mw = minimap.shape[:2]
    x0 = frame.shape[1] - mw - 18
    y0 = frame.shape[0] - mh - 18
    pad = 10
    panel = frame[y0 - pad : y0 + mh + pad, x0 - pad : x0 + mw + pad].copy()
    dark = panel.copy()
    dark[:] = (18, 18, 18)
    panel = cv2.addWeighted(panel, 0.25, dark, 0.75, 0)
    frame[y0 - pad : y0 + mh + pad, x0 - pad : x0 + mw + pad] = panel
    frame[y0 : y0 + mh, x0 : x0 + mw] = cv2.addWeighted(frame[y0 : y0 + mh, x0 : x0 + mw], 0.12, minimap, 0.88, 0)
    cv2.rectangle(frame, (x0 - pad, y0 - pad), (x0 + mw + pad, y0 + mh + pad), (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, "MINIMAP", (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)


def draw_record_on_minimap(
    minimap: np.ndarray,
    rec: dict[str, Any],
    court_spec: CourtSpec,
    color: tuple[int, int, int],
    possession: dict[str, Any] | None,
) -> None:
    if rec.get("court_x") is None or rec.get("court_y") is None:
        return

    is_possessed_ball = rec.get("class_name") == "sports ball" and rec.get("ball_owner_player_id") is not None
    has_ball = bool(rec.get("has_ball")) and rec.get("class_name") == "person"
    court_x = rec.get("court_x")
    court_y = rec.get("court_y")
    if is_possessed_ball and rec.get("possession_court_x") is not None and rec.get("possession_court_y") is not None:
        court_x = float(rec["possession_court_x"]) + 0.18
        court_y = float(rec["possession_court_y"]) + 0.18

    pt = court_to_canvas(np.asarray([[court_x, court_y]], dtype=np.float32), court_spec, pixels_per_meter=24, margin_px=18)[0]
    x, y = int(pt[0]), int(pt[1])
    if rec.get("class_name") == "sports ball":
        cv2.circle(minimap, (x, y), 6, (0, 210, 255), -1, cv2.LINE_AA)
        cv2.circle(minimap, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
        return

    radius = 8 if has_ball else 6
    cv2.circle(minimap, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(minimap, (x, y), radius, color, -1, cv2.LINE_AA)
    if has_ball:
        cv2.circle(minimap, (x, y), radius + 5, (0, 210, 255), 2, cv2.LINE_AA)
    label_id = rec.get("jersey_number") or (rec.get("player_id") if rec.get("player_id") is not None else rec.get("track_id"))
    if label_id is not None:
        cv2.putText(minimap, str(label_id), (x + 8, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(minimap, str(label_id), (x + 8, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_possession_ball_ring(frame: np.ndarray, possession: dict[str, Any] | None) -> None:
    if not possession or possession.get("state") not in {"owned", "undetected_hold"}:
        return
    ball = possession.get("ball")
    if not isinstance(ball, dict) or not isinstance(ball.get("bbox"), list):
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in ball["bbox"]]
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    radius = max(16, int(round(max(x2 - x1, y2 - y1) * 1.45)))
    cv2.circle(frame, (cx, cy), radius + 3, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, (0, 220, 255), 3, cv2.LINE_AA)


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
        line2 = f"{hud_owner_label(possession)} (oculto {missing}f)" if missing else f"{hud_owner_label(possession)} (oculto)"
    elif state == "flight":
        line2 = "Balon en vuelo"
    elif state == "loose":
        line2 = "Balon suelto"
    else:
        line2 = "Balon no detectado"

    lines = ["POSESION", line2]
    if possession.get("team"):
        lines.append(f"Equipo: {short_team(possession.get('team')) or possession.get('team')}")
    detail = []
    if possession.get("confidence") is not None:
        detail.append(f"conf {float(possession['confidence']):.2f}")
    if possession.get("assignment_reason"):
        detail.append(str(possession["assignment_reason"]))
    if detail:
        lines.append(" | ".join(detail))
    if debug and possession.get("candidates"):
        lines.append("Candidatos:")
        for cand in possession.get("candidates", [])[:3]:
            label = cand.get("jersey_number") or cand.get("identity") or f"P{cand.get('player_id')}"
            lines.append(f"{label} {cand.get('team', '')} d={cand.get('distance_m')} s={cand.get('score')}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(line, font, 0.64 if i == 0 else 0.56, 2)[0] for i, line in enumerate(lines)]
    box_w = min(max(w for w, _ in sizes) + 30, int(frame.shape[1] * 0.62))
    line_h = 25
    box_h = 20 + line_h * len(lines)
    x0 = frame.shape[1] - box_w - 18
    y0 = 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)
    border = (0, 220, 255) if state in {"owned", "undetected_hold"} else (180, 180, 180)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), border, 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        scale = 0.64 if i == 0 else 0.56
        color = (0, 220, 255) if i == 0 else (245, 245, 245)
        y = y0 + 25 + i * line_h
        cv2.putText(frame, line, (x0 + 14, y), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x0 + 14, y), font, scale, color, 2, cv2.LINE_AA)


def hud_owner_label(possession: dict[str, Any]) -> str:
    if possession.get("jersey_number"):
        return f"Jugador #{possession['jersey_number']}"
    if possession.get("identity"):
        return f"Jugador {possession['identity']}"
    if possession.get("player_id") is not None:
        return f"Jugador P{possession['player_id']}"
    return "Jugador desconocido"


def draw_corner_watermark(frame: np.ndarray) -> None:
    cv2.putText(frame, "Basketball CV", (14, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "Basketball CV", (14, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
