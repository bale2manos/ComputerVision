from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.court import (
    CourtSpec,
    court_to_canvas,
    draw_topdown_court,
    load_calibration,
    project_points,
)
from basketball_cv.events import (
    assign_ball_ownership,
    densify_ball_track_for_render,
    detect_passes,
    detect_pick_and_rolls,
    interpolate_ball_gaps,
)
from basketball_cv.player_gaps import interpolate_player_gaps, smooth_player_positions
from basketball_cv.teams import (
    UNKNOWN_TEAM,
    bgr_to_hsv,
    is_player_like_jersey,
    jersey_color,
    jersey_embedding,
    jersey_stats,
    player_appearance_embedding,
    split_mixed_team_tracks,
    stabilize_team_identity,
)
from basketball_cv.tracks import resolve_crossing_id_switches, stitch_track_fragments, summarize_players_from_records


COCO_NAMES = {0: "person", 32: "sports ball"}
TEAM_COLORS = {
    "red": (40, 40, 230),
    "dark": (45, 45, 45),
    "blue": (220, 80, 30),
    "green": (60, 170, 60),
    "orange": (35, 140, 240),
    "purple": (170, 60, 170),
    "light": (225, 225, 225),
    "team_a": (30, 180, 220),
    "team_b": (220, 120, 40),
    UNKNOWN_TEAM: (180, 180, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track basketball players/ball and export a top-down minimap.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--calibration", default=None, help="Court calibration JSON from tools/calibrate_court.py.")
    parser.add_argument("--output-dir", default="runs/sample", help="Directory for annotated video and JSON outputs.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model.")
    parser.add_argument("--use-nms-head", action="store_true", help="For YOLO26, use the one-to-many head with NMS instead of the default end-to-end head.")
    parser.add_argument("--ball-model", default=None, help="Optional single-class YOLO model trained only for the ball.")
    parser.add_argument("--ball-conf", type=float, default=0.2, help="Confidence threshold for --ball-model.")
    parser.add_argument("--ball-imgsz", type=int, default=None, help="Image size for --ball-model. Defaults to --imgsz.")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Ultralytics tracker config.")
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu, 0.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.2, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="Detector IoU threshold.")
    parser.add_argument("--court-margin", type=float, default=0.9, help="Default court margin in meters for keeping projected detections near the court.")
    parser.add_argument("--court-near-margin", type=float, default=None, help="Optional margin in meters below court_y=0.")
    parser.add_argument("--court-far-margin", type=float, default=0.6, help="Margin in meters beyond the far sideline court_y=width.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for quick tests.")
    parser.add_argument("--team-warmup-frames", type=int, default=180, help="Frames to collect jersey colors before team labels.")
    parser.add_argument("--in-play-players", type=int, default=10, help="Number of player tracks to mark as in-play per frame.")
    parser.add_argument("--keep-truncated-players", action="store_true", help="Keep person detections touching the bottom frame border.")
    parser.add_argument("--no-split-mixed-tracks", action="store_true", help="Disable splitting raw tracks with sustained uniform changes.")
    parser.add_argument("--mixed-track-min-segment-frames", type=int, default=24, help="Minimum sustained uniform frames before splitting a raw track.")
    parser.add_argument("--no-stitch-tracks", action="store_true", help="Disable post-process merging of raw tracker fragments.")
    parser.add_argument("--stitch-max-gap", type=float, default=1.6, help="Max seconds between fragments to merge.")
    parser.add_argument("--stitch-max-speed", type=float, default=5.8, help="Max implied movement speed in m/s for track stitching.")
    parser.add_argument("--stitch-min-embedding-sim", type=float, default=0.78, help="Minimum jersey embedding similarity for track stitching.")
    parser.add_argument("--no-correct-crossings", action="store_true", help="Disable same-team crossing ID-switch correction.")
    parser.add_argument("--crossing-close-distance", type=float, default=1.25, help="Max court distance in meters for a crossing candidate.")
    parser.add_argument("--crossing-lookaround", type=float, default=0.35, help="Seconds before/after a close crossing to compare trajectory continuity.")
    parser.add_argument("--crossing-min-improvement", type=float, default=0.45, help="Minimum trajectory improvement in meters required to swap IDs.")
    parser.add_argument("--crossing-min-appearance-improvement", type=float, default=0.18, help="Minimum visual-similarity improvement required to swap same-team IDs.")
    parser.add_argument("--crossing-max-appearance-motion-penalty", type=float, default=1.25, help="Max 2D continuity penalty allowed for appearance-only crossing swaps.")
    parser.add_argument("--write-video", action="store_true", help="Write an annotated MP4.")
    parser.add_argument("--ball-color-fallback", action="store_true", help="Enable experimental orange blob fallback for the ball.")
    parser.add_argument("--no-dense-ball-track", action="store_true", help="Do not add render-only ball estimates for every frame.")
    parser.add_argument("--dense-ball-max-gap", type=float, default=3.0, help="Max seconds between ball detections to linearly connect in the dense render track.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(args.calibration) if args.calibration else None
    court_spec = CourtSpec(**calibration["court"]) if calibration else CourtSpec()
    homography = calibration["homography"] if calibration else None

    meta = video_meta(args.video)
    fps = meta["fps"]
    model = YOLO(args.model)
    ball_model = YOLO(args.ball_model) if args.ball_model else None

    records: list[dict[str, Any]] = []
    last_positions: dict[int, deque[tuple[int, float, float]]] = {}
    stats = {"frames": 0, "persons": 0, "balls": 0, "projected": 0}

    track_kwargs = {
        "source": args.video,
        "stream": True,
        "persist": True,
        "tracker": args.tracker,
        "classes": [0, 32],
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "verbose": False,
    }
    if args.use_nms_head:
        track_kwargs["end2end"] = False
    results = model.track(**track_kwargs)

    for frame_index, result in enumerate(results):
        if args.max_frames and frame_index >= args.max_frames:
            break
        frame = result.orig_img.copy()

        detections = parse_detections(result)
        if ball_model is not None:
            detections = [d for d in detections if d["class_id"] != 32]
            ball_result = ball_model.predict(
                frame,
                imgsz=args.ball_imgsz or args.imgsz,
                conf=args.ball_conf,
                device=args.device,
                verbose=False,
            )[0]
            detections.extend(parse_ball_model_detections(ball_result))
        if args.ball_color_fallback and not any(d["class_id"] == 32 for d in detections):
            ball = detect_orange_ball(frame)
            if ball:
                detections.append(ball)
        detections = dedupe_ball_detections(detections)
        frame_records = enrich_detections(
            frame,
            frame_index,
            detections,
            homography,
            court_spec,
            None,
            [],
            0,
            last_positions,
            fps,
            args.keep_truncated_players,
            args.court_margin,
            args.court_near_margin if args.court_near_margin is not None else args.court_margin,
            args.court_far_margin,
        )
        records.extend(frame_records)

        stats["frames"] += 1
        stats["persons"] += sum(1 for r in frame_records if r["class_name"] == "person")
        stats["balls"] += sum(1 for r in frame_records if r["class_name"] == "sports ball")
        stats["projected"] += sum(1 for r in frame_records if r["court_x"] is not None)

    track_report = stabilize_team_identity(records)
    if not args.no_split_mixed_tracks:
        track_report = split_mixed_team_tracks(records, track_report, min_segment_frames=args.mixed_track_min_segment_frames)
    stitch_report = {"enabled": False, "player_count": track_report["candidate_track_count"], "merged_track_count": 0, "players": []}
    if not args.no_stitch_tracks:
        stitch_report = stitch_track_fragments(
            records,
            track_report,
            fps=fps,
            max_gap_s=args.stitch_max_gap,
            max_speed_mps=args.stitch_max_speed,
            min_embedding_similarity=args.stitch_min_embedding_sim,
        )
    crossing_report = {"enabled": False, "correction_count": 0, "corrections": []}
    if not args.no_stitch_tracks and not args.no_correct_crossings:
        crossing_report = resolve_crossing_id_switches(
            records,
            fps=fps,
            close_distance_m=args.crossing_close_distance,
            lookaround_s=args.crossing_lookaround,
            min_improvement_m=args.crossing_min_improvement,
            min_appearance_improvement=args.crossing_min_appearance_improvement,
            max_appearance_motion_penalty_m=args.crossing_max_appearance_motion_penalty,
        )
        stitch_report = summarize_players_from_records(records, stitch_report, crossing_report)
    gap_report = interpolate_player_gaps(records, fps=fps, max_gap_frames=max(2, int(round(0.32 * fps))))
    smooth_player_positions(records, window=5)
    mark_in_play_players(records, max_players=args.in_play_players)
    update_track_report_in_play(track_report, records)
    ball_interpolation_report = interpolate_ball_gaps(records, fps)
    ball_ownership_report = assign_ball_ownership(records, fps)
    pass_events = detect_passes(records, fps)
    pick_and_roll_events = detect_pick_and_rolls(records, fps)
    events = pass_events + pick_and_roll_events
    events.sort(key=lambda event: int(event.get("start_frame", 0)))
    dense_ball_report = {"dense_ball_frames": 0, "dense_ball_estimated_frames": 0}
    if not args.no_dense_ball_track:
        dense_ball_report = densify_ball_track_for_render(
            records,
            fps,
            frame_count=stats["frames"],
            max_linear_gap_s=args.dense_ball_max_gap,
        )

    if args.write_video:
        render_annotated_video(args.video, output_dir / "annotated.mp4", records, fps, court_spec, events=events)

    write_json(output_dir / "tracks.json", {"video": args.video, "fps": fps, "frame_count": stats["frames"], "records": records})
    write_json(output_dir / "events.json", {"video": args.video, "fps": fps, "events": events})
    write_json(
        output_dir / "ball_tracks.json",
        {
            "video": args.video,
            "fps": fps,
            "summary": {**ball_interpolation_report, **ball_ownership_report, **dense_ball_report},
            "records": [rec for rec in records if rec.get("class_name") == "sports ball"],
        },
    )
    write_json(output_dir / "track_summary.json", {"video": args.video, "fps": fps, **track_report})
    write_json(output_dir / "player_summary.json", {"video": args.video, "fps": fps, **stitch_report})
    write_json(
        output_dir / "summary.json",
        {
            "video": args.video,
            "fps": fps,
            "frames_processed": stats["frames"],
            "person_detections": stats["persons"],
            "ball_detections": stats["balls"],
            "projected_detections": stats["projected"],
            "in_play_player_detections": sum(1 for rec in records if rec.get("in_play_player")),
            "stable_player_tracks": track_report["candidate_track_count"],
            "mixed_track_splits": track_report.get("mixed_track_split_count", 0),
            "stitched_players": stitch_report["player_count"],
            "merged_track_fragments": stitch_report["merged_track_count"],
            "id_switch_corrections": crossing_report["correction_count"],
            "short_gap_fills": gap_report["short_gap_fills"],
            "estimated_player_frames": gap_report["estimated_player_frames"],
            "total_person_tracks": track_report["track_count"],
            "pick_and_roll_candidates": len(pick_and_roll_events),
            "calibrated": homography is not None,
            "team_identity_ready": track_report["candidate_track_count"] > 0,
            "team_identity_mode": track_report.get("identity_mode"),
            "ball_owned_frames": ball_ownership_report["owned_ball_frames"],
            "interpolated_ball_frames": ball_interpolation_report["interpolated_ball_frames"],
            "dense_ball_estimated_frames": dense_ball_report["dense_ball_estimated_frames"],
            "ball_unique_owners": ball_ownership_report["unique_owners"],
            "pass_candidates": len(pass_events),
        },
    )
    print(json.dumps(json.loads((output_dir / "summary.json").read_text(encoding="utf-8")), indent=2))


def video_meta(video_path: str) -> dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return {"fps": float(fps), "frames": float(frames)}


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def parse_detections(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(cls), -1)
    detections = []
    for box, score, class_id, track_id in zip(xyxy, conf, cls, ids):
        detections.append(
            {
                "xyxy": box.astype(float),
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": COCO_NAMES.get(int(class_id), str(class_id)),
                "track_id": int(track_id) if int(track_id) >= 0 else None,
                "source": "yolo",
            }
        )
    return detections


def parse_ball_model_detections(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    detections = []
    for box, score in zip(xyxy, conf):
        detections.append(
            {
                "xyxy": box.astype(float),
                "confidence": float(score),
                "class_id": 32,
                "class_name": "sports ball",
                "track_id": None,
                "source": "ball_model",
            }
        )
    return detections


def dedupe_ball_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.45) -> list[dict[str, Any]]:
    balls = sorted([d for d in detections if d["class_id"] == 32], key=lambda d: d["confidence"], reverse=True)
    kept_balls: list[dict[str, Any]] = []
    for ball in balls:
        if all(box_iou(ball["xyxy"], kept["xyxy"]) < iou_threshold for kept in kept_balls):
            kept_balls.append(ball)
    return [d for d in detections if d["class_id"] != 32] + kept_balls


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_orange_ball(frame: np.ndarray) -> dict[str, Any] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([6, 75, 65]), np.array([26, 255, 255]))
    h_img, w_img = frame.shape[:2]
    mask[: int(0.27 * h_img), :] = 0
    mask[: int(0.42 * h_img), : int(0.08 * w_img)] = 0
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 45 or area > 1250:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w / 2.0
        cy = y + h / 2.0
        if cy < 0.27 * h_img or (cx < 0.08 * w_img and cy < 0.42 * h_img):
            continue
        if w < 6 or h < 6 or w > 55 or h > 55:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.45 or aspect > 2.2:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.28:
            continue

        crop = hsv[y : y + h, x : x + w]
        hue = crop[:, :, 0][mask[y : y + h, x : x + w] > 0]
        if len(hue) == 0:
            continue
        hue_score = 1.0 - min(abs(float(np.median(hue)) - 13.0), 20.0) / 20.0
        score = circularity * 0.55 + min(area / 900.0, 1.0) * 0.25 + hue_score * 0.2
        candidates.append((score, contour, (x, y, w, h)))

    if not candidates:
        return None
    score, _contour, (x, y, w, h) = max(candidates, key=lambda item: item[0])
    pad = max(2, int(round(max(w, h) * 0.18)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1] - 1, x + w + pad)
    y2 = min(frame.shape[0] - 1, y + h + pad)
    return {
        "xyxy": np.asarray([x1, y1, x2, y2], dtype=float),
        "confidence": float(min(0.65, max(0.2, score))),
        "class_id": 32,
        "class_name": "sports ball",
        "track_id": None,
        "source": "orange_blob",
    }


def enrich_detections(
    frame: np.ndarray,
    frame_index: int,
    detections: list[dict[str, Any]],
    homography: np.ndarray | None,
    court_spec: CourtSpec,
    team_model: Any,
    color_samples: list[np.ndarray],
    warmup_frames: int,
    last_positions: dict[int, deque[tuple[int, float, float]]],
    fps: float,
    keep_truncated_players: bool,
    court_margin_m: float,
    court_near_margin_m: float,
    court_far_margin_m: float,
) -> list[dict[str, Any]]:
    records = []
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        bottom_truncated = bool(det["class_id"] == 0 and y2 >= frame.shape[0] - 2)
        if bottom_truncated and not keep_truncated_players:
            continue
        bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        anchor = ((x1 + x2) / 2.0, y2) if det["class_id"] == 0 else ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        court_x = court_y = None
        in_court = None
        if homography is not None:
            projected = project_points(np.asarray([anchor], dtype=np.float32), homography)[0]
            in_court = bool(
                inside_play_area(
                    projected,
                    court_spec,
                    x_margin_m=court_margin_m,
                    near_margin_m=court_near_margin_m,
                    far_margin_m=court_far_margin_m,
                )
            )
            if not in_court:
                continue
            court_x, court_y = float(projected[0]), float(projected[1])

        team = None
        speed_mps = None
        color = None
        color_hsv = None
        uniform_stats = None
        embedding = None
        appearance_embedding = None
        if det["class_id"] == 0:
            color = jersey_color(frame, det["xyxy"])
            color_hsv = bgr_to_hsv(color)
            uniform_stats = jersey_stats(frame, det["xyxy"])
            embedding = jersey_embedding(frame, det["xyxy"])
            appearance_embedding = player_appearance_embedding(frame, det["xyxy"])
            sample_candidate = should_sample_team_color(frame, det, bbox_area, homography is not None, bool(in_court))
            if color is not None and frame_index < warmup_frames and sample_candidate and is_player_like_jersey(color_hsv):
                color_samples.append(color)
            team = team_model.predict(color) if team_model else UNKNOWN_TEAM
            player_candidate = bool(team != UNKNOWN_TEAM and is_player_like_jersey(color_hsv))
            track_id = det["track_id"]
            if track_id is not None and court_x is not None and court_y is not None:
                history = last_positions.setdefault(track_id, deque(maxlen=12))
                if history:
                    prev_frame, prev_x, prev_y = history[0]
                    dt = max((frame_index - prev_frame) / fps, 1e-6)
                    speed_mps = float(np.hypot(court_x - prev_x, court_y - prev_y) / dt)
                history.append((frame_index, court_x, court_y))

        records.append(
            {
                "frame_index": frame_index,
                "time_s": round(frame_index / fps, 4),
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "track_id": det["track_id"],
                "source": det.get("source", "unknown"),
                "confidence": round(det["confidence"], 4),
                "bbox": [round(float(v), 2) for v in det["xyxy"]],
                "bbox_area": round(float(bbox_area), 2),
                "anchor_px": [round(float(anchor[0]), 2), round(float(anchor[1]), 2)],
                "court_x": round(court_x, 3) if court_x is not None else None,
                "court_y": round(court_y, 3) if court_y is not None else None,
                "on_court": in_court,
                "bottom_truncated": bottom_truncated,
                "team": team,
                "player_candidate": bool(det["class_id"] == 0 and player_candidate),
                "in_play_player": False,
                "jersey_bgr": [round(float(v), 2) for v in color] if color is not None else None,
                "jersey_hsv": [round(float(v), 2) for v in color_hsv] if color_hsv is not None else None,
                "jersey_stats": uniform_stats,
                "jersey_embedding": [round(float(v), 5) for v in embedding] if embedding is not None else None,
                "appearance_embedding": [round(float(v), 5) for v in appearance_embedding] if appearance_embedding is not None else None,
                "speed_mps": round(speed_mps, 3) if speed_mps is not None else None,
            }
        )
    return records


def inside_play_area(
    point_m: np.ndarray,
    court_spec: CourtSpec,
    x_margin_m: float,
    near_margin_m: float,
    far_margin_m: float,
) -> bool:
    x, y = [float(v) for v in np.asarray(point_m, dtype=np.float32).reshape(2)]
    return (
        -x_margin_m <= x <= court_spec.length_m + x_margin_m
        and -near_margin_m <= y <= court_spec.width_m + far_margin_m
    )


def should_sample_team_color(
    frame: np.ndarray,
    det: dict[str, Any],
    bbox_area: float,
    calibrated: bool,
    in_court: bool,
) -> bool:
    if det["class_id"] != 0 or det["confidence"] < 0.35:
        return False
    if calibrated and not in_court:
        return False

    x1, y1, x2, y2 = det["xyxy"]
    h, w = frame.shape[:2]
    box_h = y2 - y1
    box_w = x2 - x1
    bottom = y2 / max(h, 1)
    area_ratio = bbox_area / max(float(w * h), 1.0)

    if bottom < 0.32:
        return False
    if box_h < 70 or box_w < 20:
        return False
    if area_ratio < 0.0012:
        return False
    if box_h / max(box_w, 1.0) < 1.1:
        return False
    return True


def relabel_unknown_teams(records: list[dict[str, Any]], team_model: Any) -> None:
    for rec in records:
        if rec.get("class_name") != "person" or not rec.get("jersey_bgr"):
            continue
        team = team_model.predict(np.asarray(rec["jersey_bgr"], dtype=np.float32))
        rec["team"] = team
        color_hsv = np.asarray(rec.get("jersey_hsv"), dtype=np.float32) if rec.get("jersey_hsv") else None
        rec["player_candidate"] = bool(team != UNKNOWN_TEAM and is_player_like_jersey(color_hsv))


def mark_in_play_players(records: list[dict[str, Any]], max_players: int = 10) -> None:
    if max_players <= 0:
        return

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        rec["in_play_player"] = False
        if rec.get("class_name") != "person" or not rec.get("player_candidate"):
            continue
        if rec.get("track_id") is None:
            continue
        if rec.get("on_court") is False:
            continue
        by_frame.setdefault(int(rec["frame_index"]), []).append(rec)

    for frame_records in by_frame.values():
        teams = sorted({r.get("team") for r in frame_records if r.get("team") not in (None, UNKNOWN_TEAM)})
        selected: list[dict[str, Any]] = []
        if len(teams) >= 2 and max_players >= 10:
            per_team = max_players // 2
            for team in teams[:2]:
                selected.extend(_rank_player_candidates([r for r in frame_records if r.get("team") == team])[:per_team])

        if len(selected) < max_players:
            selected_ids = {id(r) for r in selected}
            rest = [r for r in frame_records if id(r) not in selected_ids]
            selected.extend(_rank_player_candidates(rest)[: max_players - len(selected)])

        for rec in selected[:max_players]:
            rec["in_play_player"] = True


def update_track_report_in_play(track_report: dict[str, Any], records: list[dict[str, Any]]) -> None:
    counts: dict[int, int] = {}
    player_ids: dict[int, int | None] = {}
    for rec in records:
        if rec.get("track_id") is not None and rec.get("class_name") == "person":
            player_ids[int(rec["track_id"])] = rec.get("player_id")
        if rec.get("in_play_player") and rec.get("track_id") is not None:
            counts[int(rec["track_id"])] = counts.get(int(rec["track_id"]), 0) + 1
    for track in track_report.get("tracks", []):
        track_id = int(track["track_id"])
        track["in_play_frames"] = counts.get(track_id, 0)
        track["player_id"] = player_ids.get(track_id)


def _rank_player_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            float(r.get("confidence") or 0.0),
            float(r.get("bbox_area") or 0.0),
            1.0 if r.get("speed_mps") is not None else 0.0,
        ),
        reverse=True,
    )


def render_annotated_video(
    video_path: str,
    output_path: Path,
    records: list[dict[str, Any]],
    fps: float,
    court_spec: CourtSpec,
    events: list[dict[str, Any]] | None = None,
) -> None:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_frame.setdefault(int(rec["frame_index"]), []).append(rec)
    toasts_by_frame = build_toasts_by_frame(events or [], fps)
    pass_active_by_frame = build_pass_active_by_frame(events or [])
    ball_trails_by_frame = build_ball_trails_by_frame(records, trail_frames=24)

    if not by_frame:
        return

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
        writer.write(
            draw_frame(
                frame,
                by_frame.get(frame_index, []),
                court_spec,
                toasts_by_frame.get(frame_index, []),
                pass_active_by_frame.get(frame_index, False),
                ball_trails_by_frame.get(frame_index, []),
            )
        )
        frame_index += 1

    cap.release()
    writer.release()


def build_toasts_by_frame(events: list[dict[str, Any]], fps: float, duration_s: float = 1.8) -> dict[int, list[str]]:
    toasts: dict[int, list[str]] = {}
    duration = max(1, int(round(duration_s * max(fps, 1.0))))
    for event in events:
        if event.get("type") != "pass":
            continue
        text = event.get("toast")
        if not text:
            text = f"Jugador {event.get('from_jersey_number') or event.get('from_player_id')} pasa a jugador {event.get('to_jersey_number') or event.get('to_player_id')}"
        start = int(event.get("end_frame", event.get("start_frame", 0)))
        for frame_index in range(start, start + duration):
            toasts.setdefault(frame_index, []).append(str(text))
    return toasts


def build_pass_active_by_frame(events: list[dict[str, Any]]) -> dict[int, bool]:
    active: dict[int, bool] = {}
    for event in events:
        if event.get("type") != "pass":
            continue
        start = int(event.get("start_frame", 0))
        end = int(event.get("end_frame", start))
        for frame_index in range(start, end + 1):
            active[frame_index] = True
    return active


def build_ball_trails_by_frame(records: list[dict[str, Any]], trail_frames: int = 24) -> dict[int, list[dict[str, Any]]]:
    ball_by_frame: dict[int, dict[str, Any]] = {}
    for rec in records:
        if rec.get("class_name") != "sports ball":
            continue
        frame = int(rec.get("frame_index", 0))
        current = ball_by_frame.get(frame)
        if current is None or float(rec.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
            ball_by_frame[frame] = rec
    if not ball_by_frame:
        return {}

    trails: dict[int, list[dict[str, Any]]] = {}
    ordered_frames = sorted(ball_by_frame)
    for frame in range(min(ordered_frames), max(ordered_frames) + 1):
        trail = []
        for prev_frame in range(max(min(ordered_frames), frame - trail_frames + 1), frame + 1):
            rec = ball_by_frame.get(prev_frame)
            if rec is not None:
                trail.append(rec)
        if trail:
            trails[frame] = trail
    return trails


def draw_frame(
    frame: np.ndarray,
    records: list[dict[str, Any]],
    court_spec: CourtSpec,
    toasts: list[str] | None = None,
    pass_active: bool = False,
    ball_trail: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    has_projection = any(rec["court_x"] is not None and rec["court_y"] is not None for rec in records)
    minimap = draw_topdown_court(court_spec, pixels_per_meter=24, margin_px=18) if has_projection else None
    if ball_trail:
        draw_ball_trail(frame, ball_trail)
    for rec in records:
        color = color_for_record(rec)
        x1, y1, x2, y2 = [int(round(v)) for v in rec["bbox"]]
        thickness = 4 if rec.get("has_ball") and not pass_active else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if rec["class_name"] == "sports ball":
            draw_ball_marker(frame, rec, color)
            label = "ball est" if rec.get("dense_ball_estimate") else "ball"
            if not pass_active and rec.get("ball_owner_jersey_number"):
                label += f" -> {rec['ball_owner_jersey_number']}"
            elif not pass_active and rec.get("ball_owner_player_id"):
                label += f" -> P{rec['ball_owner_player_id']}"
        else:
            label = rec["class_name"]
        if rec["class_name"] != "sports ball":
            if rec.get("jersey_number"):
                label += f" #{rec['jersey_number']}"
            elif rec.get("player_id") is not None:
                label += f" P{rec['player_id']}"
            elif rec.get("track_id") is not None:
                label += f" #{rec['track_id']}"
            if rec.get("team"):
                label += f" {rec['team']}"
            if rec.get("has_ball") and not pass_active:
                label += " ball"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        map_x = rec.get("court_x_smooth", rec.get("court_x"))
        map_y = rec.get("court_y_smooth", rec.get("court_y"))
        if minimap is not None and map_x is not None and map_y is not None:
            pt = court_to_canvas(np.asarray([[map_x, map_y]], dtype=np.float32), court_spec, pixels_per_meter=24, margin_px=18)[0]
            radius = 6 if rec["class_name"] == "person" else 4
            cv2.circle(minimap, (int(pt[0]), int(pt[1])), radius, color, -1)
            label_id = rec.get("jersey_number") or (rec.get("player_id") if rec.get("player_id") is not None else rec.get("track_id"))
            if label_id is not None:
                cv2.putText(minimap, str(label_id), (int(pt[0]) + 7, int(pt[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if toasts:
        draw_toasts(frame, toasts)

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


def draw_ball_marker(frame: np.ndarray, rec: dict[str, Any], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in rec["bbox"]]
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    radius = max(9, int(round(max(x2 - x1, y2 - y1) * 0.55)))
    estimated = bool(rec.get("dense_ball_estimate"))
    marker_color = (0, 190, 255) if estimated else color
    line_thickness = 1 if estimated else 2
    cv2.circle(frame, (cx, cy), radius + 4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius + 2, marker_color, line_thickness, cv2.LINE_AA)
    cv2.line(frame, (cx - radius - 6, cy), (cx + radius + 6, cy), marker_color, line_thickness, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - radius - 6), (cx, cy + radius + 6), marker_color, line_thickness, cv2.LINE_AA)


def draw_ball_trail(frame: np.ndarray, trail: list[dict[str, Any]]) -> None:
    points: list[tuple[int, int, bool]] = []
    for rec in trail:
        anchor = rec.get("anchor_px")
        if not isinstance(anchor, list) or len(anchor) != 2:
            bbox = rec.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            anchor = [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]
        points.append((int(round(float(anchor[0]))), int(round(float(anchor[1]))), bool(rec.get("dense_ball_estimate"))))
    if len(points) < 2:
        return
    total = len(points)
    for idx in range(1, total):
        x1, y1, prev_est = points[idx - 1]
        x2, y2, est = points[idx]
        alpha = idx / max(total - 1, 1)
        color = (0, int(120 + 100 * alpha), 255) if not (prev_est and est) else (0, int(150 + 70 * alpha), 210)
        thickness = 1 if prev_est and est else 2
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if idx % 3 == 0 or idx == total - 1:
            cv2.circle(frame, (x2, y2), 2 if est else 3, color, -1, cv2.LINE_AA)


def draw_toasts(frame: np.ndarray, toasts: list[str]) -> None:
    x0, y0 = 16, 18
    max_width = int(frame.shape[1] * 0.45)
    for index, text in enumerate(toasts[:3]):
        font_scale = 0.62
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if tw > max_width:
            font_scale = max(0.42, font_scale * max_width / max(tw, 1))
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        y = y0 + index * 42
        cv2.rectangle(frame, (x0, y), (x0 + tw + 22, y + th + baseline + 16), (20, 20, 20), -1)
        cv2.rectangle(frame, (x0, y), (x0 + tw + 22, y + th + baseline + 16), (240, 240, 240), 1)
        cv2.putText(frame, text, (x0 + 11, y + th + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def color_for_record(rec: dict[str, Any]) -> tuple[int, int, int]:
    if rec["class_name"] == "sports ball":
        return (20, 130, 255)
    return TEAM_COLORS.get(rec.get("team") or UNKNOWN_TEAM, TEAM_COLORS[UNKNOWN_TEAM])


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
