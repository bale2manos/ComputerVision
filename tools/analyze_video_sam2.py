from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import SAM, YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.court import CourtSpec, load_calibration
from basketball_cv.events import assign_ball_ownership, densify_ball_track_for_render, detect_passes, detect_pick_and_rolls, interpolate_ball_gaps
from basketball_cv.masks import cleanup_disconnected_mask, mask_to_xyxy
from basketball_cv.teams import split_mixed_team_tracks, stabilize_team_identity
from basketball_cv.tracks import resolve_crossing_id_switches, stitch_track_fragments, summarize_players_from_records
from tools.analyze_video import (
    COCO_NAMES,
    dedupe_ball_detections,
    detect_orange_ball,
    enrich_detections,
    mark_in_play_players,
    parse_ball_model_detections,
    parse_detections,
    render_annotated_video,
    update_track_report_in_play,
    video_meta,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2-first basketball tracking from detector boxes.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--calibration", required=True, help="Court calibration JSON.")
    parser.add_argument("--output-dir", default="runs/sam2", help="Directory for outputs.")
    parser.add_argument("--detector-model", default="yolo11m.pt", help="Detector used to prompt SAM.")
    parser.add_argument("--sam-model", default="sam2_b.pt", help="Ultralytics SAM/SAM2 weights.")
    parser.add_argument("--ball-model", default=None, help="Optional single-class YOLO ball model.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=768, help="Detector image size.")
    parser.add_argument("--sam-imgsz", type=int, default=1024, help="SAM image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="Detector IoU threshold.")
    parser.add_argument("--ball-conf", type=float, default=0.2, help="Ball model confidence.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit.")
    parser.add_argument("--track-max-gap", type=int, default=24, help="Max missing frames before retiring a SAM track.")
    parser.add_argument("--track-min-iou", type=float, default=0.1, help="Minimum mask-box IoU for same-track matching.")
    parser.add_argument("--mask-distance", type=float, default=90.0, help="Drop mask islands farther than this from main body.")
    parser.add_argument("--court-margin", type=float, default=0.9, help="Default court margin in meters.")
    parser.add_argument("--court-near-margin", type=float, default=1.2, help="Near sideline margin in meters.")
    parser.add_argument("--court-far-margin", type=float, default=0.5, help="Far sideline margin in meters.")
    parser.add_argument("--ball-color-fallback", action="store_true", help="Enable experimental orange blob ball fallback.")
    parser.add_argument("--no-dense-ball-track", action="store_true", help="Do not add render-only ball estimates for every frame.")
    parser.add_argument("--dense-ball-max-gap", type=float, default=3.0, help="Max seconds between ball detections to linearly connect in the dense render track.")
    parser.add_argument("--write-video", action="store_true", help="Write annotated video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(args.calibration)
    court_spec = CourtSpec(**calibration["court"])
    homography = calibration["homography"]
    meta = video_meta(args.video)
    fps = meta["fps"]

    detector = YOLO(args.detector_model)
    sam = SAM(args.sam_model)
    ball_model = YOLO(args.ball_model) if args.ball_model else None
    tracker = SimpleBoxTracker(max_gap=args.track_max_gap, min_iou=args.track_min_iou)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    records: list[dict[str, Any]] = []
    last_positions: dict[int, deque[tuple[int, float, float]]] = {}
    stats = {"frames": 0, "persons": 0, "balls": 0, "projected": 0, "sam_masks": 0, "mask_cleanups": 0}
    frame_index = 0
    while True:
        if args.max_frames and frame_index >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        result = detector.predict(
            frame,
            classes=[0, 32],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        detections = parse_detections(result)
        person_detections = [det for det in detections if det["class_id"] == 0]
        ball_detections = [det for det in detections if det["class_id"] == 32]

        segmented_people, sam_stats = segment_people_with_sam(
            sam,
            frame,
            person_detections,
            device=args.device,
            imgsz=args.sam_imgsz,
            max_center_distance_px=args.mask_distance,
        )
        tracker.update(segmented_people, frame_index)

        if ball_model is not None:
            ball_result = ball_model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.ball_conf,
                device=args.device,
                verbose=False,
            )[0]
            ball_detections = parse_ball_model_detections(ball_result)
        if args.ball_color_fallback and not ball_detections:
            ball = detect_orange_ball(frame)
            if ball:
                ball_detections.append(ball)

        frame_detections = dedupe_ball_detections(segmented_people + ball_detections)
        frame_records = enrich_detections(
            frame,
            frame_index,
            frame_detections,
            homography,
            court_spec,
            None,
            [],
            0,
            last_positions,
            fps,
            False,
            args.court_margin,
            args.court_near_margin,
            args.court_far_margin,
        )
        records.extend(frame_records)

        stats["frames"] += 1
        stats["persons"] += sum(1 for rec in frame_records if rec["class_name"] == "person")
        stats["balls"] += sum(1 for rec in frame_records if rec["class_name"] == "sports ball")
        stats["projected"] += sum(1 for rec in frame_records if rec["court_x"] is not None)
        stats["sam_masks"] += sam_stats["masks"]
        stats["mask_cleanups"] += sam_stats["cleaned"]
        frame_index += 1

    cap.release()

    track_report = stabilize_team_identity(records)
    track_report = split_mixed_team_tracks(records, track_report)
    stitch_report = stitch_track_fragments(records, track_report, fps=fps)
    crossing_report = resolve_crossing_id_switches(records, fps=fps)
    stitch_report = summarize_players_from_records(records, stitch_report, crossing_report)
    mark_in_play_players(records)
    update_track_report_in_play(track_report, records)

    ball_interpolation_report = interpolate_ball_gaps(records, fps)
    ball_ownership_report = assign_ball_ownership(records, fps)
    pass_events = detect_passes(records, fps)
    pick_and_roll_events = detect_pick_and_rolls(records, fps)
    events = sorted(pass_events + pick_and_roll_events, key=lambda event: int(event.get("start_frame", 0)))
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
    write_json(output_dir / "track_summary.json", {"video": args.video, "fps": fps, **track_report})
    write_json(output_dir / "player_summary.json", {"video": args.video, "fps": fps, **stitch_report})
    write_json(
        output_dir / "ball_tracks.json",
        {
            "video": args.video,
            "fps": fps,
            "summary": {**ball_interpolation_report, **ball_ownership_report, **dense_ball_report},
            "records": [rec for rec in records if rec.get("class_name") == "sports ball"],
        },
    )
    summary = {
        "video": args.video,
        "fps": fps,
        "frames_processed": stats["frames"],
        "person_detections": stats["persons"],
        "ball_detections": stats["balls"],
        "sam_masks": stats["sam_masks"],
        "mask_cleanups": stats["mask_cleanups"],
        "projected_detections": stats["projected"],
        "stitched_players": stitch_report["player_count"],
        "id_switch_corrections": crossing_report["correction_count"],
        "interpolated_ball_frames": ball_interpolation_report["interpolated_ball_frames"],
        "dense_ball_estimated_frames": dense_ball_report["dense_ball_estimated_frames"],
        "ball_owned_frames": ball_ownership_report["owned_ball_frames"],
        "pass_candidates": len(pass_events),
        "pick_and_roll_candidates": len(pick_and_roll_events),
        "tracking_mode": "detector_sam2_mask_prompt",
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def segment_people_with_sam(
    sam: SAM,
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    device: str,
    imgsz: int,
    max_center_distance_px: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not detections:
        return [], {"masks": 0, "cleaned": 0}

    boxes = [det["xyxy"].astype(float).tolist() for det in detections]
    masks = predict_sam_masks(sam, frame, boxes, device=device, imgsz=imgsz)
    output: list[dict[str, Any]] = []
    cleaned_count = 0
    for det, mask in zip(detections, masks):
        next_det = dict(det)
        next_det["source"] = "detector_sam2"
        if mask is not None:
            cleaned = cleanup_disconnected_mask(mask, max_center_distance_px=max_center_distance_px)
            cleaned_box = mask_to_xyxy(cleaned)
            if cleaned_box is not None:
                next_det["xyxy"] = np.asarray(cleaned_box, dtype=float)
                next_det["mask_area"] = int(cleaned.sum())
                cleaned_count += 1
        output.append(next_det)
    return output, {"masks": sum(1 for mask in masks if mask is not None), "cleaned": cleaned_count}


def predict_sam_masks(
    sam: SAM,
    frame: np.ndarray,
    boxes: list[list[float]],
    device: str,
    imgsz: int,
) -> list[np.ndarray | None]:
    try:
        result = sam.predict(frame, bboxes=boxes, device=device, imgsz=imgsz, verbose=False)[0]
    except Exception:
        return [None for _ in boxes]
    if result.masks is None or result.masks.data is None:
        return [None for _ in boxes]

    data = result.masks.data.detach().cpu().numpy()
    if len(data) == len(boxes):
        return [(mask > 0.5).astype(np.uint8) for mask in data]

    masks = [(mask > 0.5).astype(np.uint8) for mask in data]
    assigned: list[np.ndarray | None] = []
    used: set[int] = set()
    for box in boxes:
        best_index = None
        best_iou = 0.0
        for index, mask in enumerate(masks):
            if index in used:
                continue
            mask_box = mask_to_xyxy(mask)
            if mask_box is None:
                continue
            score = box_iou(np.asarray(box, dtype=float), np.asarray(mask_box, dtype=float))
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index is None:
            assigned.append(None)
        else:
            used.add(best_index)
            assigned.append(masks[best_index])
    return assigned


class SimpleBoxTracker:
    def __init__(self, max_gap: int = 24, min_iou: float = 0.1) -> None:
        self.max_gap = max_gap
        self.min_iou = min_iou
        self.next_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}

    def update(self, detections: list[dict[str, Any]], frame_index: int) -> None:
        active = {
            track_id: track
            for track_id, track in self.tracks.items()
            if frame_index - int(track["last_frame"]) <= self.max_gap
        }
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for det_index, det in enumerate(detections):
            best_track_id = None
            best_score = 0.0
            for track_id, track in active.items():
                if track_id in used_tracks:
                    continue
                iou = box_iou(det["xyxy"], track["bbox"])
                distance_score = center_distance_score(det["xyxy"], track["bbox"])
                score = max(iou, distance_score)
                if iou < self.min_iou and distance_score < 0.35:
                    continue
                if score > best_score:
                    best_score = score
                    best_track_id = track_id
            if best_track_id is None:
                best_track_id = self.next_id
                self.next_id += 1
            assignments[det_index] = best_track_id
            used_tracks.add(best_track_id)

        for det_index, track_id in assignments.items():
            detections[det_index]["track_id"] = track_id
            self.tracks[track_id] = {"bbox": detections[det_index]["xyxy"].copy(), "last_frame": frame_index}


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance_score(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ac = np.asarray([(ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0], dtype=np.float32)
    bc = np.asarray([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0], dtype=np.float32)
    scale = max(max(ax2 - ax1, ay2 - ay1), max(bx2 - bx1, by2 - by1), 1.0)
    distance = float(np.linalg.norm(ac - bc))
    return max(0.0, 1.0 - distance / scale)


if __name__ == "__main__":
    main()
