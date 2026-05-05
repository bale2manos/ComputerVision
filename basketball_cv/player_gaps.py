from __future__ import annotations

from collections import defaultdict
from typing import Any


def interpolate_player_gaps(
    records: list[dict[str, Any]],
    fps: float,
    max_gap_frames: int = 8,
    max_speed_mps: float = 8.5,
) -> dict[str, int]:
    synthetic: list[dict[str, Any]] = []
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("player_id") is None or rec.get("is_estimated"):
            continue
        if rec.get("court_x") is None or rec.get("court_y") is None:
            continue
        if rec.get("bbox") is None or rec.get("anchor_px") is None:
            continue
        by_player[int(rec["player_id"])].append(rec)
        by_frame[int(rec["frame_index"])].append(rec)

    for player_records in by_player.values():
        player_records.sort(key=lambda rec: int(rec["frame_index"]))
        for prev, nxt in zip(player_records[:-1], player_records[1:]):
            gap = int(nxt["frame_index"]) - int(prev["frame_index"]) - 1
            if gap <= 0 or gap > max_gap_frames:
                continue
            if prev.get("team") != nxt.get("team"):
                continue
            if not _bbox_is_compatible(prev["bbox"], nxt["bbox"]):
                continue
            dx = float(nxt["court_x"]) - float(prev["court_x"])
            dy = float(nxt["court_y"]) - float(prev["court_y"])
            seconds = max((gap + 1) / max(fps, 1e-6), 1e-6)
            if ((dx * dx + dy * dy) ** 0.5) / seconds > max_speed_mps:
                continue
            total = float(int(nxt["frame_index"]) - int(prev["frame_index"]))
            candidate_records: list[dict[str, Any]] = []
            blocked = False
            for frame in range(int(prev["frame_index"]) + 1, int(nxt["frame_index"])):
                alpha = (frame - int(prev["frame_index"])) / total
                est_court_x = round(float(prev["court_x"]) * (1.0 - alpha) + float(nxt["court_x"]) * alpha, 3)
                est_court_y = round(float(prev["court_y"]) * (1.0 - alpha) + float(nxt["court_y"]) * alpha, 3)
                if _has_nearby_competing_detection(
                    by_frame.get(frame, []),
                    player_id=int(prev["player_id"]),
                    est_court_x=est_court_x,
                    est_court_y=est_court_y,
                    min_confidence=min(float(prev.get("confidence") or 0.0), float(nxt.get("confidence") or 0.0)),
                ):
                    blocked = True
                    break
                candidate_records.append(
                    {
                        **prev,
                        "frame_index": frame,
                        "time_s": round(frame / max(fps, 1e-6), 4),
                        "bbox": [round(float(a) * (1.0 - alpha) + float(b) * alpha, 2) for a, b in zip(prev["bbox"], nxt["bbox"])],
                        "anchor_px": [round(float(a) * (1.0 - alpha) + float(b) * alpha, 2) for a, b in zip(prev["anchor_px"], nxt["anchor_px"])],
                        "court_x": est_court_x,
                        "court_y": est_court_y,
                        "is_estimated": True,
                        "estimate_reason": "short_gap_linear",
                        "source": "interpolated_player_gap",
                        "confidence": round(min(float(prev.get("confidence") or 0.0), float(nxt.get("confidence") or 0.0)) * 0.55, 4),
                    }
                )
            if not blocked:
                synthetic.extend(candidate_records)

    records.extend(synthetic)
    records.sort(
        key=lambda rec: (
            int(rec.get("frame_index", 0)),
            int(rec.get("player_id") or 9999),
            bool(rec.get("is_estimated")),
            str(rec.get("source", "")),
        )
    )
    return {"short_gap_fills": len(synthetic), "estimated_player_frames": len(synthetic)}


def smooth_player_positions(records: list[dict[str, Any]], window: int = 5) -> None:
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") == "person" and rec.get("player_id") is not None and rec.get("court_x") is not None and rec.get("court_y") is not None:
            by_player[int(rec["player_id"])].append(rec)

    radius = max(0, window // 2)
    for player_records in by_player.values():
        player_records.sort(key=lambda rec: int(rec["frame_index"]))
        xs = [float(rec["court_x"]) for rec in player_records]
        ys = [float(rec["court_y"]) for rec in player_records]
        for index, rec in enumerate(player_records):
            lo = max(0, index - radius)
            hi = min(len(player_records), index + radius + 1)
            rec["court_x_smooth"] = round(sum(xs[lo:hi]) / float(hi - lo), 3)
            rec["court_y_smooth"] = round(sum(ys[lo:hi]) / float(hi - lo), 3)


def _bbox_is_compatible(prev_bbox: list[float], next_bbox: list[float], max_size_delta: float = 0.35, max_aspect_delta: float = 0.28) -> bool:
    prev_w = max(float(prev_bbox[2]) - float(prev_bbox[0]), 1.0)
    prev_h = max(float(prev_bbox[3]) - float(prev_bbox[1]), 1.0)
    next_w = max(float(next_bbox[2]) - float(next_bbox[0]), 1.0)
    next_h = max(float(next_bbox[3]) - float(next_bbox[1]), 1.0)
    prev_area = prev_w * prev_h
    next_area = next_w * next_h
    area_delta = abs(next_area - prev_area) / max(prev_area, next_area, 1.0)
    prev_aspect = prev_w / prev_h
    next_aspect = next_w / next_h
    aspect_delta = abs(next_aspect - prev_aspect) / max(prev_aspect, next_aspect, 1.0)
    return area_delta <= max_size_delta and aspect_delta <= max_aspect_delta


def _has_nearby_competing_detection(
    frame_records: list[dict[str, Any]],
    player_id: int,
    est_court_x: float,
    est_court_y: float,
    min_confidence: float,
    max_distance_m: float = 0.9,
) -> bool:
    for rec in frame_records:
        if rec.get("player_id") is None or int(rec["player_id"]) == player_id:
            continue
        if rec.get("court_x") is None or rec.get("court_y") is None:
            continue
        if float(rec.get("confidence") or 0.0) + 1e-6 < min_confidence:
            continue
        dx = float(rec["court_x"]) - est_court_x
        dy = float(rec["court_y"]) - est_court_y
        if (dx * dx + dy * dy) ** 0.5 <= max_distance_m:
            return True
    return False
