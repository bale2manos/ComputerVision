from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from basketball_cv.events import (
    BallOwnershipConfig,
    _ball_player_image_distance,
    _ball_player_overlap_fraction,
    _distance,
    _eligible_players,
    _ownership_score,
    _player_key,
    _record_identity,
)


@dataclass
class EnhancedBallOwnershipConfig(BallOwnershipConfig):
    """Temporal possession configuration for render/debug workflows."""

    short_occlusion_hold_s: float = 0.35
    owner_switch_margin: float = 0.18
    debug_candidate_count: int = 3


def assign_enhanced_ball_ownership(
    records: list[dict[str, Any]],
    fps: float,
    config: EnhancedBallOwnershipConfig | None = None,
) -> dict[str, Any]:
    """Assign possession with temporal smoothing and explicit ball states.

    States:
    - owned: a player is the current owner.
    - undetected_hold: ball is briefly missing, previous owner is held.
    - loose: ball exists, but no confident owner.
    - flight: ball exists and appears to be travelling after a previous owner.
    - undetected: no ball and no short hold.
    """

    config = config or EnhancedBallOwnershipConfig()
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_frame[int(rec["frame_index"])].append(rec)

    previous_owner: int | None = None
    previous_team: str | None = None
    previous_ball: dict[str, Any] | None = None
    missing_ball_frames = 0
    max_missing_hold_frames = max(1, int(round(config.short_occlusion_hold_s * max(fps, 1.0))))

    owner_counts: dict[int, int] = defaultdict(int)
    ball_frames = 0
    owned_ball_frames = 0
    loose_ball_frames = 0
    flight_ball_frames = 0
    held_without_ball_frames = 0

    for frame in sorted(by_frame):
        frame_records = by_frame[frame]
        for rec in frame_records:
            clear_possession_fields(rec)

        ball = best_non_dense_ball(frame_records)
        players = _eligible_players(frame_records)

        if ball is None:
            missing_ball_frames += 1
            held_owner = hold_previous_owner_during_occlusion(
                frame_records,
                previous_owner,
                previous_team,
                missing_ball_frames,
                max_missing_hold_frames,
            )
            if held_owner is None:
                previous_owner = None
                previous_team = None
            else:
                held_without_ball_frames += 1
                owner_id = _player_key(held_owner)
                if owner_id is not None:
                    owner_counts[owner_id] += 1
            continue

        missing_ball_frames = 0
        ball_frames += 1
        ball["ball_state"] = "unassigned"
        ball["ball_source_reliability"] = round(float(ball_source_reliability(ball)), 3)

        candidates = rank_owner_candidates(ball, players, previous_owner, previous_team, config)
        if candidates:
            ball["owner_candidates"] = candidates[: max(0, int(config.debug_candidate_count))]

        owner = choose_temporal_owner(ball, players, previous_owner, previous_team, config)
        if owner is None:
            state = unowned_ball_state(ball, previous_ball, previous_owner, fps, config)
            ball["ball_state"] = state
            if state == "flight":
                flight_ball_frames += 1
            else:
                loose_ball_frames += 1
            previous_ball = ball
            continue

        owner_id = _player_key(owner)
        if owner_id is None:
            ball["ball_state"] = "loose"
            loose_ball_frames += 1
            previous_owner = None
            previous_team = None
            previous_ball = ball
            continue

        attach_owner_fields(ball, owner, owner_id, previous_owner, previous_team, config)
        owner_counts[owner_id] += 1
        owned_ball_frames += 1
        previous_owner = owner_id
        previous_team = owner.get("team")
        previous_ball = ball

    return {
        "ball_frames": ball_frames,
        "owned_ball_frames": owned_ball_frames,
        "loose_ball_frames": loose_ball_frames,
        "flight_ball_frames": flight_ball_frames,
        "held_without_ball_frames": held_without_ball_frames,
        "unique_owners": len(owner_counts),
        "owner_frame_counts": {str(k): int(v) for k, v in sorted(owner_counts.items())},
        "parameters": {
            "possession_radius_m": config.possession_radius_m,
            "keep_owner_radius_m": config.keep_owner_radius_m,
            "short_occlusion_hold_s": config.short_occlusion_hold_s,
            "owner_switch_margin": config.owner_switch_margin,
            "min_owner_confidence": config.min_owner_confidence,
        },
    }


def clear_possession_fields(rec: dict[str, Any]) -> None:
    for key in (
        "has_ball",
        "ball_state",
        "ball_owner_player_id",
        "ball_owner_identity",
        "ball_owner_jersey_number",
        "ball_owner_team",
        "ball_owner_distance_m",
        "ball_owner_confidence",
        "ball_owner_source",
        "ball_owner_image_overlap",
        "ball_owner_score",
        "ball_source_reliability",
        "owner_candidates",
        "ball_missing_frames",
        "possession_hold_reason",
    ):
        rec.pop(key, None)


def best_non_dense_ball(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    balls = [
        rec
        for rec in records
        if rec.get("class_name") == "sports ball"
        and rec.get("court_x") is not None
        and rec.get("court_y") is not None
        and not str(rec.get("source", "")).startswith("dense_ball_")
    ]
    if not balls:
        return None
    return max(balls, key=lambda rec: float(rec.get("confidence") or 0.0) * ball_source_reliability(rec))


def choose_temporal_owner(
    ball: dict[str, Any],
    players: list[dict[str, Any]],
    previous_owner: int | None,
    previous_team: str | None,
    config: EnhancedBallOwnershipConfig,
) -> dict[str, Any] | None:
    if not players:
        return None

    ranked = sorted(players, key=lambda player: _ownership_score(ball, player, previous_owner, previous_team, config))
    best = ranked[0]
    best_owner_id = _player_key(best)
    best_distance = _distance(ball, best)

    if previous_owner is not None and best_owner_id != previous_owner:
        previous_candidates = [player for player in players if _player_key(player) == previous_owner]
        if previous_candidates:
            previous = previous_candidates[0]
            previous_distance = _distance(ball, previous)
            previous_score = _ownership_score(ball, previous, previous_owner, previous_team, config)
            best_score = _ownership_score(ball, best, previous_owner, previous_team, config)
            if previous_distance <= config.keep_owner_radius_m and best_score > previous_score - config.owner_switch_margin:
                best = previous
                best_distance = previous_distance
                best_owner_id = previous_owner

    radius = config.keep_owner_radius_m if previous_owner is not None and best_owner_id == previous_owner else config.possession_radius_m
    overlap = _ball_player_overlap_fraction(ball, best)
    if best_distance > radius and overlap <= 0.08:
        return None

    score = _ownership_score(ball, best, previous_owner, previous_team, config)
    confidence = ownership_confidence(best_distance, score, config, ball)
    if confidence < config.min_owner_confidence:
        return None
    return best


def attach_owner_fields(
    ball: dict[str, Any],
    owner: dict[str, Any],
    owner_id: int,
    previous_owner: int | None,
    previous_team: str | None,
    config: EnhancedBallOwnershipConfig,
) -> None:
    distance_m = _distance(ball, owner)
    score = _ownership_score(ball, owner, previous_owner, previous_team, config)
    confidence = ownership_confidence(distance_m, score, config, ball)
    identity = _record_identity(owner)

    ball["ball_state"] = "owned"
    ball["ball_owner_player_id"] = owner_id
    ball["ball_owner_identity"] = identity
    ball["ball_owner_jersey_number"] = owner.get("jersey_number")
    ball["ball_owner_team"] = owner.get("team")
    ball["ball_owner_distance_m"] = round(float(distance_m), 3)
    ball["ball_owner_image_overlap"] = round(float(_ball_player_overlap_fraction(ball, owner)), 3)
    ball["ball_owner_score"] = round(float(score), 3)
    ball["ball_owner_confidence"] = round(float(confidence), 3)

    owner["has_ball"] = True
    owner["ball_state"] = "owned"
    owner["ball_owner_player_id"] = owner_id
    owner["ball_owner_identity"] = identity
    owner["ball_owner_jersey_number"] = owner.get("jersey_number")
    owner["ball_owner_team"] = owner.get("team")
    owner["ball_owner_distance_m"] = round(float(distance_m), 3)
    owner["ball_owner_confidence"] = round(float(confidence), 3)
    owner["ball_owner_source"] = ball.get("source")


def rank_owner_candidates(
    ball: dict[str, Any],
    players: list[dict[str, Any]],
    previous_owner: int | None,
    previous_team: str | None,
    config: EnhancedBallOwnershipConfig,
) -> list[dict[str, Any]]:
    ranked = sorted(players, key=lambda player: _ownership_score(ball, player, previous_owner, previous_team, config))
    output = []
    for player in ranked:
        distance = _distance(ball, player)
        score = _ownership_score(ball, player, previous_owner, previous_team, config)
        output.append(
            {
                "player_id": _player_key(player),
                "identity": _record_identity(player),
                "jersey_number": player.get("jersey_number"),
                "team": player.get("team"),
                "distance_m": round(float(distance), 3),
                "image_distance": round(float(_ball_player_image_distance(ball, player)), 3),
                "overlap": round(float(_ball_player_overlap_fraction(ball, player)), 3),
                "score": round(float(score), 3),
                "confidence": round(float(ownership_confidence(distance, score, config, ball)), 3),
                "was_previous_owner": bool(previous_owner is not None and _player_key(player) == previous_owner),
            }
        )
    return output


def hold_previous_owner_during_occlusion(
    frame_records: list[dict[str, Any]],
    previous_owner: int | None,
    previous_team: str | None,
    missing_ball_frames: int,
    max_missing_hold_frames: int,
) -> dict[str, Any] | None:
    if previous_owner is None or missing_ball_frames > max_missing_hold_frames:
        return None
    owners = [rec for rec in _eligible_players(frame_records) if _player_key(rec) == previous_owner]
    if not owners:
        return None
    owner = owners[0]
    decay = max(0.15, 1.0 - missing_ball_frames / max(max_missing_hold_frames + 1, 1))
    owner["has_ball"] = True
    owner["ball_state"] = "undetected_hold"
    owner["ball_owner_player_id"] = previous_owner
    owner["ball_owner_identity"] = _record_identity(owner)
    owner["ball_owner_jersey_number"] = owner.get("jersey_number")
    owner["ball_owner_team"] = owner.get("team") or previous_team
    owner["ball_owner_confidence"] = round(float(0.45 * decay), 3)
    owner["ball_missing_frames"] = int(missing_ball_frames)
    owner["possession_hold_reason"] = "short_ball_occlusion"
    return owner


def unowned_ball_state(
    ball: dict[str, Any],
    previous_ball: dict[str, Any] | None,
    previous_owner: int | None,
    fps: float,
    config: EnhancedBallOwnershipConfig,
) -> str:
    if previous_owner is None or previous_ball is None:
        return "loose"
    frame_delta = int(ball.get("frame_index", 0)) - int(previous_ball.get("frame_index", 0))
    dt = max(frame_delta / max(fps, 1e-6), 1e-6)
    speed = _distance(ball, previous_ball) / dt
    if _distance(ball, previous_ball) >= config.flight_min_distance_m or speed >= 7.0:
        return "flight"
    return "loose"


def ownership_confidence(
    distance_m: float,
    score: float,
    config: EnhancedBallOwnershipConfig,
    ball: dict[str, Any] | None = None,
) -> float:
    distance_conf = 1.0 - distance_m / max(config.keep_owner_radius_m, 1e-6)
    score_conf = 1.0 - max(score, 0.0) / 2.0
    return max(0.0, min(1.0, (0.55 * distance_conf + 0.45 * score_conf) * ball_source_reliability(ball)))


def ball_source_reliability(ball: dict[str, Any] | None) -> float:
    if ball is None:
        return 1.0
    source = str(ball.get("source", ""))
    if source == "ball_model":
        return 1.0
    if source == "yolo":
        return 0.86
    if source == "orange_blob":
        return 0.72
    if source == "interpolated_ball":
        return 0.58
    if source.startswith("dense_ball_"):
        return 0.0
    return 0.75


def build_possession_by_frame(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_frame[int(rec.get("frame_index", 0))].append(rec)
    return {frame: best_possession_for_frame(frame_records) for frame, frame_records in by_frame.items()}


def best_possession_for_frame(records: list[dict[str, Any]]) -> dict[str, Any]:
    owners = [rec for rec in records if rec.get("class_name") == "person" and rec.get("has_ball")]
    ball = best_non_dense_ball(records)
    if owners:
        owner = max(owners, key=lambda rec: float(rec.get("ball_owner_confidence") or 0.0))
        return {
            "state": owner.get("ball_state") or "owned",
            "owner": owner,
            "ball": ball,
            "player_id": owner.get("ball_owner_player_id", owner.get("player_id")),
            "identity": owner.get("ball_owner_identity") or display_record_identity(owner),
            "jersey_number": owner.get("ball_owner_jersey_number") or owner.get("jersey_number"),
            "team": owner.get("ball_owner_team") or owner.get("team"),
            "confidence": owner.get("ball_owner_confidence"),
            "distance_m": owner.get("ball_owner_distance_m"),
            "source": owner.get("ball_owner_source") or (ball.get("source") if ball else None),
            "missing_frames": owner.get("ball_missing_frames"),
            "candidates": ball.get("owner_candidates", []) if ball else [],
        }
    if ball is not None:
        return {
            "state": ball.get("ball_state") or "loose",
            "owner": None,
            "ball": ball,
            "source": ball.get("source"),
            "candidates": ball.get("owner_candidates", []),
        }
    return {"state": "undetected", "owner": None, "ball": None, "candidates": []}


def build_possession_timeline(records: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    by_frame = build_possession_by_frame(records)
    timeline: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for frame in sorted(by_frame):
        possession = by_frame[frame]
        key = (possession.get("state"), possession.get("player_id"), possession.get("identity"), possession.get("team"))
        if current is None or current["_key"] != key or frame != int(current["end_frame"]) + 1:
            if current is not None:
                finalize_timeline_segment(current, fps)
                timeline.append(current)
            current = {
                "_key": key,
                "state": possession.get("state"),
                "start_frame": frame,
                "end_frame": frame,
                "player_id": possession.get("player_id"),
                "identity": possession.get("identity"),
                "jersey_number": possession.get("jersey_number"),
                "team": possession.get("team"),
                "sources": [],
                "confidences": [],
                "distances_m": [],
            }
        else:
            current["end_frame"] = frame

        if possession.get("source"):
            current["sources"].append(possession.get("source"))
        if possession.get("confidence") is not None:
            current["confidences"].append(float(possession["confidence"]))
        if possession.get("distance_m") is not None:
            current["distances_m"].append(float(possession["distance_m"]))

    if current is not None:
        finalize_timeline_segment(current, fps)
        timeline.append(current)
    for segment in timeline:
        segment.pop("_key", None)
    return timeline


def finalize_timeline_segment(segment: dict[str, Any], fps: float) -> None:
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    segment["start_time_s"] = round(start / max(fps, 1e-6), 3)
    segment["end_time_s"] = round(end / max(fps, 1e-6), 3)
    segment["duration_s"] = round((end - start + 1) / max(fps, 1e-6), 3)
    confidences = segment.pop("confidences", [])
    distances = segment.pop("distances_m", [])
    sources = segment.get("sources", [])
    if confidences:
        segment["mean_confidence"] = round(float(np.mean(confidences)), 3)
        segment["max_confidence"] = round(float(np.max(confidences)), 3)
    if distances:
        segment["mean_distance_m"] = round(float(np.mean(distances)), 3)
    if sources:
        counts: dict[str, int] = {}
        for source in sources:
            counts[str(source)] = counts.get(str(source), 0) + 1
        segment["source_counts"] = counts
    else:
        segment.pop("sources", None)


def display_record_identity(rec: dict[str, Any] | None) -> str | None:
    if rec is None:
        return None
    if rec.get("jersey_identity"):
        identity = str(rec["jersey_identity"])
        return identity.split("_", 1)[1] if "_" in identity else identity
    if rec.get("jersey_number"):
        return f"#{rec['jersey_number']}"
    if rec.get("player_id") is not None:
        return f"P{rec['player_id']}"
    if rec.get("track_id") is not None:
        return f"T{rec['track_id']}"
    return None
