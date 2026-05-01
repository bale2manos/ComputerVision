from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from basketball_cv.events import _distance, _eligible_players, _player_key, _record_identity
from basketball_cv.possession import (
    EnhancedBallOwnershipConfig,
    assign_enhanced_ball_ownership,
    ball_player_image_contact,
    best_non_dense_ball,
    build_possession_by_frame,
    enhanced_ownership_score,
    ownership_confidence,
)


@dataclass
class BalancedBallOwnershipConfig(EnhancedBallOwnershipConfig):
    """Extra post-processing for realistic basketball possession.

    This layer fixes two common failure modes in wide-angle basketball footage:
    - a previous owner is held too aggressively even when another player has both
      hands clearly on the ball;
    - one-frame / two-frame ownership flips during dribbles or body contact.
    """

    hand_contact_override_threshold: float = 0.93
    hand_contact_override_confidence: float = 0.74
    hand_contact_override_margin: float = 0.16
    short_flip_max_frames: int = 3
    bridge_loose_max_frames: int = 4


def assign_balanced_ball_ownership(
    records: list[dict[str, Any]],
    fps: float,
    config: BalancedBallOwnershipConfig | None = None,
) -> dict[str, Any]:
    config = config or BalancedBallOwnershipConfig()
    report = assign_enhanced_ball_ownership(records, fps, config)

    by_frame = group_by_frame(records)
    hand_contact_overrides = apply_hand_contact_overrides(by_frame, config)
    short_flip_corrections = smooth_short_owner_flips(records, fps, config)
    loose_bridge_corrections = bridge_short_loose_gaps(records, fps, config)

    report["hand_contact_overrides"] = hand_contact_overrides
    report["short_flip_corrections"] = short_flip_corrections
    report["loose_bridge_corrections"] = loose_bridge_corrections
    report.setdefault("parameters", {})["hand_contact_override_threshold"] = config.hand_contact_override_threshold
    report.setdefault("parameters", {})["short_flip_max_frames"] = config.short_flip_max_frames
    report.setdefault("parameters", {})["bridge_loose_max_frames"] = config.bridge_loose_max_frames
    return report


def group_by_frame(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_frame[int(rec.get("frame_index", 0))].append(rec)
    return by_frame


def apply_hand_contact_overrides(
    by_frame: dict[int, list[dict[str, Any]]],
    config: BalancedBallOwnershipConfig,
) -> int:
    corrections = 0
    for frame_records in by_frame.values():
        ball = best_non_dense_ball(frame_records)
        if ball is None:
            continue
        current_owner = find_player_by_owner(frame_records, ball.get("ball_owner_player_id"))
        winner = strongest_hand_contact_candidate(ball, frame_records, config)
        if winner is None:
            continue
        if current_owner is not None and _player_key(current_owner) == _player_key(winner):
            continue
        if should_override_with_hand_contact(ball, winner, current_owner, config):
            reassign_frame_owner(frame_records, ball, winner, config, reason="hand_contact_override")
            corrections += 1
    return corrections


def strongest_hand_contact_candidate(
    ball: dict[str, Any],
    frame_records: list[dict[str, Any]],
    config: BalancedBallOwnershipConfig,
) -> dict[str, Any] | None:
    players = _eligible_players(frame_records)
    strong: list[tuple[float, float, float, dict[str, Any]]] = []
    for player in players:
        contact = ball_player_image_contact(ball, player)
        if contact < config.hand_contact_override_threshold:
            continue
        score = enhanced_ownership_score(ball, player, ball.get("ball_owner_player_id"), ball.get("ball_owner_team"), config)
        confidence = ownership_confidence(_distance(ball, player), score, config, ball, contact)
        if confidence < config.hand_contact_override_confidence:
            continue
        strong.append((contact, confidence, -score, player))
    if not strong:
        return None
    strong.sort(key=lambda item: item[:3], reverse=True)
    return strong[0][3]


def should_override_with_hand_contact(
    ball: dict[str, Any],
    winner: dict[str, Any],
    current_owner: dict[str, Any] | None,
    config: BalancedBallOwnershipConfig,
) -> bool:
    winner_contact = ball_player_image_contact(ball, winner)
    winner_score = enhanced_ownership_score(ball, winner, ball.get("ball_owner_player_id"), ball.get("ball_owner_team"), config)
    winner_conf = ownership_confidence(_distance(ball, winner), winner_score, config, ball, winner_contact)
    if current_owner is None:
        return True
    current_contact = ball_player_image_contact(ball, current_owner)
    current_score = enhanced_ownership_score(ball, current_owner, ball.get("ball_owner_player_id"), ball.get("ball_owner_team"), config)
    current_conf = float(current_owner.get("ball_owner_confidence") or 0.0)

    if winner_contact >= 0.98 and winner_conf >= current_conf + 0.08:
        return True
    if winner_contact >= current_contact + config.hand_contact_override_margin and winner_score <= current_score - 0.35:
        return True
    if winner_contact >= 0.95 and _distance(ball, winner) <= max(2.4, _distance(ball, current_owner) * 0.65):
        return True
    return False


def smooth_short_owner_flips(records: list[dict[str, Any]], fps: float, config: BalancedBallOwnershipConfig) -> int:
    by_frame = group_by_frame(records)
    timeline = owner_segments(by_frame)
    corrections = 0
    for prev_seg, seg, next_seg in zip(timeline[:-2], timeline[1:-1], timeline[2:]):
        if seg["state"] != "owned":
            continue
        if seg["duration"] > config.short_flip_max_frames:
            continue
        if prev_seg["state"] != "owned" or next_seg["state"] != "owned":
            continue
        if prev_seg["owner"] != next_seg["owner"] or prev_seg["owner"] == seg["owner"]:
            continue
        stable_owner = prev_seg["owner"]
        for frame in range(seg["start"], seg["end"] + 1):
            frame_records = by_frame.get(frame, [])
            ball = best_non_dense_ball(frame_records)
            owner = find_player_by_owner(frame_records, stable_owner)
            if ball is None or owner is None:
                continue
            if owner_plausible_for_bridge(ball, owner, config):
                reassign_frame_owner(frame_records, ball, owner, config, reason="short_flip_smoothing")
                corrections += 1
    return corrections


def bridge_short_loose_gaps(records: list[dict[str, Any]], fps: float, config: BalancedBallOwnershipConfig) -> int:
    by_frame = group_by_frame(records)
    timeline = owner_segments(by_frame)
    corrections = 0
    for prev_seg, seg, next_seg in zip(timeline[:-2], timeline[1:-1], timeline[2:]):
        if seg["state"] == "owned":
            continue
        if seg["duration"] > config.bridge_loose_max_frames:
            continue
        if prev_seg["state"] != "owned" or next_seg["state"] != "owned":
            continue
        if prev_seg["owner"] != next_seg["owner"]:
            continue
        stable_owner = prev_seg["owner"]
        for frame in range(seg["start"], seg["end"] + 1):
            frame_records = by_frame.get(frame, [])
            ball = best_non_dense_ball(frame_records)
            owner = find_player_by_owner(frame_records, stable_owner)
            if ball is None or owner is None:
                continue
            if owner_plausible_for_bridge(ball, owner, config):
                reassign_frame_owner(frame_records, ball, owner, config, reason="short_loose_bridge")
                corrections += 1
    return corrections


def owner_segments(by_frame: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    possession_by_frame = build_possession_by_frame([rec for records in by_frame.values() for rec in records])
    for frame in sorted(by_frame):
        possession = possession_by_frame.get(frame, {"state": "undetected"})
        state = possession.get("state") or "undetected"
        owner = possession.get("player_id") if state == "owned" else None
        team = possession.get("team") if state == "owned" else None
        key = (state, owner, team)
        if current is None or current["key"] != key or frame != current["end"] + 1:
            if current is not None:
                current["duration"] = current["end"] - current["start"] + 1
                segments.append(current)
            current = {"key": key, "state": state, "owner": owner, "team": team, "start": frame, "end": frame}
        else:
            current["end"] = frame
    if current is not None:
        current["duration"] = current["end"] - current["start"] + 1
        segments.append(current)
    return segments


def owner_plausible_for_bridge(ball: dict[str, Any], owner: dict[str, Any], config: BalancedBallOwnershipConfig) -> bool:
    contact = ball_player_image_contact(ball, owner)
    if contact >= 0.14:
        return True
    if _distance(ball, owner) <= max(2.6, config.keep_owner_radius_m):
        return True
    return False


def find_player_by_owner(frame_records: list[dict[str, Any]], owner_id: Any) -> dict[str, Any] | None:
    if owner_id is None:
        return None
    try:
        owner_int = int(owner_id)
    except (TypeError, ValueError):
        return None
    for rec in frame_records:
        if rec.get("class_name") == "person" and _player_key(rec) == owner_int:
            return rec
    return None


def reassign_frame_owner(
    frame_records: list[dict[str, Any]],
    ball: dict[str, Any],
    owner: dict[str, Any],
    config: BalancedBallOwnershipConfig,
    reason: str,
) -> None:
    for rec in frame_records:
        if rec.get("class_name") == "person":
            clear_player_owner_fields(rec)

    owner_id = _player_key(owner)
    if owner_id is None:
        return

    contact = ball_player_image_contact(ball, owner)
    score = enhanced_ownership_score(ball, owner, ball.get("ball_owner_player_id"), ball.get("ball_owner_team"), config)
    distance = _distance(ball, owner)
    confidence = ownership_confidence(distance, score, config, ball, contact)
    identity = _record_identity(owner)

    ball["ball_state"] = "owned"
    ball["ball_owner_player_id"] = owner_id
    ball["ball_owner_identity"] = identity
    ball["ball_owner_jersey_number"] = owner.get("jersey_number")
    ball["ball_owner_team"] = owner.get("team")
    ball["ball_owner_distance_m"] = round(float(distance), 3)
    ball["ball_owner_image_contact"] = round(float(contact), 3)
    ball["ball_owner_score"] = round(float(score), 3)
    ball["ball_owner_confidence"] = round(float(confidence), 3)
    ball["ball_owner_assignment_reason"] = reason
    if owner.get("court_x") is not None and owner.get("court_y") is not None:
        ball["possession_court_x"] = owner.get("court_x")
        ball["possession_court_y"] = owner.get("court_y")

    owner["has_ball"] = True
    owner["ball_state"] = "owned"
    owner["ball_owner_player_id"] = owner_id
    owner["ball_owner_identity"] = identity
    owner["ball_owner_jersey_number"] = owner.get("jersey_number")
    owner["ball_owner_team"] = owner.get("team")
    owner["ball_owner_distance_m"] = round(float(distance), 3)
    owner["ball_owner_image_contact"] = round(float(contact), 3)
    owner["ball_owner_confidence"] = round(float(confidence), 3)
    owner["ball_owner_assignment_reason"] = reason
    owner["ball_owner_source"] = ball.get("source")


def clear_player_owner_fields(rec: dict[str, Any]) -> None:
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
        "ball_owner_image_contact",
        "ball_owner_assignment_reason",
    ):
        rec.pop(key, None)
