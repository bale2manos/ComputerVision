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
    """Temporal possession configuration for render/debug workflows.

    Important: court coordinates come from a homography fitted on the floor. A
    ball held above the floor can project several metres away in the minimap. For
    possession, image-space contact with the player's upper body/hands therefore
    has to override raw court distance when it is strong.
    """

    short_occlusion_hold_s: float = 0.35
    owner_switch_margin: float = 0.18
    debug_candidate_count: int = 3
    strong_image_contact: float = 0.62
    image_contact_bonus: float = 1.25
    image_contact_confidence_weight: float = 0.72
    contact_override_max_court_distance_m: float = 7.5
    opponent_switch_min_frames: int = 3
    same_team_switch_min_frames: int = 2
    decisive_switch_contact_advantage: float = 0.45
    decisive_switch_score_advantage: float = 1.25
    previous_owner_grace_radius_m: float = 8.0
    previous_owner_min_contact_to_hold: float = 0.18


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
    pending_owner: int | None = None
    pending_owner_team: str | None = None
    pending_owner_frames = 0
    missing_ball_frames = 0
    max_missing_hold_frames = max(1, int(round(config.short_occlusion_hold_s * max(fps, 1.0))))

    owner_counts: dict[int, int] = defaultdict(int)
    ball_frames = 0
    owned_ball_frames = 0
    loose_ball_frames = 0
    flight_ball_frames = 0
    held_without_ball_frames = 0
    image_contact_owned_frames = 0
    suppressed_switch_frames = 0
    decisive_switch_frames = 0

    for frame in sorted(by_frame):
        frame_records = by_frame[frame]
        for rec in frame_records:
            clear_possession_fields(rec)

        ball = best_non_dense_ball(frame_records)
        players = _eligible_players(frame_records)

        if ball is None:
            missing_ball_frames += 1
            pending_owner = None
            pending_owner_team = None
            pending_owner_frames = 0
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

        raw_owner = choose_temporal_owner(ball, players, previous_owner, previous_team, config)
        owner = raw_owner

        if owner is None and previous_owner is not None:
            previous_candidate = find_player_by_owner(players, previous_owner)
            if previous_candidate is not None and previous_owner_still_plausible(ball, previous_candidate, config):
                owner = previous_candidate
                ball["possession_hold_reason"] = "dribble_or_contested_ball_hold"
                suppressed_switch_frames += 1

        if owner is not None and previous_owner is not None:
            owner_id_tmp = _player_key(owner)
            if owner_id_tmp is not None and owner_id_tmp != previous_owner:
                previous_candidate = find_player_by_owner(players, previous_owner)
                if previous_candidate is not None and previous_owner_still_plausible(ball, previous_candidate, config):
                    owner_team_tmp = owner.get("team")
                    if pending_owner == owner_id_tmp:
                        pending_owner_frames += 1
                    else:
                        pending_owner = owner_id_tmp
                        pending_owner_team = owner_team_tmp
                        pending_owner_frames = 1

                    required_frames = (
                        config.same_team_switch_min_frames
                        if owner_team_tmp is not None and owner_team_tmp == previous_team
                        else config.opponent_switch_min_frames
                    )
                    decisive = is_decisive_takeover(ball, owner, previous_candidate, previous_owner, previous_team, config)
                    if pending_owner_frames < required_frames and not decisive:
                        ball["possession_switch_suppressed"] = {
                            "candidate_player_id": owner_id_tmp,
                            "candidate_team": owner_team_tmp,
                            "previous_player_id": previous_owner,
                            "previous_team": previous_team,
                            "pending_frames": pending_owner_frames,
                            "required_frames": required_frames,
                            "reason": "temporal_hysteresis",
                        }
                        owner = previous_candidate
                        suppressed_switch_frames += 1
                    elif decisive:
                        ball["possession_switch_decisive"] = True
                        decisive_switch_frames += 1
                        pending_owner = None
                        pending_owner_team = None
                        pending_owner_frames = 0
                else:
                    pending_owner = None
                    pending_owner_team = None
                    pending_owner_frames = 0
            else:
                pending_owner = None
                pending_owner_team = None
                pending_owner_frames = 0

        if owner is None:
            pending_owner = None
            pending_owner_team = None
            pending_owner_frames = 0
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
        if ball.get("possession_switch_suppressed"):
            ball["ball_owner_assignment_reason"] = "temporal_hysteresis"
            owner["ball_owner_assignment_reason"] = "temporal_hysteresis"
        if ball.get("possession_hold_reason"):
            ball["ball_owner_assignment_reason"] = str(ball["possession_hold_reason"])
            owner["ball_owner_assignment_reason"] = str(ball["possession_hold_reason"])
        if float(ball.get("ball_owner_image_contact") or 0.0) >= config.strong_image_contact:
            image_contact_owned_frames += 1
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
        "image_contact_owned_frames": image_contact_owned_frames,
        "suppressed_switch_frames": suppressed_switch_frames,
        "decisive_switch_frames": decisive_switch_frames,
        "unique_owners": len(owner_counts),
        "owner_frame_counts": {str(k): int(v) for k, v in sorted(owner_counts.items())},
        "parameters": {
            "possession_radius_m": config.possession_radius_m,
            "keep_owner_radius_m": config.keep_owner_radius_m,
            "short_occlusion_hold_s": config.short_occlusion_hold_s,
            "owner_switch_margin": config.owner_switch_margin,
            "strong_image_contact": config.strong_image_contact,
            "image_contact_bonus": config.image_contact_bonus,
            "contact_override_max_court_distance_m": config.contact_override_max_court_distance_m,
            "opponent_switch_min_frames": config.opponent_switch_min_frames,
            "same_team_switch_min_frames": config.same_team_switch_min_frames,
            "previous_owner_grace_radius_m": config.previous_owner_grace_radius_m,
            "previous_owner_min_contact_to_hold": config.previous_owner_min_contact_to_hold,
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
        "ball_owner_image_contact",
        "ball_owner_assignment_reason",
        "ball_owner_score",
        "ball_source_reliability",
        "owner_candidates",
        "ball_missing_frames",
        "possession_hold_reason",
        "possession_switch_suppressed",
        "possession_switch_decisive",
        "possession_court_x",
        "possession_court_y",
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

    ranked = sorted(players, key=lambda player: enhanced_ownership_score(ball, player, previous_owner, previous_team, config))
    best = ranked[0]
    best_owner_id = _player_key(best)
    best_distance = _distance(ball, best)
    best_contact = ball_player_image_contact(ball, best)

    if previous_owner is not None and best_owner_id != previous_owner:
        previous_candidates = [player for player in players if _player_key(player) == previous_owner]
        if previous_candidates:
            previous = previous_candidates[0]
            previous_distance = _distance(ball, previous)
            previous_score = enhanced_ownership_score(ball, previous, previous_owner, previous_team, config)
            best_score = enhanced_ownership_score(ball, best, previous_owner, previous_team, config)
            previous_contact = ball_player_image_contact(ball, previous)
            if previous_contact >= config.strong_image_contact and best_contact < previous_contact + 0.12:
                best = previous
                best_distance = previous_distance
                best_contact = previous_contact
                best_owner_id = previous_owner
            elif previous_distance <= config.keep_owner_radius_m and best_score > previous_score - config.owner_switch_margin:
                best = previous
                best_distance = previous_distance
                best_contact = previous_contact
                best_owner_id = previous_owner

    score = enhanced_ownership_score(ball, best, previous_owner, previous_team, config)
    confidence = ownership_confidence(best_distance, score, config, ball, best_contact)

    if best_contact >= config.strong_image_contact:
        # A ball inside/next to the upper body or hands should count as possession
        # even if its floor-plane projection is displaced by perspective.
        if best_distance <= config.contact_override_max_court_distance_m or _ball_player_overlap_fraction(ball, best) > 0.02:
            return best if confidence >= max(0.25, config.min_owner_confidence * 0.75) else None

    radius = config.keep_owner_radius_m if previous_owner is not None and best_owner_id == previous_owner else config.possession_radius_m
    overlap = _ball_player_overlap_fraction(ball, best)
    if best_distance > radius and overlap <= 0.08:
        return None
    if confidence < config.min_owner_confidence:
        return None
    return best


def find_player_by_owner(players: list[dict[str, Any]], owner_id: int | None) -> dict[str, Any] | None:
    if owner_id is None:
        return None
    for player in players:
        if _player_key(player) == owner_id:
            return player
    return None


def previous_owner_still_plausible(
    ball: dict[str, Any],
    previous_player: dict[str, Any],
    config: EnhancedBallOwnershipConfig,
) -> bool:
    contact = ball_player_image_contact(ball, previous_player)
    if contact >= config.previous_owner_min_contact_to_hold:
        return True
    if _ball_player_overlap_fraction(ball, previous_player) > 0.02:
        return True
    if _distance(ball, previous_player) <= config.previous_owner_grace_radius_m:
        return True
    return False


def is_decisive_takeover(
    ball: dict[str, Any],
    new_owner: dict[str, Any],
    previous_player: dict[str, Any],
    previous_owner: int | None,
    previous_team: str | None,
    config: EnhancedBallOwnershipConfig,
) -> bool:
    new_contact = ball_player_image_contact(ball, new_owner)
    prev_contact = ball_player_image_contact(ball, previous_player)
    new_score = enhanced_ownership_score(ball, new_owner, previous_owner, previous_team, config)
    prev_score = enhanced_ownership_score(ball, previous_player, previous_owner, previous_team, config)
    new_dist = _distance(ball, new_owner)
    prev_dist = _distance(ball, previous_player)

    # A true steal/catch should be visibly and geometrically much better, not
    # just a one-frame overlap during a dribble through bodies.
    if new_contact >= 0.96 and prev_contact <= 0.15 and new_score <= prev_score - config.decisive_switch_score_advantage:
        return True
    if new_contact >= prev_contact + config.decisive_switch_contact_advantage and new_dist <= max(1.15, prev_dist * 0.45):
        return True
    return False


def attach_owner_fields(
    ball: dict[str, Any],
    owner: dict[str, Any],
    owner_id: int,
    previous_owner: int | None,
    previous_team: str | None,
    config: EnhancedBallOwnershipConfig,
) -> None:
    distance_m = _distance(ball, owner)
    contact = ball_player_image_contact(ball, owner)
    score = enhanced_ownership_score(ball, owner, previous_owner, previous_team, config)
    confidence = ownership_confidence(distance_m, score, config, ball, contact)
    identity = _record_identity(owner)
    assignment_reason = "image_contact" if contact >= config.strong_image_contact else "court_distance"

    ball["ball_state"] = "owned"
    ball["ball_owner_player_id"] = owner_id
    ball["ball_owner_identity"] = identity
    ball["ball_owner_jersey_number"] = owner.get("jersey_number")
    ball["ball_owner_team"] = owner.get("team")
    ball["ball_owner_distance_m"] = round(float(distance_m), 3)
    ball["ball_owner_image_overlap"] = round(float(_ball_player_overlap_fraction(ball, owner)), 3)
    ball["ball_owner_image_contact"] = round(float(contact), 3)
    ball["ball_owner_score"] = round(float(score), 3)
    ball["ball_owner_confidence"] = round(float(confidence), 3)
    ball["ball_owner_assignment_reason"] = assignment_reason
    if owner.get("court_x") is not None and owner.get("court_y") is not None:
        # Minimap/render coordinate for a possessed elevated ball. The raw ball
        # court_x/court_y is still kept for debugging, but possession_court_* is
        # what should be shown tactically.
        ball["possession_court_x"] = owner.get("court_x")
        ball["possession_court_y"] = owner.get("court_y")

    owner["has_ball"] = True
    owner["ball_state"] = "owned"
    owner["ball_owner_player_id"] = owner_id
    owner["ball_owner_identity"] = identity
    owner["ball_owner_jersey_number"] = owner.get("jersey_number")
    owner["ball_owner_team"] = owner.get("team")
    owner["ball_owner_distance_m"] = round(float(distance_m), 3)
    owner["ball_owner_image_contact"] = round(float(contact), 3)
    owner["ball_owner_confidence"] = round(float(confidence), 3)
    owner["ball_owner_assignment_reason"] = assignment_reason
    owner["ball_owner_source"] = ball.get("source")


def enhanced_ownership_score(
    ball: dict[str, Any],
    player: dict[str, Any],
    previous_owner: int | None,
    previous_team: str | None,
    config: EnhancedBallOwnershipConfig,
) -> float:
    base = _ownership_score(ball, player, previous_owner, previous_team, config)
    contact = ball_player_image_contact(ball, player)
    overlap = _ball_player_overlap_fraction(ball, player)
    score = base - config.image_contact_bonus * contact - 0.45 * min(1.0, overlap * 4.0)
    if previous_owner is not None and _player_key(player) == previous_owner and contact > 0.35:
        score -= 0.22
    return float(score)


def ball_player_image_contact(ball: dict[str, Any], player: dict[str, Any]) -> float:
    """Estimate if the detected ball is visually in a player's hands/upper body.

    This intentionally works in image coordinates, not court coordinates, because
    homography projection is invalid for an elevated ball.
    """

    ball_box = ball.get("bbox")
    player_box = player.get("bbox")
    if not isinstance(ball_box, list) or not isinstance(player_box, list):
        return 0.0
    bx1, by1, bx2, by2 = [float(v) for v in ball_box]
    px1, py1, px2, py2 = [float(v) for v in player_box]
    pw = max(px2 - px1, 1.0)
    ph = max(py2 - py1, 1.0)
    bc_x = (bx1 + bx2) / 2.0
    bc_y = (by1 + by2) / 2.0
    ball_size = max(bx2 - bx1, by2 - by1, 1.0)

    expanded_x1 = px1 - 0.30 * pw
    expanded_x2 = px2 + 0.30 * pw
    expanded_y1 = py1 - 0.34 * ph
    expanded_y2 = py2 + 0.08 * ph
    vertical = (bc_y - py1) / ph
    horizontal_inside = expanded_x1 <= bc_x <= expanded_x2
    vertical_inside = expanded_y1 <= bc_y <= expanded_y2

    contact = 0.0
    if horizontal_inside and vertical_inside:
        if -0.25 <= vertical <= 0.68:
            contact = 0.88
        elif -0.40 <= vertical <= 0.85:
            contact = 0.62
        else:
            contact = 0.38

    # Ball center in original player box is a very strong signal, even if it is
    # above the torso/head due to a shot/pass gather.
    if px1 <= bc_x <= px2 and py1 - 0.18 * ph <= bc_y <= py1 + 0.72 * ph:
        contact = max(contact, 0.96)

    # Near-hands/arms: horizontally close and vertically in the upper 75%.
    nearest_x = min(max(bc_x, px1), px2)
    nearest_y = min(max(bc_y, py1 - 0.18 * ph), py2)
    normalized_dist = float(np.hypot(bc_x - nearest_x, bc_y - nearest_y)) / max(0.35 * ph, 0.65 * pw, ball_size, 1.0)
    if normalized_dist < 1.0 and vertical <= 0.82:
        contact = max(contact, 1.0 - normalized_dist * 0.42)

    overlap = _ball_player_overlap_fraction(ball, player)
    if overlap > 0:
        contact = max(contact, min(1.0, 0.48 + overlap * 4.0))

    # Ball below the waist is less reliable for possession because it can be on
    # the floor or visually overlapping a leg.
    if vertical > 0.72:
        contact *= 0.65
    return float(max(0.0, min(1.0, contact)))


def rank_owner_candidates(
    ball: dict[str, Any],
    players: list[dict[str, Any]],
    previous_owner: int | None,
    previous_team: str | None,
    config: EnhancedBallOwnershipConfig,
) -> list[dict[str, Any]]:
    ranked = sorted(players, key=lambda player: enhanced_ownership_score(ball, player, previous_owner, previous_team, config))
    output = []
    for player in ranked:
        distance = _distance(ball, player)
        contact = ball_player_image_contact(ball, player)
        score = enhanced_ownership_score(ball, player, previous_owner, previous_team, config)
        output.append(
            {
                "player_id": _player_key(player),
                "identity": _record_identity(player),
                "jersey_number": player.get("jersey_number"),
                "team": player.get("team"),
                "distance_m": round(float(distance), 3),
                "image_distance": round(float(_ball_player_image_distance(ball, player)), 3),
                "image_contact": round(float(contact), 3),
                "overlap": round(float(_ball_player_overlap_fraction(ball, player)), 3),
                "score": round(float(score), 3),
                "confidence": round(float(ownership_confidence(distance, score, config, ball, contact)), 3),
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
    image_contact: float = 0.0,
) -> float:
    distance_conf = 1.0 - distance_m / max(config.keep_owner_radius_m, 1e-6)
    score_conf = 1.0 - max(score, 0.0) / 2.0
    court_conf = max(0.0, min(1.0, 0.55 * distance_conf + 0.45 * score_conf))
    contact_conf = max(0.0, min(1.0, image_contact * config.image_contact_confidence_weight + 0.18))
    reliability = ball_source_reliability(ball)
    return max(0.0, min(1.0, max(court_conf, contact_conf) * reliability))


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
            "image_contact": owner.get("ball_owner_image_contact"),
            "assignment_reason": owner.get("ball_owner_assignment_reason"),
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
                "image_contacts": [],
                "assignment_reasons": [],
            }
        else:
            current["end_frame"] = frame

        if possession.get("source"):
            current["sources"].append(possession.get("source"))
        if possession.get("assignment_reason"):
            current["assignment_reasons"].append(possession.get("assignment_reason"))
        if possession.get("confidence") is not None:
            current["confidences"].append(float(possession["confidence"]))
        if possession.get("distance_m") is not None:
            current["distances_m"].append(float(possession["distance_m"]))
        if possession.get("image_contact") is not None:
            current["image_contacts"].append(float(possession["image_contact"]))

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
    image_contacts = segment.pop("image_contacts", [])
    sources = segment.get("sources", [])
    reasons = segment.get("assignment_reasons", [])
    if confidences:
        segment["mean_confidence"] = round(float(np.mean(confidences)), 3)
        segment["max_confidence"] = round(float(np.max(confidences)), 3)
    if distances:
        segment["mean_distance_m"] = round(float(np.mean(distances)), 3)
    if image_contacts:
        segment["mean_image_contact"] = round(float(np.mean(image_contacts)), 3)
        segment["max_image_contact"] = round(float(np.max(image_contacts)), 3)
    if sources:
        segment["source_counts"] = count_values(sources)
    else:
        segment.pop("sources", None)
    if reasons:
        segment["assignment_reason_counts"] = count_values(reasons)
    else:
        segment.pop("assignment_reasons", None)


def count_values(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


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
