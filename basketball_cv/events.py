from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PickRollConfig:
    ball_possession_radius_m: float = 1.6
    defender_radius_m: float = 2.4
    screen_defender_radius_m: float = 1.35
    screen_handler_radius_m: float = 3.2
    min_duration_s: float = 0.25
    merge_gap_s: float = 0.35


@dataclass
class BallOwnershipConfig:
    possession_radius_m: float = 1.65
    keep_owner_radius_m: float = 2.15
    max_ball_gap_s: float = 0.45
    min_owner_confidence: float = 0.3
    min_pass_gap_s: float = 0.08
    max_pass_gap_s: float = 2.2
    min_owner_frames: int = 2
    require_same_team: bool = True
    image_distance_weight: float = 0.9
    court_distance_weight: float = 0.65
    ball_overlap_bonus: float = 0.95
    same_team_bias: float = 0.35
    opponent_team_penalty: float = 0.55
    flight_min_distance_m: float = 1.8
    flight_receiver_radius_m: float = 1.75


def assign_ball_ownership(
    records: list[dict[str, Any]],
    fps: float,
    config: BallOwnershipConfig | None = None,
) -> dict[str, Any]:
    """Attach ball-owner fields to ball/player records.

    This is deliberately conservative: it only claims possession when the ball is
    close to an in-play player in court coordinates. A previous owner gets a
    slightly larger radius so dribbles and short occlusions do not flicker.
    """

    config = config or BallOwnershipConfig()
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_frame[int(rec["frame_index"])].append(rec)

    previous_owner: int | None = None
    previous_team: str | None = None
    owner_counts: dict[int, int] = defaultdict(int)
    owned_frames = 0
    ball_frames = 0
    for frame_index in sorted(by_frame):
        frame_records = by_frame[frame_index]
        for rec in frame_records:
            rec.pop("has_ball", None)
            rec.pop("ball_owner_player_id", None)
            rec.pop("ball_owner_identity", None)
            rec.pop("ball_owner_jersey_number", None)
            rec.pop("ball_owner_team", None)
            rec.pop("ball_owner_distance_m", None)
            rec.pop("ball_owner_confidence", None)

        ball = _best_frame_ball(frame_records)
        if ball is None:
            previous_owner = None
            previous_team = None
            continue

        ball_frames += 1
        players = _eligible_players(frame_records)
        owner = _choose_ball_owner(ball, players, previous_owner, previous_team, config)
        if owner is None:
            previous_owner = None
            previous_team = None
            continue

        owner_id = _player_key(owner)
        if owner_id is None:
            previous_owner = None
            previous_team = None
            continue

        distance_m = _distance(ball, owner)
        ownership_score = _ownership_score(ball, owner, previous_owner, previous_team, config)
        confidence = _ownership_confidence(distance_m, ownership_score, config)
        owner_identity = _record_identity(owner)
        ball["ball_owner_player_id"] = owner_id
        ball["ball_owner_identity"] = owner_identity
        ball["ball_owner_jersey_number"] = owner.get("jersey_number")
        ball["ball_owner_team"] = owner.get("team")
        ball["ball_owner_distance_m"] = round(float(distance_m), 3)
        ball["ball_owner_image_overlap"] = round(float(_ball_player_overlap_fraction(ball, owner)), 3)
        ball["ball_owner_score"] = round(float(ownership_score), 3)
        ball["ball_owner_confidence"] = round(float(confidence), 3)

        owner["has_ball"] = True
        owner["ball_owner_confidence"] = round(float(confidence), 3)
        owner_counts[owner_id] += 1
        owned_frames += 1
        previous_owner = owner_id
        previous_team = owner.get("team")

    return {
        "ball_frames": ball_frames,
        "owned_ball_frames": owned_frames,
        "unique_owners": len(owner_counts),
        "owner_frame_counts": {str(k): int(v) for k, v in sorted(owner_counts.items())},
        "parameters": {
            "possession_radius_m": config.possession_radius_m,
            "keep_owner_radius_m": config.keep_owner_radius_m,
            "max_ball_gap_s": config.max_ball_gap_s,
            "min_owner_confidence": config.min_owner_confidence,
        },
    }


def interpolate_ball_gaps(
    records: list[dict[str, Any]],
    fps: float,
    max_gap_s: float = 0.45,
) -> dict[str, Any]:
    """Fill short missing-ball gaps with linear records for stable possession."""

    if fps <= 0:
        return {"interpolated_ball_frames": 0, "max_gap_s": max_gap_s}

    by_frame: dict[int, dict[str, Any]] = {}
    for rec in records:
        if rec.get("class_name") != "sports ball":
            continue
        frame = int(rec["frame_index"])
        current = by_frame.get(frame)
        if current is None or float(rec.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
            by_frame[frame] = rec
            rec["ball_track_id"] = rec.get("ball_track_id", 1)

    max_gap_frames = max(1, int(round(max_gap_s * fps)))
    synthetic: list[dict[str, Any]] = []
    frames = sorted(by_frame)
    for start_frame, end_frame in zip(frames[:-1], frames[1:]):
        gap = end_frame - start_frame - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        start = by_frame[start_frame]
        end = by_frame[end_frame]
        if not _can_interpolate_ball(start, end, fps, gap):
            continue
        for frame in range(start_frame + 1, end_frame):
            alpha = (frame - start_frame) / float(end_frame - start_frame)
            rec = dict(start)
            rec["frame_index"] = frame
            rec["time_s"] = round(frame / fps, 4)
            rec["source"] = "interpolated_ball"
            rec["track_id"] = None
            rec["ball_track_id"] = 1
            rec["confidence"] = round(min(float(start.get("confidence") or 0.0), float(end.get("confidence") or 0.0)) * 0.55, 4)
            rec["bbox"] = _lerp_list(start.get("bbox"), end.get("bbox"), alpha, ndigits=2)
            rec["anchor_px"] = _lerp_list(start.get("anchor_px"), end.get("anchor_px"), alpha, ndigits=2)
            rec["court_x"] = _lerp_value(start.get("court_x"), end.get("court_x"), alpha, ndigits=3)
            rec["court_y"] = _lerp_value(start.get("court_y"), end.get("court_y"), alpha, ndigits=3)
            rec["interpolated_ball"] = True
            synthetic.append(rec)

    records.extend(synthetic)
    records.sort(key=lambda rec: (int(rec.get("frame_index", 0)), 1 if rec.get("class_name") == "sports ball" else 0))
    return {"interpolated_ball_frames": len(synthetic), "max_gap_s": max_gap_s}


def densify_ball_track_for_render(
    records: list[dict[str, Any]],
    fps: float,
    frame_count: int | None = None,
    max_linear_gap_s: float = 3.0,
    max_linear_speed_mps: float = 22.0,
    cover_edges: bool = True,
) -> dict[str, Any]:
    """Create a visible ball record for every frame.

    The dense records are render estimates. They are added after pass detection so
    they do not create fake possession events.
    """

    if fps <= 0:
        return {"dense_ball_frames": 0, "dense_ball_estimated_frames": 0}

    records[:] = [rec for rec in records if not str(rec.get("source", "")).startswith("dense_ball_")]
    existing = _best_ball_by_frame(records)
    if not existing:
        return {"dense_ball_frames": 0, "dense_ball_estimated_frames": 0}

    max_frame = frame_count - 1 if frame_count is not None and frame_count > 0 else max(
        int(rec.get("frame_index", 0)) for rec in records
    )
    max_linear_gap = max(1, int(round(max_linear_gap_s * fps)))
    known_frames = sorted(existing)
    synthetic: list[dict[str, Any]] = []
    used_frames = set(existing)

    for start_frame, end_frame in zip(known_frames[:-1], known_frames[1:]):
        start = existing[start_frame]
        end = existing[end_frame]
        gap = end_frame - start_frame - 1
        if gap <= 0:
            continue
        linear_ok = gap <= max_linear_gap and _can_dense_linear_ball(start, end, fps, gap, max_linear_speed_mps)
        for frame in range(start_frame + 1, end_frame):
            if frame in used_frames:
                continue
            if linear_ok:
                alpha = (frame - start_frame) / float(end_frame - start_frame)
                synthetic.append(_make_dense_ball_record(start, end, frame, fps, alpha, "dense_ball_interpolated"))
            else:
                anchor = start if frame - start_frame <= end_frame - frame else end
                synthetic.append(_make_dense_ball_hold(anchor, frame, fps, "dense_ball_hold_gap"))
            used_frames.add(frame)

    if cover_edges:
        first_frame = known_frames[0]
        last_frame = known_frames[-1]
        for frame in range(0, min(first_frame, max_frame + 1)):
            if frame not in used_frames:
                synthetic.append(_make_dense_ball_hold(existing[first_frame], frame, fps, "dense_ball_hold_before"))
                used_frames.add(frame)
        for frame in range(last_frame + 1, max_frame + 1):
            if frame not in used_frames:
                synthetic.append(_make_dense_ball_hold(existing[last_frame], frame, fps, "dense_ball_hold_after"))
                used_frames.add(frame)

    for rec in existing.values():
        rec["ball_track_id"] = rec.get("ball_track_id", 1)
        rec["dense_ball"] = True
        rec["dense_ball_estimate"] = False

    records.extend(synthetic)
    records.sort(key=lambda rec: (int(rec.get("frame_index", 0)), 1 if rec.get("class_name") == "sports ball" else 0))
    return {
        "dense_ball_frames": len(existing) + len(synthetic),
        "dense_ball_estimated_frames": len(synthetic),
        "dense_ball_known_frames": len(existing),
        "dense_ball_max_linear_gap_s": max_linear_gap_s,
    }


def detect_passes(
    records: list[dict[str, Any]],
    fps: float,
    config: BallOwnershipConfig | None = None,
) -> list[dict[str, Any]]:
    """Detect simple possession-change pass events from ball ownership."""

    config = config or BallOwnershipConfig()
    frame_owner = _frame_owner_sequence(records)
    if not frame_owner or fps <= 0:
        return []

    min_gap = max(1, int(round(config.min_pass_gap_s * fps)))
    max_gap = max(min_gap, int(round(config.max_pass_gap_s * fps)))
    owner_runs = _owner_runs(frame_owner, min_frames=config.min_owner_frames)
    events: list[dict[str, Any]] = []
    for prev_run, next_run in zip(owner_runs[:-1], owner_runs[1:]):
        if prev_run["owner_id"] == next_run["owner_id"]:
            continue
        gap = int(next_run["start_frame"]) - int(prev_run["end_frame"]) - 1
        if gap < min_gap or gap > max_gap:
            continue

        prev_rec = prev_run["record"]
        next_rec = next_run["record"]
        if config.require_same_team:
            prev_team = prev_rec.get("ball_owner_team") or prev_rec.get("team")
            next_team = next_rec.get("ball_owner_team") or next_rec.get("team")
            if prev_team and next_team and prev_team != next_team:
                continue

        confidence = min(
            float(prev_run["confidence"]),
            float(next_run["confidence"]),
            _gap_confidence(gap, fps, config),
        )
        events.append(
            {
                "type": "pass",
                "start_frame": int(prev_run["end_frame"]),
                "end_frame": int(next_run["start_frame"]),
                "start_time_s": round(int(prev_run["end_frame"]) / fps, 3),
                "end_time_s": round(int(next_run["start_frame"]) / fps, 3),
                "from_player_id": int(prev_run["owner_id"]),
                "to_player_id": int(next_run["owner_id"]),
                "from_identity": prev_run["identity"],
                "to_identity": next_run["identity"],
                "from_jersey_number": prev_run["jersey_number"],
                "to_jersey_number": next_run["jersey_number"],
                "team": prev_rec.get("ball_owner_team") or prev_rec.get("team"),
                "confidence": round(float(confidence), 3),
                "toast": f"Jugador {_display_identity(prev_run)} pasa a jugador {_display_identity(next_run)}",
            }
        )
    owner_events = events
    flight_events = _detect_ball_flight_passes(records, fps, config, owner_events)
    events = list(flight_events)
    for event in owner_events:
        if float(event.get("confidence") or 0.0) < 0.55:
            continue
        if _overlaps_any_pass(event, flight_events):
            continue
        events.append(event)
    events.sort(key=lambda event: int(event.get("start_frame", 0)))
    return events


def detect_pick_and_rolls(records: list[dict[str, Any]], fps: float, config: PickRollConfig | None = None) -> list[dict[str, Any]]:
    config = config or PickRollConfig()
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_frame[int(rec["frame_index"])].append(rec)

    hits: list[dict[str, Any]] = []
    for frame_index in sorted(by_frame):
        frame_records = by_frame[frame_index]
        players = [
            r
            for r in frame_records
            if r.get("class_name") == "person"
            and not r.get("is_estimated")
            and r.get("in_play_player", r.get("player_candidate"))
            and r.get("track_id") is not None
            and r.get("team") not in (None, "unknown")
            and r.get("court_x") is not None
        ]
        balls = [r for r in frame_records if r.get("class_name") == "sports ball" and r.get("court_x") is not None]
        if not players or not balls:
            continue

        ball = _largest_confident_ball(balls)
        handler = _nearest(ball, players)
        if handler is None or _distance(ball, handler) > config.ball_possession_radius_m:
            continue

        teammates = [p for p in players if p["team"] == handler["team"] and p["track_id"] != handler["track_id"]]
        opponents = [p for p in players if p["team"] != handler["team"]]
        if not teammates or not opponents:
            continue

        defender = _nearest(handler, opponents)
        if defender is None or _distance(handler, defender) > config.defender_radius_m:
            continue

        for screener in teammates:
            d_sd = _distance(screener, defender)
            d_sh = _distance(screener, handler)
            if d_sd <= config.screen_defender_radius_m and d_sh <= config.screen_handler_radius_m:
                hits.append(
                    {
                        "frame_index": frame_index,
                        "handler_id": handler["track_id"],
                        "screener_id": screener["track_id"],
                        "defender_id": defender["track_id"],
                        "team": handler["team"],
                        "screen_defender_distance_m": round(float(d_sd), 3),
                        "screen_handler_distance_m": round(float(d_sh), 3),
                    }
                )

    return _merge_hits(hits, fps, config)


def _eligible_players(frame_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in frame_records
        if r.get("class_name") == "person"
        and not r.get("is_estimated")
        and r.get("in_play_player", r.get("player_candidate"))
        and _player_key(r) is not None
        and r.get("team") not in (None, "unknown")
        and r.get("court_x") is not None
        and r.get("court_y") is not None
    ]


def _best_frame_ball(frame_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    balls = [
        r
        for r in frame_records
        if r.get("class_name") == "sports ball" and r.get("court_x") is not None and r.get("court_y") is not None
    ]
    return _largest_confident_ball(balls) if balls else None


def _best_ball_by_frame(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    for rec in records:
        if rec.get("class_name") != "sports ball":
            continue
        if str(rec.get("source", "")).startswith("dense_ball_"):
            continue
        frame = int(rec["frame_index"])
        current = by_frame.get(frame)
        if current is None or float(rec.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
            by_frame[frame] = rec
    return by_frame


def _choose_ball_owner(
    ball: dict[str, Any],
    players: list[dict[str, Any]],
    previous_owner: int | None,
    previous_team: str | None,
    config: BallOwnershipConfig,
) -> dict[str, Any] | None:
    if not players:
        return None
    ranked = sorted(players, key=lambda player: _ownership_score(ball, player, previous_owner, previous_team, config))
    best = ranked[0]
    best_distance = _distance(ball, best)
    best_owner_id = _player_key(best)
    radius = config.keep_owner_radius_m if previous_owner is not None and best_owner_id == previous_owner else config.possession_radius_m
    image_overlap = _ball_player_overlap_fraction(ball, best)
    if best_distance > radius and image_overlap <= 0.08:
        if previous_owner is None:
            return None
        previous = [player for player in players if _player_key(player) == previous_owner]
        if not previous:
            return None
        previous_distance = _distance(ball, previous[0])
        if previous_distance <= config.keep_owner_radius_m:
            return previous[0]
        return None
    if _ownership_confidence(best_distance, _ownership_score(ball, best, previous_owner, previous_team, config), config) < config.min_owner_confidence:
        return None
    return best


def _ownership_score(
    ball: dict[str, Any],
    player: dict[str, Any],
    previous_owner: int | None,
    previous_team: str | None,
    config: BallOwnershipConfig,
) -> float:
    court_score = _distance(ball, player) / max(config.possession_radius_m, 1e-6)
    image_score = _ball_player_image_distance(ball, player)
    overlap = _ball_player_overlap_fraction(ball, player)
    score = config.court_distance_weight * court_score + config.image_distance_weight * image_score
    score -= config.ball_overlap_bonus * overlap
    player_id = _player_key(player)
    if previous_owner is not None and player_id == previous_owner:
        score -= 0.25
    if previous_team is not None and player.get("team") == previous_team:
        score -= config.same_team_bias
    elif previous_team is not None and player.get("team") not in (None, previous_team):
        score += config.opponent_team_penalty
    return float(score)


def _ownership_confidence(distance_m: float, score: float, config: BallOwnershipConfig) -> float:
    distance_conf = 1.0 - distance_m / max(config.keep_owner_radius_m, 1e-6)
    score_conf = 1.0 - max(score, 0.0) / 2.0
    return max(0.0, min(1.0, 0.55 * distance_conf + 0.45 * score_conf))


def _player_key(rec: dict[str, Any]) -> int | None:
    value = rec.get("canonical_player_id", rec.get("player_id", rec.get("track_id")))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_identity(rec: dict[str, Any]) -> str | None:
    if rec.get("jersey_identity"):
        return str(rec["jersey_identity"])
    if rec.get("team") and rec.get("jersey_number"):
        return f"{rec['team']}_{rec['jersey_number']}"
    player_id = rec.get("player_id")
    return f"P{player_id}" if player_id is not None else None


def _frame_owner_sequence(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    sequence: dict[int, dict[str, Any]] = {}
    for rec in records:
        if rec.get("class_name") != "sports ball" or rec.get("ball_owner_player_id") is None:
            continue
        frame = int(rec["frame_index"])
        current = sequence.get(frame)
        if current is None or float(rec.get("ball_owner_confidence") or 0.0) > float(current.get("ball_owner_confidence") or 0.0):
            sequence[frame] = rec
    return sequence


def _owner_runs(frame_owner: dict[int, dict[str, Any]], min_frames: int) -> list[dict[str, Any]]:
    runs = []
    current: dict[str, Any] | None = None
    previous_frame: int | None = None
    for frame in sorted(frame_owner):
        rec = frame_owner[frame]
        owner_id = int(rec["ball_owner_player_id"])
        starts_new = (
            current is None
            or owner_id != current["owner_id"]
            or (previous_frame is not None and frame - previous_frame > 1)
        )
        if starts_new:
            if current is not None and current["frames"] >= min_frames:
                runs.append(current)
            current = {
                "owner_id": owner_id,
                "start_frame": frame,
                "end_frame": frame,
                "frames": 1,
                "record": rec,
                "identity": rec.get("ball_owner_identity"),
                "jersey_number": rec.get("ball_owner_jersey_number"),
                "confidence": float(rec.get("ball_owner_confidence") or 0.0),
            }
        else:
            current["end_frame"] = frame
            current["frames"] += 1
            current["confidence"] = max(float(current["confidence"]), float(rec.get("ball_owner_confidence") or 0.0))
            if rec.get("ball_owner_jersey_number"):
                current["jersey_number"] = rec.get("ball_owner_jersey_number")
            if rec.get("ball_owner_identity"):
                current["identity"] = rec.get("ball_owner_identity")
            current["record"] = rec
        previous_frame = frame
    if current is not None and current["frames"] >= min_frames:
        runs.append(current)
    return runs


def _gap_confidence(gap_frames: int, fps: float, config: BallOwnershipConfig) -> float:
    gap_s = gap_frames / max(fps, 1e-6)
    if gap_s <= config.min_pass_gap_s:
        return 0.55
    return max(0.35, 1.0 - gap_s / max(config.max_pass_gap_s, 1e-6))


def _display_identity(run: dict[str, Any]) -> str:
    if run.get("jersey_number"):
        return str(run["jersey_number"])
    identity = run.get("identity")
    if identity and "_" in str(identity):
        return str(identity).split("_", 1)[1]
    return f"P{run['owner_id']}"


def _detect_ball_flight_passes(
    records: list[dict[str, Any]],
    fps: float,
    config: BallOwnershipConfig,
    existing_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_frame[int(rec["frame_index"])].append(rec)

    frame_balls = {frame: _best_frame_ball(frame_records) for frame, frame_records in by_frame.items()}
    frame_balls = {frame: ball for frame, ball in frame_balls.items() if ball is not None}
    events: list[dict[str, Any]] = []
    sparse_ball_gap = max(2, int(round(min(config.max_pass_gap_s, 1.2) * fps)))
    for start_frame, end_frame in _contiguous_runs(sorted(frame_balls), max_gap=sparse_ball_gap):
        duration_s = (end_frame - start_frame + 1) / max(fps, 1e-6)
        if duration_s < config.min_pass_gap_s or duration_s > config.max_pass_gap_s:
            continue

        start_ball = frame_balls[start_frame]
        end_ball = frame_balls[end_frame]
        if _distance(start_ball, end_ball) < config.flight_min_distance_m:
            continue

        start_players = _eligible_players(by_frame.get(start_frame, []))
        end_players = _eligible_players(by_frame.get(end_frame, []))
        sender = _choose_ball_owner(start_ball, start_players, None, None, config)
        if sender is None:
            continue
        sender_id = _player_key(sender)
        if sender_id is None:
            continue
        receiver = _choose_flight_receiver(end_ball, end_players, sender, config)
        if receiver is None:
            continue
        receiver_id = _player_key(receiver)
        if receiver_id is None or receiver_id == sender_id:
            continue
        if config.require_same_team and sender.get("team") != receiver.get("team"):
            continue
        if _is_duplicate_pass(existing_events + events, sender_id, receiver_id, start_frame, end_frame):
            continue

        sender_run = _run_from_record(sender_id, sender)
        receiver_run = _run_from_record(receiver_id, receiver)
        distance_m = _distance(start_ball, end_ball)
        receiver_distance = _distance(end_ball, receiver)
        confidence = min(0.86, 0.38 + min(distance_m / 7.0, 0.28) + max(0.0, 1.0 - receiver_distance / config.flight_receiver_radius_m) * 0.24)
        events.append(
            {
                "type": "pass",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": round(start_frame / fps, 3),
                "end_time_s": round(end_frame / fps, 3),
                "from_player_id": sender_id,
                "to_player_id": receiver_id,
                "from_identity": sender_run["identity"],
                "to_identity": receiver_run["identity"],
                "from_jersey_number": sender_run["jersey_number"],
                "to_jersey_number": receiver_run["jersey_number"],
                "team": sender.get("team"),
                "confidence": round(float(confidence), 3),
                "detection_mode": "ball_flight_same_team",
                "ball_travel_distance_m": round(float(distance_m), 3),
                "receiver_distance_m": round(float(receiver_distance), 3),
                "toast": f"Jugador {_display_identity(sender_run)} pasa a jugador {_display_identity(receiver_run)}",
            }
        )
    return events


def _choose_flight_receiver(
    ball: dict[str, Any],
    players: list[dict[str, Any]],
    sender: dict[str, Any],
    config: BallOwnershipConfig,
) -> dict[str, Any] | None:
    sender_id = _player_key(sender)
    candidates = [
        player
        for player in players
        if _player_key(player) is not None
        and _player_key(player) != sender_id
        and player.get("team") == sender.get("team")
    ]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda player: (
            _distance(ball, player) / max(config.flight_receiver_radius_m, 1e-6)
            + 0.55 * _ball_player_image_distance(ball, player)
            - 0.35 * _ball_player_overlap_fraction(ball, player)
        ),
    )
    best = ranked[0]
    if _distance(ball, best) > config.flight_receiver_radius_m and _ball_player_image_distance(ball, best) > 0.9:
        return None
    return best


def _run_from_record(owner_id: int, rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "owner_id": owner_id,
        "identity": _record_identity(rec),
        "jersey_number": rec.get("jersey_number"),
        "record": rec,
        "confidence": float(rec.get("ball_owner_confidence") or rec.get("confidence") or 0.0),
    }


def _is_duplicate_pass(
    events: list[dict[str, Any]],
    sender_id: int,
    receiver_id: int,
    start_frame: int,
    end_frame: int,
    tolerance_frames: int = 90,
) -> bool:
    for event in events:
        if event.get("type") != "pass":
            continue
        if int(event.get("from_player_id", -1)) != sender_id or int(event.get("to_player_id", -1)) != receiver_id:
            continue
        if abs(int(event.get("start_frame", start_frame)) - start_frame) <= tolerance_frames:
            return True
        if abs(int(event.get("end_frame", end_frame)) - end_frame) <= tolerance_frames:
            return True
    return False


def _overlaps_any_pass(event: dict[str, Any], others: list[dict[str, Any]], tolerance_frames: int = 90) -> bool:
    start = int(event.get("start_frame", 0))
    end = int(event.get("end_frame", start))
    for other in others:
        other_start = int(other.get("start_frame", 0))
        other_end = int(other.get("end_frame", other_start))
        if max(start, other_start) <= min(end, other_end) + tolerance_frames:
            return True
    return False


def _contiguous_runs(frames: list[int], max_gap: int) -> list[tuple[int, int]]:
    if not frames:
        return []
    runs = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame - prev <= max_gap:
            prev = frame
            continue
        runs.append((start, prev))
        start = prev = frame
    runs.append((start, prev))
    return runs


def _ball_player_overlap_fraction(ball: dict[str, Any], player: dict[str, Any]) -> float:
    ball_box = ball.get("bbox")
    player_box = player.get("bbox")
    if not isinstance(ball_box, list) or not isinstance(player_box, list):
        return 0.0
    bx1, by1, bx2, by2 = [float(v) for v in ball_box]
    px1, py1, px2, py2 = [float(v) for v in player_box]
    ix1, iy1 = max(bx1, px1), max(by1, py1)
    ix2, iy2 = min(bx2, px2), min(by2, py2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ball_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / ball_area if ball_area > 0 else 0.0


def _ball_player_image_distance(ball: dict[str, Any], player: dict[str, Any]) -> float:
    ball_box = ball.get("bbox")
    player_box = player.get("bbox")
    if not isinstance(ball_box, list) or not isinstance(player_box, list):
        return 1.0
    bx1, by1, bx2, by2 = [float(v) for v in ball_box]
    px1, py1, px2, py2 = [float(v) for v in player_box]
    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    dx = max(px1 - cx, 0.0, cx - px2)
    dy = max(py1 - cy, 0.0, cy - py2)
    box_h = max(py2 - py1, 1.0)
    return min(2.0, float(np.hypot(dx, dy) / box_h))


def _can_interpolate_ball(start: dict[str, Any], end: dict[str, Any], fps: float, gap: int) -> bool:
    if start.get("court_x") is None or end.get("court_x") is None:
        return False
    distance_m = _distance(start, end)
    seconds = max((gap + 1) / max(fps, 1e-6), 1e-6)
    return distance_m / seconds <= 18.0


def _can_dense_linear_ball(
    start: dict[str, Any],
    end: dict[str, Any],
    fps: float,
    gap: int,
    max_speed_mps: float,
) -> bool:
    if start.get("court_x") is None or end.get("court_x") is None:
        return True
    distance_m = _distance(start, end)
    seconds = max((gap + 1) / max(fps, 1e-6), 1e-6)
    return distance_m / seconds <= max_speed_mps


def _make_dense_ball_record(
    start: dict[str, Any],
    end: dict[str, Any],
    frame: int,
    fps: float,
    alpha: float,
    source: str,
) -> dict[str, Any]:
    rec = dict(start)
    rec["frame_index"] = frame
    rec["time_s"] = round(frame / fps, 4)
    rec["source"] = source
    rec["track_id"] = None
    rec["ball_track_id"] = 1
    rec["confidence"] = round(min(float(start.get("confidence") or 0.0), float(end.get("confidence") or 0.0)) * 0.45, 4)
    rec["bbox"] = _lerp_list(start.get("bbox"), end.get("bbox"), alpha, ndigits=2)
    rec["anchor_px"] = _lerp_list(start.get("anchor_px"), end.get("anchor_px"), alpha, ndigits=2)
    rec["court_x"] = _lerp_value(start.get("court_x"), end.get("court_x"), alpha, ndigits=3)
    rec["court_y"] = _lerp_value(start.get("court_y"), end.get("court_y"), alpha, ndigits=3)
    rec["dense_ball"] = True
    rec["dense_ball_estimate"] = True
    rec["interpolated_ball"] = True
    _clear_ball_owner_fields(rec)
    return rec


def _make_dense_ball_hold(anchor: dict[str, Any], frame: int, fps: float, source: str) -> dict[str, Any]:
    rec = dict(anchor)
    rec["frame_index"] = frame
    rec["time_s"] = round(frame / fps, 4)
    rec["source"] = source
    rec["track_id"] = None
    rec["ball_track_id"] = 1
    rec["confidence"] = round(float(anchor.get("confidence") or 0.0) * 0.25, 4)
    rec["dense_ball"] = True
    rec["dense_ball_estimate"] = True
    rec["held_ball_estimate"] = True
    _clear_ball_owner_fields(rec)
    return rec


def _clear_ball_owner_fields(rec: dict[str, Any]) -> None:
    for key in (
        "ball_owner_player_id",
        "ball_owner_identity",
        "ball_owner_jersey_number",
        "ball_owner_team",
        "ball_owner_distance_m",
        "ball_owner_image_overlap",
        "ball_owner_score",
        "ball_owner_confidence",
    ):
        rec.pop(key, None)


def _lerp_list(a: Any, b: Any, alpha: float, ndigits: int) -> list[float] | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return a if isinstance(a, list) else None
    return [round(float(x) * (1.0 - alpha) + float(y) * alpha, ndigits) for x, y in zip(a, b)]


def _lerp_value(a: Any, b: Any, alpha: float, ndigits: int) -> float | None:
    if a is None or b is None:
        return None
    return round(float(a) * (1.0 - alpha) + float(b) * alpha, ndigits)


def _largest_confident_ball(balls: list[dict[str, Any]]) -> dict[str, Any]:
    return max(balls, key=lambda r: (float(r.get("confidence") or 0.0), float(r.get("bbox_area") or 0.0)))


def _nearest(target: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: _distance(target, c))


def _distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(np.hypot(float(a["court_x"]) - float(b["court_x"]), float(a["court_y"]) - float(b["court_y"])))


def _merge_hits(hits: list[dict[str, Any]], fps: float, config: PickRollConfig) -> list[dict[str, Any]]:
    if not hits or fps <= 0:
        return []

    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        key = (int(hit["handler_id"]), int(hit["screener_id"]), int(hit["defender_id"]))
        grouped[key].append(hit)

    events: list[dict[str, Any]] = []
    min_frames = max(1, int(round(config.min_duration_s * fps)))
    merge_gap = max(1, int(round(config.merge_gap_s * fps)))
    for (handler_id, screener_id, defender_id), group in grouped.items():
        group = sorted(group, key=lambda h: h["frame_index"])
        segment = [group[0]]
        for hit in group[1:]:
            if hit["frame_index"] - segment[-1]["frame_index"] <= merge_gap:
                segment.append(hit)
            else:
                _append_segment(events, segment, fps, min_frames, handler_id, screener_id, defender_id)
                segment = [hit]
        _append_segment(events, segment, fps, min_frames, handler_id, screener_id, defender_id)

    events.sort(key=lambda e: e["start_frame"])
    return events


def _append_segment(
    events: list[dict[str, Any]],
    segment: list[dict[str, Any]],
    fps: float,
    min_frames: int,
    handler_id: int,
    screener_id: int,
    defender_id: int,
) -> None:
    start = int(segment[0]["frame_index"])
    end = int(segment[-1]["frame_index"])
    if end - start + 1 < min_frames:
        return
    avg_sd = float(np.mean([s["screen_defender_distance_m"] for s in segment]))
    avg_sh = float(np.mean([s["screen_handler_distance_m"] for s in segment]))
    duration_s = (end - start + 1) / fps
    confidence = min(0.95, 0.35 + 0.15 * duration_s + max(0.0, 1.4 - avg_sd) * 0.25)
    events.append(
        {
            "type": "pick_and_roll_candidate",
            "start_frame": start,
            "end_frame": end,
            "start_time_s": round(start / fps, 3),
            "end_time_s": round(end / fps, 3),
            "handler_id": handler_id,
            "screener_id": screener_id,
            "defender_id": defender_id,
            "team": segment[0]["team"],
            "confidence": round(float(confidence), 3),
            "avg_screen_defender_distance_m": round(avg_sd, 3),
            "avg_screen_handler_distance_m": round(avg_sh, 3),
            "note": "Rule-based candidate; validate after ball and team tracking are tuned.",
        }
    )
