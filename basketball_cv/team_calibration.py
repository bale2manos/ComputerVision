from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

UNKNOWN_TEAM = "unknown"
REFEREE_TEAM = "referee"


@dataclass
class TeamCalibrationConfig:
    seed_window_frames: int = 90
    min_margin: float = 0.015
    max_distance: float = 0.72
    frame_override_margin: float = 0.025
    mark_referee_tracks: bool = True
    min_switch_frames: int = 10
    switch_max_gap_frames: int = 4
    strong_switch_margin: float = 0.045


def load_team_calibration(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_team_calibration(
    records: list[dict[str, Any]],
    calibration: dict[str, Any],
    config: TeamCalibrationConfig | None = None,
) -> dict[str, Any]:
    """Apply manual team prototypes picked at the start of a game.

    The calibration JSON contains clicked seed track ids for team_a, team_b and
    referee. For each team, we build a visual prototype from records close to the
    clicked frame. Then every player record is assigned to the nearest prototype.

    Team labels are temporally smoothed per track. A player cannot switch teams
    because of one contaminated crop; a switch only happens after sustained
    evidence. This keeps labels stable during contact, screens and crossings.
    """

    config = config or TeamCalibrationConfig()
    prototypes = build_team_prototypes(records, calibration, config)
    referee_track_ids = seed_track_ids(calibration, "referee")
    report: dict[str, Any] = {
        "enabled": True,
        "prototype_status": {team: proto.get("status") for team, proto in prototypes.items()},
        "referee_track_ids": sorted(referee_track_ids),
        "parameters": config.__dict__,
    }
    if "team_a" not in prototypes or "team_b" not in prototypes:
        report["status"] = "missing_team_prototypes"
        return report

    track_votes: dict[int, Counter[str]] = defaultdict(Counter)
    frame_assignments: dict[tuple[int, int], dict[str, Any]] = {}
    assigned_frames = 0
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        track_id = int(rec["track_id"])
        if track_id in referee_track_ids:
            continue
        if rec.get("role") not in (None, "player"):
            continue
        if rec.get("on_court") is False or rec.get("bottom_truncated"):
            continue
        feature = record_feature(rec)
        if feature is None:
            continue
        assignment = classify_feature(feature, prototypes, config)
        if assignment is None:
            continue
        key = (int(rec.get("frame_index", 0)), track_id)
        frame_assignments[key] = assignment
        track_votes[track_id][assignment["team"]] += 1
        assigned_frames += 1

    track_team: dict[int, str] = {}
    for track_id, votes in track_votes.items():
        if not votes:
            continue
        top = votes.most_common(2)
        team, count = top[0]
        second = top[1][1] if len(top) > 1 else 0
        dominance = (count - second) / max(sum(votes.values()), 1)
        if count >= 2 and dominance >= 0.08:
            track_team[track_id] = team

    temporal_team, smoothing_report = build_temporal_team_assignments(
        records,
        frame_assignments,
        track_team,
        referee_track_ids,
        config,
    )

    referee_records = 0
    team_records = 0
    frame_overrides = 0
    weak_records = 0
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        track_id = int(rec["track_id"])
        rec["team_before_manual_calibration"] = rec.get("team")

        if config.mark_referee_tracks and track_id in referee_track_ids:
            rec["role"] = "referee"
            rec["team"] = REFEREE_TEAM
            rec["player_candidate"] = False
            rec["track_player_candidate"] = False
            rec["in_play_player"] = False
            rec["team_calibration_source"] = "manual_referee_seed"
            referee_records += 1
            continue

        if rec.get("role") not in (None, "player"):
            continue
        sample_key = (int(rec.get("frame_index", 0)), track_id)
        smoothed = temporal_team.get(sample_key)
        if smoothed is None:
            weak_records += 1
            continue
        team = smoothed["team"]
        rec["team"] = team
        rec["team_calibration_source"] = smoothed["source"]
        rec["team_calibration_confidence"] = round(float(smoothed.get("confidence") or 0.0), 4)
        if smoothed["source"] == "temporal_confirmed_switch":
            frame_overrides += 1
        rec["player_candidate"] = bool(rec.get("on_court") is not False)
        rec["track_player_candidate"] = True
        team_records += 1

    report.update(
        {
            "status": "ok",
            "assigned_frame_samples": assigned_frames,
            "team_records_updated": team_records,
            "referee_records_updated": referee_records,
            "frame_overrides": frame_overrides,
            "weak_unassigned_records": weak_records,
            "track_team": {str(k): v for k, v in sorted(track_team.items())},
            "temporal_smoothing": smoothing_report,
        }
    )
    return report


def build_temporal_team_assignments(
    records: list[dict[str, Any]],
    frame_assignments: dict[tuple[int, int], dict[str, Any]],
    track_team: dict[int, str],
    referee_track_ids: set[int],
    config: TeamCalibrationConfig,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    """Return smoothed team labels per frame/track.

    Per-frame prototype classification is allowed to correct tracker ID switches,
    but only after the alternative team is seen for several consecutive/nearby
    samples. This suppresses single-frame flicker caused by occlusion or a crop
    including another player's shirt.
    """

    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        track_id = int(rec["track_id"])
        if track_id in referee_track_ids:
            continue
        if rec.get("role") not in (None, "player"):
            continue
        by_track[track_id].append(rec)

    output: dict[tuple[int, int], dict[str, Any]] = {}
    confirmed_switches = 0
    suppressed_switches = 0
    tracks_smoothed = 0

    for track_id, track_records in by_track.items():
        track_records = sorted(track_records, key=lambda r: int(r.get("frame_index", 0)))
        current_team = track_team.get(track_id)
        if current_team is None:
            current_team = first_reliable_team(track_records, frame_assignments, track_id, config)
        pending_team: str | None = None
        pending_count = 0
        pending_last_frame: int | None = None
        pending_confidences: list[float] = []
        track_had_suppression = False

        for rec in track_records:
            frame = int(rec.get("frame_index", 0))
            key = (frame, track_id)
            raw = frame_assignments.get(key)
            observed_team = raw.get("team") if raw else None
            margin = float(raw.get("margin", 0.0)) if raw else 0.0
            confidence = float(raw.get("confidence", 0.0)) if raw else 0.0

            if current_team is None and observed_team is not None:
                current_team = str(observed_team)

            source = "temporal_track_hold"
            if observed_team is None or current_team is None:
                if current_team is None:
                    continue
                output[key] = {"team": current_team, "source": source, "confidence": confidence}
                continue

            observed_team = str(observed_team)
            if observed_team == current_team:
                pending_team = None
                pending_count = 0
                pending_last_frame = None
                pending_confidences.clear()
                source = "temporal_frame_agrees"
            elif margin >= config.strong_switch_margin:
                if pending_team != observed_team or pending_last_frame is None or frame - pending_last_frame > config.switch_max_gap_frames:
                    pending_team = observed_team
                    pending_count = 1
                    pending_confidences = [confidence]
                else:
                    pending_count += 1
                    pending_confidences.append(confidence)
                pending_last_frame = frame
                if pending_count >= config.min_switch_frames:
                    current_team = observed_team
                    confirmed_switches += 1
                    source = "temporal_confirmed_switch"
                    pending_team = None
                    pending_count = 0
                    pending_last_frame = None
                    pending_confidences.clear()
                else:
                    suppressed_switches += 1
                    track_had_suppression = True
                    source = "temporal_suppressed_switch"
            else:
                suppressed_switches += 1
                track_had_suppression = True
                source = "temporal_low_margin_hold"

            output[key] = {
                "team": current_team,
                "source": source,
                "confidence": confidence if source != "temporal_confirmed_switch" else float(np.mean(pending_confidences)) if pending_confidences else confidence,
                "raw_team": observed_team,
                "raw_margin": round(float(margin), 4),
            }

        if track_had_suppression:
            tracks_smoothed += 1

    return output, {
        "mode": "per_track_hysteresis",
        "confirmed_switches": confirmed_switches,
        "suppressed_switch_frames": suppressed_switches,
        "tracks_smoothed": tracks_smoothed,
    }


def first_reliable_team(
    track_records: list[dict[str, Any]],
    frame_assignments: dict[tuple[int, int], dict[str, Any]],
    track_id: int,
    config: TeamCalibrationConfig,
) -> str | None:
    votes: Counter[str] = Counter()
    for rec in track_records[: max(config.min_switch_frames * 3, 12)]:
        key = (int(rec.get("frame_index", 0)), track_id)
        raw = frame_assignments.get(key)
        if raw and float(raw.get("margin", 0.0)) >= config.frame_override_margin:
            votes[str(raw["team"])] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def build_team_prototypes(
    records: list[dict[str, Any]],
    calibration: dict[str, Any],
    config: TeamCalibrationConfig,
) -> dict[str, dict[str, Any]]:
    prototypes: dict[str, dict[str, Any]] = {}
    for team in ("team_a", "team_b"):
        features = []
        seeds = calibration.get(team, []) or []
        for seed in seeds:
            track_id = seed.get("track_id")
            frame = seed.get("frame_index", seed.get("frame"))
            if track_id is None:
                continue
            for rec in records:
                if rec.get("class_name") != "person" or rec.get("track_id") is None:
                    continue
                if int(rec["track_id"]) != int(track_id):
                    continue
                if frame is not None and abs(int(rec.get("frame_index", 0)) - int(frame)) > config.seed_window_frames:
                    continue
                feature = record_feature(rec)
                if feature is not None:
                    features.append(feature)
        if not features:
            prototypes[team] = {"status": "no_features", "samples": 0}
            continue
        matrix = np.asarray(features, dtype=np.float32)
        proto = np.median(matrix, axis=0)
        norm = float(np.linalg.norm(proto))
        if norm > 0:
            proto = proto / norm
        prototypes[team] = {"status": "ok", "samples": len(features), "feature": proto}
    return prototypes


def seed_track_ids(calibration: dict[str, Any], key: str) -> set[int]:
    ids = set()
    for item in calibration.get(key, []) or []:
        if item.get("track_id") is not None:
            ids.add(int(item["track_id"]))
    return ids


def classify_feature(
    feature: np.ndarray,
    prototypes: dict[str, dict[str, Any]],
    config: TeamCalibrationConfig,
) -> dict[str, Any] | None:
    distances = []
    for team in ("team_a", "team_b"):
        proto = prototypes.get(team, {})
        if proto.get("status") != "ok" or proto.get("feature") is None:
            continue
        distances.append((team, float(np.linalg.norm(feature - proto["feature"]))))
    if len(distances) < 2:
        return None
    distances.sort(key=lambda item: item[1])
    best_team, best_dist = distances[0]
    _second_team, second_dist = distances[1]
    margin = second_dist - best_dist
    if best_dist > config.max_distance and margin < config.min_margin:
        return None
    confidence = margin / max(second_dist, 1e-6)
    return {
        "team": best_team,
        "distance": round(best_dist, 4),
        "second_distance": round(second_dist, 4),
        "margin": round(margin, 4),
        "confidence": round(float(confidence), 4),
    }


def record_feature(rec: dict[str, Any]) -> np.ndarray | None:
    if rec.get("jersey_embedding"):
        feat = np.asarray(rec["jersey_embedding"], dtype=np.float32)
    elif rec.get("jersey_bgr"):
        feat = np.asarray(rec["jersey_bgr"], dtype=np.float32)
    else:
        return None
    norm = float(np.linalg.norm(feat))
    if norm <= 0:
        return None
    return feat / norm
