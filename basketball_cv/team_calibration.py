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
    This prevents the unsupervised clustering from swapping teams when uniforms
    change from red/black to white/blue/yellow/etc.
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

    referee_records = 0
    team_records = 0
    frame_overrides = 0
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
        frame_assignment = frame_assignments.get(sample_key)
        fallback_team = track_team.get(track_id)
        team = fallback_team
        source = "manual_track_majority"
        confidence = None
        if frame_assignment is not None and float(frame_assignment.get("margin", 0.0)) >= config.frame_override_margin:
            team = str(frame_assignment["team"])
            source = "manual_frame_prototype"
            confidence = frame_assignment.get("confidence")
            if fallback_team is not None and team != fallback_team:
                frame_overrides += 1
        elif frame_assignment is not None and fallback_team is None:
            team = str(frame_assignment["team"])
            source = "manual_frame_weak"
            confidence = frame_assignment.get("confidence")
        if team is None:
            continue
        rec["team"] = team
        rec["team_calibration_source"] = source
        if confidence is not None:
            rec["team_calibration_confidence"] = round(float(confidence), 4)
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
            "track_team": {str(k): v for k, v in sorted(track_team.items())},
        }
    )
    return report


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
