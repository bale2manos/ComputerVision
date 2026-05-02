from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]

UNKNOWN_TEAM = "unknown"
REFEREE_TEAM = "referee"
PLAYER_ROLE_NAMES = {"player", "jugador", "in_play_player", "basketball_player"}
REFEREE_ROLE_NAMES = {"referee", "arbitro", "arbitro_ref", "official"}
OTHER_ROLE_NAMES = {"other", "spectator", "bench", "coach", "ignore", "background", "non_player"}


@dataclass
class RoleClassifierConfig:
    min_confidence: float = 0.55
    crop_margin: float = 0.08
    sample_step: int = 3
    relabel_unknown_below_confidence: bool = False
    generic_team_clustering: bool = True


class PersonRoleClassifier:
    """Classify tracked people as player/referee/other using full-body crops.

    This is deliberately separate from team identity. Referees should be filtered
    before team clustering, then players are clustered into generic team_a/team_b
    from their jersey appearance. This keeps the pipeline general when jerseys are
    yellow/blue, white/green, etc.
    """

    def __init__(self, model_path: str | Path, config: RoleClassifierConfig | None = None) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is required to use --role-model")
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.config = config or RoleClassifierConfig()
        self.names = getattr(self.model, "names", {}) or {}

    def apply_to_records(self, records: list[dict[str, Any]], video_path: str | Path, fps: float) -> dict[str, Any]:
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            if rec.get("class_name") == "person":
                by_frame[int(rec.get("frame_index", 0))].append(rec)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for role classifier: {video_path}")

        track_votes: dict[int, list[tuple[str, float]]] = defaultdict(list)
        current_frame = -1
        frame_img = None
        predicted_crops = 0
        for frame_index in sorted(by_frame):
            if self.config.sample_step > 1 and frame_index % self.config.sample_step != 0:
                continue
            if frame_index != current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_img = cap.read()
                current_frame = frame_index
                if not ok:
                    frame_img = None
            if frame_img is None:
                continue

            for rec in by_frame[frame_index]:
                track_id = rec.get("track_id")
                if track_id is None:
                    continue
                crop = person_crop(frame_img, rec, self.config.crop_margin)
                if crop is None:
                    continue
                pred = self.predict_crop(crop)
                if pred is None:
                    continue
                predicted_crops += 1
                role = normalize_role(pred["name"])
                conf = float(pred["confidence"])
                if conf >= self.config.min_confidence:
                    track_votes[int(track_id)].append((role, conf))

        cap.release()
        track_roles = finalize_track_roles(track_votes)
        apply_roles(records, track_roles, self.config)
        team_report = {}
        if self.config.generic_team_clustering:
            team_report = apply_generic_team_clustering(records)
        return {
            "role_model": str(self.model_path),
            "classes": self.names,
            "predicted_crops": predicted_crops,
            "track_roles": {str(k): v for k, v in sorted(track_roles.items())},
            "role_counts": dict(Counter(v["role"] for v in track_roles.values())),
            "generic_team_clustering": team_report,
            "parameters": self.config.__dict__,
        }

    def predict_crop(self, crop: np.ndarray) -> dict[str, Any] | None:
        results = self.model.predict(crop, verbose=False)
        if not results:
            return None
        probs = getattr(results[0], "probs", None)
        if probs is None:
            return None
        top1 = int(probs.top1)
        return {
            "class_id": top1,
            "name": str(self.names.get(top1, top1)),
            "confidence": float(probs.top1conf),
        }


def person_crop(frame: np.ndarray, rec: dict[str, Any], margin: float = 0.08) -> np.ndarray | None:
    bbox = rec.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    pad = max(w, h) * margin
    x1 = max(0, int(round(x1 - pad)))
    y1 = max(0, int(round(y1 - pad)))
    x2 = min(frame.shape[1], int(round(x2 + pad)))
    y2 = min(frame.shape[0], int(round(y2 + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def normalize_role(name: str) -> str:
    value = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if value in PLAYER_ROLE_NAMES:
        return "player"
    if value in REFEREE_ROLE_NAMES:
        return "referee"
    if value in OTHER_ROLE_NAMES:
        return "other"
    return value


def finalize_track_roles(track_votes: dict[int, list[tuple[str, float]]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for track_id, votes in track_votes.items():
        scores: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for role, conf in votes:
            scores[role] += conf
            counts[role] += 1
        role = max(scores, key=scores.get)
        output[track_id] = {
            "role": role,
            "confidence": round(scores[role] / max(counts[role], 1), 4),
            "votes": dict(counts),
        }
    return output


def apply_roles(records: list[dict[str, Any]], track_roles: dict[int, dict[str, Any]], config: RoleClassifierConfig) -> None:
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        role_info = track_roles.get(int(rec["track_id"]))
        if role_info is None:
            continue
        role = role_info["role"]
        rec["role"] = role
        rec["role_confidence"] = role_info["confidence"]
        if role == "referee":
            rec["team"] = REFEREE_TEAM
            rec["player_candidate"] = False
            rec["in_play_player"] = False
            rec["track_player_candidate"] = False
        elif role == "other":
            rec["team"] = UNKNOWN_TEAM
            rec["player_candidate"] = False
            rec["in_play_player"] = False
            rec["track_player_candidate"] = False
        elif role == "player":
            rec["track_player_candidate"] = True
            if rec.get("on_court") is not False:
                rec["player_candidate"] = True


def apply_generic_team_clustering(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Cluster player uniforms into generic team_a/team_b with frame-level correction.

    Earlier versions assigned one team per track. That fails when the tracker ID
    jumps from one player to another during congestion: the new player inherits
    the old track's team. This version clusters clean per-frame jersey samples,
    uses track majority as a fallback, and lets high-margin frame samples override
    the track label. It remains color-agnostic: labels are generic team_a/team_b.
    """

    samples = collect_team_samples(records)
    if len(samples) < 8:
        return {"enabled": True, "samples_used": len(samples), "status": "not_enough_samples"}

    matrix = np.asarray([sample["feature"] for sample in samples], dtype=np.float32)
    centers, assignment = kmeans(matrix, 2)
    labels = {0: "team_a", 1: "team_b"}

    sample_assignments: dict[tuple[int, int], dict[str, Any]] = {}
    track_votes: dict[int, Counter[str]] = defaultdict(Counter)
    for sample, cluster_id in zip(samples, assignment):
        distances = np.linalg.norm(centers - sample["feature"], axis=1)
        order = np.argsort(distances)
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        margin = float(distances[second] - distances[best])
        confidence = float(margin / max(float(distances[second]), 1e-6)) if second != best else 1.0
        team = labels[int(cluster_id)]
        key = (int(sample["frame_index"]), int(sample["track_id"]))
        sample_assignments[key] = {
            "team": team,
            "cluster": int(cluster_id),
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
        }
        track_votes[int(sample["track_id"])][team] += 1

    track_to_team: dict[int, str] = {}
    track_confidence: dict[int, float] = {}
    ambiguous_tracks = []
    for track_id, votes in track_votes.items():
        total = sum(votes.values())
        top = votes.most_common(2)
        best_team, best_count = top[0]
        second_count = top[1][1] if len(top) > 1 else 0
        dominance = (best_count - second_count) / max(total, 1)
        if best_count >= 3 and dominance >= 0.12:
            track_to_team[track_id] = best_team
            track_confidence[track_id] = round(float(dominance), 4)
        else:
            ambiguous_tracks.append(track_id)

    frame_level_overrides = 0
    track_fallbacks = 0
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        if rec.get("role") not in (None, "player"):
            continue
        track_id = int(rec["track_id"])
        old_team = rec.get("team")
        rec["team_observation_before_generic_cluster"] = old_team

        sample_key = (int(rec.get("frame_index", 0)), track_id)
        sample_team = sample_assignments.get(sample_key)
        fallback_team = track_to_team.get(track_id)
        assigned_team = fallback_team
        source = "track_majority"
        confidence = track_confidence.get(track_id)

        # If the current frame's jersey crop is clearly in the opposite cluster,
        # trust it. This handles tracker ID switches like a red player inheriting
        # a previous black player's track ID.
        if sample_team is not None and float(sample_team["confidence"]) >= 0.035:
            assigned_team = str(sample_team["team"])
            source = "frame_cluster"
            confidence = float(sample_team["confidence"])
            if assigned_team != fallback_team:
                frame_level_overrides += 1
        elif fallback_team is not None:
            track_fallbacks += 1

        if assigned_team is None:
            continue
        rec["team"] = assigned_team
        rec["team_cluster_source"] = source
        rec["team_cluster_confidence"] = round(float(confidence or 0.0), 4)
        rec["player_candidate"] = bool(rec.get("on_court") is not False)
        rec["track_player_candidate"] = True

    return {
        "enabled": True,
        "mode": "frame_aware_generic_cluster",
        "samples_used": len(samples),
        "tracks_used": len(track_votes),
        "labels": labels,
        "track_to_team": {str(k): v for k, v in sorted(track_to_team.items())},
        "track_confidence": {str(k): v for k, v in sorted(track_confidence.items())},
        "ambiguous_tracks": [int(v) for v in sorted(ambiguous_tracks)],
        "frame_level_overrides": frame_level_overrides,
        "track_fallbacks": track_fallbacks,
    }


def collect_team_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        if rec.get("role") not in (None, "player"):
            continue
        if rec.get("player_candidate") is False:
            continue
        if rec.get("on_court") is False or rec.get("bottom_truncated"):
            continue
        if float(rec.get("confidence") or 0.0) < 0.35:
            continue
        feature = record_team_feature(rec)
        if feature is None:
            continue
        samples.append(
            {
                "track_id": int(rec["track_id"]),
                "frame_index": int(rec.get("frame_index", 0)),
                "feature": feature,
            }
        )
    return samples


def record_team_feature(rec: dict[str, Any]) -> np.ndarray | None:
    if rec.get("jersey_embedding"):
        return np.asarray(rec["jersey_embedding"], dtype=np.float32)
    if rec.get("jersey_bgr"):
        color = np.asarray(rec["jersey_bgr"], dtype=np.float32)
        norm = float(np.linalg.norm(color))
        return color / norm if norm > 0 else color
    return None


def median_track_feature(track_records: list[dict[str, Any]]) -> np.ndarray | None:
    embeddings = [np.asarray(r["jersey_embedding"], dtype=np.float32) for r in track_records if r.get("jersey_embedding")]
    if embeddings:
        return np.median(np.asarray(embeddings, dtype=np.float32), axis=0)
    colors = [np.asarray(r["jersey_bgr"], dtype=np.float32) for r in track_records if r.get("jersey_bgr")]
    if len(colors) < 2:
        return None
    colors = np.asarray(colors, dtype=np.float32)
    med = np.median(colors, axis=0)
    norm = float(np.linalg.norm(med))
    return med / norm if norm > 0 else med


def kmeans(matrix: np.ndarray, k: int = 2, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic farthest-pair initialization.
    dists = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(dists)), dists.shape)
    centers = matrix[[i, j]].copy()
    assignment = np.zeros(len(matrix), dtype=np.int32)
    for _ in range(iterations):
        d = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
        new_assignment = np.argmin(d, axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            members = matrix[new_assignment == idx]
            if len(members):
                new_centers[idx] = members.mean(axis=0)
        if np.array_equal(new_assignment, assignment) and np.allclose(new_centers, centers):
            break
        assignment = new_assignment
        centers = new_centers
    return centers, assignment


def label_clusters_by_brightness(centers: np.ndarray) -> dict[int, str]:
    if centers.shape[1] >= 3:
        brightness = centers[:, :3].mean(axis=1)
        order = list(np.argsort(brightness))
        return {int(order[0]): "team_a", int(order[1]): "team_b"}
    return {0: "team_a", 1: "team_b"}
