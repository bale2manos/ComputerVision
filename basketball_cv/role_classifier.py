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
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        if rec.get("role") not in (None, "player"):
            continue
        if rec.get("player_candidate") is False:
            continue
        tracks[int(rec["track_id"])].append(rec)

    track_features = []
    track_ids = []
    for track_id, track_records in tracks.items():
        feature = median_track_feature(track_records)
        if feature is not None:
            track_ids.append(track_id)
            track_features.append(feature)

    if len(track_features) < 2:
        return {"enabled": True, "tracks_used": len(track_features), "status": "not_enough_tracks"}

    matrix = np.asarray(track_features, dtype=np.float32)
    centers, assignment = kmeans(matrix, 2)
    labels = label_clusters_by_brightness(centers)
    track_to_team = {track_id: labels[int(cluster)] for track_id, cluster in zip(track_ids, assignment)}

    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        if rec.get("role") not in (None, "player"):
            continue
        team = track_to_team.get(int(rec["track_id"]))
        if team is None:
            continue
        rec["team_observation_before_generic_cluster"] = rec.get("team")
        rec["team"] = team
        rec["player_candidate"] = bool(rec.get("on_court") is not False)
        rec["track_player_candidate"] = True

    return {
        "enabled": True,
        "tracks_used": len(track_ids),
        "labels": labels,
        "track_to_team": {str(k): v for k, v in sorted(track_to_team.items())},
    }


def median_track_feature(track_records: list[dict[str, Any]]) -> np.ndarray | None:
    embeddings = [np.asarray(r["jersey_embedding"], dtype=np.float32) for r in track_records if r.get("jersey_embedding")]
    if embeddings:
        return np.median(np.asarray(embeddings, dtype=np.float32), axis=0)
    colors = [np.asarray(r["jersey_bgr"], dtype=np.float32) for r in track_records if r.get("jersey_bgr")]
    if len(colors) < 2:
        return None
    colors = np.asarray(colors, dtype=np.float32)
    # Normalize BGR so the cluster is not tied to red/dark names.
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
    # Generic names: stable and not tied to concrete colors.
    if centers.shape[1] >= 3:
        brightness = centers[:, :3].mean(axis=1)
        order = list(np.argsort(brightness))
        return {int(order[0]): "team_a", int(order[1]): "team_b"}
    return {0: "team_a", 1: "team_b"}
