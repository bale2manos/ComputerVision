from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


UNKNOWN_TEAM = "unknown"
REFEREE_TEAM = "referee"


@dataclass
class TeamModel:
    centers_bgr: np.ndarray
    labels: list[str]
    max_distance: float = 85.0

    def predict(self, color_bgr: np.ndarray | None) -> str:
        if color_bgr is None or len(self.centers_bgr) == 0:
            return UNKNOWN_TEAM
        color = np.asarray(color_bgr, dtype=np.float32).reshape(1, 3)
        dists = np.linalg.norm(self.centers_bgr.astype(np.float32) - color, axis=1)
        idx = int(np.argmin(dists))
        if float(dists[idx]) > self.max_distance:
            return UNKNOWN_TEAM
        return self.labels[idx]


def jersey_color(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    # Upper torso, avoiding shorts and some background.
    tx1 = x1 + int(0.22 * bw)
    tx2 = x2 - int(0.22 * bw)
    ty1 = y1 + int(0.18 * bh)
    ty2 = y1 + int(0.58 * bh)
    crop = frame[max(0, ty1) : min(h, ty2), max(0, tx1) : min(w, tx2)]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > 35) & (val > 35)
    if int(mask.sum()) < 20:
        mask = val > 25
    pixels = crop[mask]
    if len(pixels) < 20:
        return None
    return np.median(pixels, axis=0).astype(np.float32)


def jersey_stats(frame: np.ndarray, xyxy: np.ndarray) -> dict[str, float] | None:
    crop = _torso_crop(frame, xyxy, x_pad=0.25, y1_ratio=0.18, y2_ratio=0.48)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    red = ((hue <= 10) | (hue >= 165)) & (sat > 70) & (val > 50)
    dark = val < 95
    gray = (sat < 45) & (val > 60)
    return {
        "red_fraction": round(float(red.mean()), 4),
        "dark_fraction": round(float(dark.mean()), 4),
        "gray_fraction": round(float(gray.mean()), 4),
        "median_hue": round(float(np.median(hue)), 2),
        "median_saturation": round(float(np.median(sat)), 2),
        "median_value": round(float(np.median(val)), 2),
    }


def jersey_embedding(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Central-crop visual embedding inspired by the SigLIP crop step in Roboflow's pipeline."""

    crop = _torso_crop(frame, xyxy, x_pad=0.25, y1_ratio=0.16, y2_ratio=0.62)
    if crop is None:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (val > 35) & (sat > 25)
    if int(mask.sum()) < 25:
        mask = val > 25
    if int(mask.sum()) < 25:
        return None

    hue_hist = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [18], [0, 180]).flatten()
    sat_hist = cv2.calcHist([hsv], [1], mask.astype(np.uint8), [8], [0, 256]).flatten()
    val_hist = cv2.calcHist([hsv], [2], mask.astype(np.uint8), [8], [0, 256]).flatten()
    lab_a_hist = cv2.calcHist([lab], [1], mask.astype(np.uint8), [8], [0, 256]).flatten()
    lab_b_hist = cv2.calcHist([lab], [2], mask.astype(np.uint8), [8], [0, 256]).flatten()

    stats = jersey_stats(frame, xyxy) or {}
    stat_vec = np.asarray(
        [
            stats.get("red_fraction", 0.0),
            stats.get("dark_fraction", 0.0),
            stats.get("gray_fraction", 0.0),
            stats.get("median_hue", 0.0) / 180.0,
            stats.get("median_saturation", 0.0) / 255.0,
            stats.get("median_value", 0.0) / 255.0,
        ],
        dtype=np.float32,
    )

    embedding = np.concatenate([hue_hist, sat_hist, val_hist, lab_a_hist, lab_b_hist, stat_vec]).astype(np.float32)
    total = float(embedding.sum())
    if total > 0:
        embedding /= total
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding /= norm
    return embedding


def player_appearance_embedding(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Full-body appearance signature for same-team re-identification.

    This is intentionally lightweight: color histograms from body bands plus a
    small grayscale torso texture patch. It helps with crossings where two
    teammates share the same uniform color but have different numbers, shorts,
    shoes, sleeves, or body silhouette.
    """

    crop = _body_crop(frame, xyxy, x_pad=0.10, y1_ratio=0.02, y2_ratio=0.98)
    if crop is None:
        return None
    crop = cv2.resize(crop, (48, 96), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    features = []
    bands = [(0.02, 0.25), (0.20, 0.50), (0.47, 0.74), (0.70, 0.98)]
    for y_start, y_end in bands:
        y1 = int(round(y_start * crop.shape[0]))
        y2 = int(round(y_end * crop.shape[0]))
        x1 = int(round(0.12 * crop.shape[1]))
        x2 = int(round(0.88 * crop.shape[1]))
        band_hsv = hsv[y1:y2, x1:x2]
        band_lab = lab[y1:y2, x1:x2]
        if band_hsv.size == 0:
            continue
        mask = ((band_hsv[:, :, 2] > 25) & (band_hsv[:, :, 1] > 12)).astype(np.uint8)
        if int(mask.sum()) < 12:
            mask = (band_hsv[:, :, 2] > 20).astype(np.uint8)
        features.extend(
            [
                _normalized_hist(band_hsv, 0, mask, bins=12, value_range=(0, 180)),
                _normalized_hist(band_hsv, 1, mask, bins=5, value_range=(0, 256)),
                _normalized_hist(band_hsv, 2, mask, bins=6, value_range=(0, 256)),
                _normalized_hist(band_lab, 0, mask, bins=6, value_range=(0, 256)),
            ]
        )

    torso = crop[int(0.16 * crop.shape[0]) : int(0.58 * crop.shape[0]), int(0.24 * crop.shape[1]) : int(0.76 * crop.shape[1])]
    if torso.size:
        gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (8, 12), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gray = gray - float(gray.mean())
        std = float(gray.std())
        if std > 1e-4:
            gray = gray / std
        features.append(gray.flatten() * 0.18)

    if not features:
        return None
    embedding = np.concatenate(features).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        return None
    return embedding / norm


def _body_crop(
    frame: np.ndarray,
    xyxy: np.ndarray,
    x_pad: float = 0.10,
    y1_ratio: float = 0.02,
    y2_ratio: float = 0.98,
) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    tx1 = x1 + int(x_pad * bw)
    tx2 = x2 - int(x_pad * bw)
    ty1 = y1 + int(y1_ratio * bh)
    ty2 = y1 + int(y2_ratio * bh)
    crop = frame[max(0, ty1) : min(h, ty2), max(0, tx1) : min(w, tx2)]
    return crop if crop.size else None


def _normalized_hist(
    image: np.ndarray,
    channel: int,
    mask: np.ndarray,
    bins: int,
    value_range: tuple[int, int],
) -> np.ndarray:
    hist = cv2.calcHist([image], [channel], mask, [bins], list(value_range)).flatten().astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def _torso_crop(
    frame: np.ndarray,
    xyxy: np.ndarray,
    x_pad: float = 0.22,
    y1_ratio: float = 0.18,
    y2_ratio: float = 0.58,
) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    tx1 = x1 + int(x_pad * bw)
    tx2 = x2 - int(x_pad * bw)
    ty1 = y1 + int(y1_ratio * bh)
    ty2 = y1 + int(y2_ratio * bh)
    crop = frame[max(0, ty1) : min(h, ty2), max(0, tx1) : min(w, tx2)]
    return crop if crop.size else None


def bgr_to_hsv(color_bgr: np.ndarray | list[float] | None) -> np.ndarray | None:
    if color_bgr is None:
        return None
    sample = np.uint8([[np.clip(np.asarray(color_bgr, dtype=np.float32), 0, 255)]])
    return cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)


def is_player_like_jersey(color_hsv: np.ndarray | list[float] | None) -> bool:
    if color_hsv is None:
        return False
    _h, saturation, value = [float(v) for v in color_hsv]
    return saturation >= 35 and value >= 35


def fit_team_model(samples_bgr: list[np.ndarray], k: int = 2, seed: int = 7) -> TeamModel | None:
    if len(samples_bgr) < k * 4:
        return None
    samples = np.asarray(samples_bgr, dtype=np.float32).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    centers = samples[rng.choice(len(samples), size=k, replace=False)].copy()

    for _ in range(25):
        dists = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
        assignment = np.argmin(dists, axis=1)
        next_centers = centers.copy()
        for i in range(k):
            members = samples[assignment == i]
            if len(members):
                next_centers[i] = members.mean(axis=0)
        if np.allclose(next_centers, centers, atol=0.5):
            break
        centers = next_centers

    dists = np.linalg.norm(samples - centers[np.argmin(np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2), axis=1)], axis=1)
    spread = float(np.percentile(dists, 85)) if len(dists) else 45.0
    max_distance = float(np.clip(spread * 2.2, 45.0, 95.0))

    labels = [_label_color(center) for center in centers]
    if len(set(labels)) < len(labels):
        labels = [f"team_{chr(ord('a') + i)}" for i in range(len(labels))]
    return TeamModel(centers_bgr=centers, labels=labels, max_distance=max_distance)


def stabilize_team_identity(
    records: list[dict[str, Any]],
    min_track_samples: int = 4,
    min_clean_sample_ratio: float = 0.45,
    min_track_saturation: float = 40.0,
) -> dict[str, Any]:
    """Assign a single stable team to each track using clean central crop colors."""

    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") == "person" and rec.get("track_id") is not None:
            tracks[int(rec["track_id"])].append(rec)

    summaries: dict[int, dict[str, Any]] = {}
    embedding_reps: list[np.ndarray] = []
    embedding_track_ids: list[int] = []
    referee_like: dict[int, bool] = {}
    for track_id, track_records in tracks.items():
        samples = _clean_track_color_samples(track_records)
        embedding_samples = _clean_track_embedding_samples(track_records)
        summary = _summarize_track(track_id, track_records, samples, embedding_samples)
        summaries[track_id] = summary
        referee_like[track_id] = _track_is_referee_like(summary)
        if (
            not referee_like[track_id]
            and len(embedding_samples) >= min_track_samples
            and _track_is_team_candidate(summary, min_track_samples, min_clean_sample_ratio, min_track_saturation)
        ):
            embedding_reps.append(np.median(np.asarray(embedding_samples, dtype=np.float32), axis=0))
            embedding_track_ids.append(track_id)

    cluster_info = _cluster_tracks_by_embedding(embedding_track_ids, embedding_reps, summaries)
    for track_id, summary in summaries.items():
        if referee_like.get(track_id):
            summary["stable_team"] = UNKNOWN_TEAM
            summary["team_votes"] = {"referee_like": 1}
            continue

        strong_uniform_team = _classify_track_by_uniform_stats(summary, strong=True)
        if strong_uniform_team != UNKNOWN_TEAM:
            summary["stable_team"] = strong_uniform_team
            summary["team_votes"] = {"strong_uniform_rule": 1}
            continue

        if track_id in cluster_info["track_team"]:
            summary["stable_team"] = cluster_info["track_team"][track_id]
            summary["team_votes"] = {"embedding_cluster": int(cluster_info["track_cluster"][track_id])}
            continue

        fallback_team = _classify_track_by_uniform_stats(summary, strong=False)
        summary["stable_team"] = fallback_team
        summary["team_votes"] = {"uniform_rule_fallback": 1} if fallback_team != UNKNOWN_TEAM else {}

    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        summary = summaries.get(int(rec["track_id"]), {})
        stable_team = summary.get("stable_team", UNKNOWN_TEAM)
        rec["team_observation"] = rec.get("team", UNKNOWN_TEAM)
        rec["team"] = stable_team
        rec["track_player_candidate"] = bool(stable_team != UNKNOWN_TEAM and not rec.get("bottom_truncated"))
        rec["player_candidate"] = bool(
            rec["track_player_candidate"]
            and is_player_like_jersey(rec.get("jersey_hsv"))
            and rec.get("on_court") is not False
        )

    return {
        "identity_mode": "central_crop_embedding_cluster",
        "embedding_cluster": cluster_info["report"],
        "team_model": {
            "labels": cluster_info["report"].get("labels", []),
            "centers_bgr": [],
            "max_distance": None,
        },
        "tracks": [summaries[track_id] for track_id in sorted(summaries)],
        "track_count": len(summaries),
        "candidate_track_count": sum(1 for s in summaries.values() if s.get("stable_team") != UNKNOWN_TEAM),
    }


def split_mixed_team_tracks(
    records: list[dict[str, Any]],
    track_report: dict[str, Any],
    min_segment_frames: int = 24,
    max_hint_gap: int = 4,
) -> dict[str, Any]:
    """Split raw tracker IDs that contain sustained uniform changes.

    YOLO/ByteTrack can keep the same numeric track while the box jumps from one
    player to another during congestion. A single stable team per raw track then
    hides the error. This pass creates pseudo-track IDs for long same-team
    segments before player stitching runs.
    """

    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        rec.setdefault("detector_track_id", rec.get("track_id"))
        tracks[int(rec["detector_track_id"])].append(rec)

    original_summaries = {int(t["track_id"]): t for t in track_report.get("tracks", [])}
    split_reports = []
    for original_track_id, track_records in tracks.items():
        track_records = sorted(track_records, key=lambda r: int(r["frame_index"]))
        original_team = original_summaries.get(original_track_id, {}).get("stable_team", UNKNOWN_TEAM)
        split_points = _mixed_track_split_points(
            track_records,
            original_team=original_team,
            min_segment_frames=min_segment_frames,
            max_hint_gap=max_hint_gap,
        )
        if not split_points:
            for rec in track_records:
                rec["track_segment_index"] = 0
                rec["track_segment_team"] = rec.get("team", original_team)
            continue

        bounds = [0] + split_points + [len(track_records)]
        segment_report = []
        for segment_index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            segment_records = track_records[start:end]
            if not segment_records:
                continue
            segment_team = _segment_team(segment_records, fallback=original_team)
            segment_track_id = original_track_id * 1000 + segment_index
            for rec in segment_records:
                rec["track_id"] = segment_track_id
                rec["team"] = segment_team
                rec["track_segment_index"] = segment_index
                rec["track_segment_team"] = segment_team
                rec["track_player_candidate"] = bool(segment_team != UNKNOWN_TEAM and not rec.get("bottom_truncated"))
                rec["player_candidate"] = bool(
                    rec["track_player_candidate"]
                    and is_player_like_jersey(rec.get("jersey_hsv"))
                    and rec.get("on_court") is not False
                )
            segment_report.append(
                {
                    "track_id": segment_track_id,
                    "team": segment_team,
                    "first_frame": int(segment_records[0]["frame_index"]),
                    "last_frame": int(segment_records[-1]["frame_index"]),
                    "frames": len(segment_records),
                }
            )
        split_reports.append(
            {
                "detector_track_id": original_track_id,
                "original_team": original_team,
                "segments": segment_report,
            }
        )

    rebuilt = _rebuild_track_report_after_split(records, track_report, split_reports)
    rebuilt["mixed_track_split_parameters"] = {
        "min_segment_frames": min_segment_frames,
        "max_hint_gap": max_hint_gap,
    }
    return rebuilt


def _mixed_track_split_points(
    track_records: list[dict[str, Any]],
    original_team: str,
    min_segment_frames: int,
    max_hint_gap: int,
) -> list[int]:
    hints = [_record_uniform_hint(rec) for rec in track_records]
    candidate_runs = _hint_runs(hints, min_segment_frames=min_segment_frames, max_gap=max_hint_gap)
    if not candidate_runs:
        return []

    split_points = []
    current_team = original_team if original_team != UNKNOWN_TEAM else candidate_runs[0]["team"]
    for run in candidate_runs:
        if run["team"] == current_team:
            continue
        if run["start_index"] <= 0:
            current_team = run["team"]
            continue
        split_points.append(run["start_index"])
        current_team = run["team"]
    return sorted(set(split_points))


def _hint_runs(hints: list[str], min_segment_frames: int, max_gap: int) -> list[dict[str, Any]]:
    runs = []
    for team in ("red", "dark"):
        indices = [index for index, hint in enumerate(hints) if hint == team]
        if not indices:
            continue
        start = prev = indices[0]
        count = 1
        for index in indices[1:]:
            if index - prev <= max_gap + 1:
                prev = index
                count += 1
                continue
            if count >= min_segment_frames:
                runs.append({"team": team, "start_index": start, "end_index": prev, "frames": count})
            start = prev = index
            count = 1
        if count >= min_segment_frames:
            runs.append({"team": team, "start_index": start, "end_index": prev, "frames": count})
    return sorted(runs, key=lambda run: (run["start_index"], -run["frames"]))


def _record_uniform_hint(rec: dict[str, Any]) -> str:
    stats = rec.get("jersey_stats") or {}
    hsv = rec.get("jersey_hsv") or [0.0, 0.0, 0.0]
    red_fraction = float(stats.get("red_fraction", 0.0))
    dark_fraction = float(stats.get("dark_fraction", 0.0))
    hue = float(stats.get("median_hue", hsv[0] if len(hsv) > 0 else 0.0))
    saturation = float(stats.get("median_saturation", hsv[1] if len(hsv) > 1 else 0.0))
    value = float(stats.get("median_value", hsv[2] if len(hsv) > 2 else 0.0))

    if red_fraction >= 0.38 and dark_fraction < 0.45:
        return "red"
    if red_fraction >= 0.20 and dark_fraction < 0.25 and value > 115.0 and saturation > 55.0:
        return "red"
    if dark_fraction >= 0.48 and red_fraction < 0.32:
        return "dark"
    if dark_fraction >= 0.35 and value < 125.0 and red_fraction < 0.28:
        return "dark"
    if (hue <= 8.0 or hue >= 168.0) and saturation > 80.0 and value > 95.0 and dark_fraction < 0.35:
        return "red"
    return UNKNOWN_TEAM


def _segment_team(segment_records: list[dict[str, Any]], fallback: str) -> str:
    counts = Counter(_record_uniform_hint(rec) for rec in segment_records)
    counts.pop(UNKNOWN_TEAM, None)
    if counts:
        team, count = counts.most_common(1)[0]
        if count >= max(4, int(0.18 * len(segment_records))):
            return team
    return fallback


def _rebuild_track_report_after_split(
    records: list[dict[str, Any]],
    track_report: dict[str, Any],
    split_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") == "person" and rec.get("track_id") is not None:
            tracks[int(rec["track_id"])].append(rec)

    summaries = []
    for track_id, track_records in sorted(tracks.items()):
        samples = _clean_track_color_samples(track_records)
        embedding_samples = _clean_track_embedding_samples(track_records)
        summary = _summarize_track(track_id, track_records, samples, embedding_samples)
        teams = Counter(rec.get("team", UNKNOWN_TEAM) for rec in track_records)
        teams.pop(UNKNOWN_TEAM, None)
        stable_team = teams.most_common(1)[0][0] if teams else UNKNOWN_TEAM
        summary["stable_team"] = stable_team
        summary["team_votes"] = {"mixed_track_segment": 1} if any(rec.get("track_segment_index") for rec in track_records) else {}
        detector_ids = sorted({int(rec["detector_track_id"]) for rec in track_records if rec.get("detector_track_id") is not None})
        summary["detector_track_ids"] = detector_ids
        summaries.append(summary)

    rebuilt = dict(track_report)
    rebuilt["identity_mode"] = f"{track_report.get('identity_mode', 'team_identity')}+mixed_track_split"
    rebuilt["tracks"] = summaries
    rebuilt["track_count"] = len(summaries)
    rebuilt["candidate_track_count"] = sum(1 for summary in summaries if summary.get("stable_team") != UNKNOWN_TEAM)
    rebuilt["mixed_track_splits"] = split_reports
    rebuilt["mixed_track_split_count"] = len(split_reports)
    return rebuilt


def _clean_track_color_samples(track_records: list[dict[str, Any]]) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for rec in track_records:
        if rec.get("confidence", 0.0) < 0.35:
            continue
        if rec.get("bottom_truncated"):
            continue
        if rec.get("on_court") is False:
            continue
        if not rec.get("jersey_bgr") or not rec.get("jersey_hsv"):
            continue
        if not is_player_like_jersey(rec["jersey_hsv"]):
            continue
        samples.append(np.asarray(rec["jersey_bgr"], dtype=np.float32))
    return samples


def _clean_track_embedding_samples(track_records: list[dict[str, Any]]) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for rec in track_records:
        if rec.get("confidence", 0.0) < 0.35:
            continue
        if rec.get("bottom_truncated"):
            continue
        if rec.get("on_court") is False:
            continue
        if not rec.get("jersey_embedding"):
            continue
        samples.append(np.asarray(rec["jersey_embedding"], dtype=np.float32))
    return samples


def _summarize_track(
    track_id: int,
    records: list[dict[str, Any]],
    samples: list[np.ndarray],
    embedding_samples: list[np.ndarray],
) -> dict[str, Any]:
    frames = [int(r["frame_index"]) for r in records]
    hsv_samples = [np.asarray(r["jersey_hsv"], dtype=np.float32) for r in records if r.get("jersey_hsv")]
    stat_keys = ["red_fraction", "dark_fraction", "gray_fraction", "median_hue", "median_saturation", "median_value"]
    stat_summary = {}
    for key in stat_keys:
        values = [float(r["jersey_stats"][key]) for r in records if r.get("jersey_stats") and key in r["jersey_stats"]]
        if values:
            stat_summary[key] = round(float(np.median(values)), 4)
    median_hsv = np.median(np.asarray(hsv_samples), axis=0).round(2).tolist() if hsv_samples else None
    median_bgr = np.median(np.asarray(samples), axis=0).round(2).tolist() if samples else None
    median_embedding = np.median(np.asarray(embedding_samples), axis=0).round(5).tolist() if embedding_samples else None
    return {
        "track_id": track_id,
        "first_frame": min(frames),
        "last_frame": max(frames),
        "frames": len(records),
        "clean_color_samples": len(samples),
        "clean_embedding_samples": len(embedding_samples),
        "clean_sample_ratio": round(len(samples) / max(len(records), 1), 4),
        "stable_team": UNKNOWN_TEAM,
        "team_votes": {},
        "median_clean_bgr": median_bgr,
        "median_hsv": median_hsv,
        "median_embedding": median_embedding,
        "uniform_stats": stat_summary,
        "mean_confidence": round(float(np.mean([r.get("confidence", 0.0) for r in records])), 4),
        "bottom_truncated_frames": sum(1 for r in records if r.get("bottom_truncated")),
        "in_play_frames": 0,
    }


def _cluster_tracks_by_embedding(
    track_ids: list[int],
    embeddings: list[np.ndarray],
    summaries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    empty = {
        "track_team": {},
        "track_cluster": {},
        "report": {"labels": [], "cluster_count": 0, "tracks_used": 0},
    }
    if len(embeddings) < 2:
        return empty

    matrix = np.asarray(embeddings, dtype=np.float32)
    centers, assignment = _kmeans(matrix, k=2, seed=11)
    cluster_labels = _label_embedding_clusters(track_ids, assignment, summaries)
    if len(set(cluster_labels.values())) < 2:
        return empty

    track_team: dict[int, str] = {}
    track_cluster: dict[int, int] = {}
    for track_id, cluster_id in zip(track_ids, assignment):
        cluster_id = int(cluster_id)
        track_team[int(track_id)] = cluster_labels[cluster_id]
        track_cluster[int(track_id)] = cluster_id

    return {
        "track_team": track_team,
        "track_cluster": track_cluster,
        "report": {
            "labels": [cluster_labels[i] for i in sorted(cluster_labels)],
            "cluster_count": len(cluster_labels),
            "tracks_used": len(track_ids),
            "centers": centers.round(5).tolist(),
        },
    }


def _kmeans(matrix: np.ndarray, k: int, seed: int = 11, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(matrix) < k:
        raise ValueError("Not enough samples for k-means.")
    centers = matrix[rng.choice(len(matrix), size=k, replace=False)].copy()
    assignment = np.zeros(len(matrix), dtype=np.int32)
    for _ in range(iterations):
        dists = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
        next_assignment = np.argmin(dists, axis=1)
        next_centers = centers.copy()
        for i in range(k):
            members = matrix[next_assignment == i]
            if len(members):
                next_centers[i] = members.mean(axis=0)
        if np.array_equal(next_assignment, assignment) and np.allclose(next_centers, centers, atol=1e-5):
            break
        assignment = next_assignment
        centers = next_centers
    return centers, assignment


def _label_embedding_clusters(
    track_ids: list[int],
    assignment: np.ndarray,
    summaries: dict[int, dict[str, Any]],
) -> dict[int, str]:
    scores: dict[int, dict[str, float]] = {}
    for cluster_id in sorted(set(int(v) for v in assignment)):
        cluster_track_ids = [track_id for track_id, a in zip(track_ids, assignment) if int(a) == cluster_id]
        red_values = []
        dark_values = []
        for track_id in cluster_track_ids:
            stats = summaries[track_id].get("uniform_stats") or {}
            weight = max(float(summaries[track_id].get("frames", 1)), 1.0)
            red_values.extend([float(stats.get("red_fraction", 0.0))] * int(min(weight, 120)))
            dark_values.extend([float(stats.get("dark_fraction", 0.0))] * int(min(weight, 120)))
        scores[cluster_id] = {
            "red": float(np.median(red_values)) if red_values else 0.0,
            "dark": float(np.median(dark_values)) if dark_values else 0.0,
        }

    cluster_ids = sorted(scores)
    if len(cluster_ids) != 2:
        return {}
    red_cluster = max(cluster_ids, key=lambda i: scores[i]["red"] - scores[i]["dark"])
    dark_cluster = min(cluster_ids, key=lambda i: scores[i]["red"] - scores[i]["dark"])
    if red_cluster == dark_cluster:
        return {}
    return {red_cluster: "red", dark_cluster: "dark"}


def _classify_track_by_uniform_stats(summary: dict[str, Any], strong: bool = False) -> str:
    stats = summary.get("uniform_stats") or {}
    if not stats:
        return UNKNOWN_TEAM

    red_fraction = float(stats.get("red_fraction", 0.0))
    dark_fraction = float(stats.get("dark_fraction", 0.0))
    hue = float(stats.get("median_hue", 0.0))
    saturation = float(stats.get("median_saturation", 0.0))
    value = float(stats.get("median_value", 0.0))

    # Referees in this video are grey/blue-ish with moderate value and little base red/dark uniform.
    if 75.0 <= hue <= 145.0 and red_fraction < 0.18 and dark_fraction < 0.35 and saturation < 130.0:
        return UNKNOWN_TEAM

    if strong:
        if red_fraction >= 0.42 and dark_fraction < 0.35:
            return "red"
        if dark_fraction >= 0.48 and red_fraction < 0.24:
            return "dark"
        return UNKNOWN_TEAM

    if red_fraction >= 0.22 and dark_fraction < 0.45:
        return "red"
    if dark_fraction >= 0.35:
        return "dark"
    if 10.0 <= hue <= 45.0 and red_fraction < 0.22:
        return "dark"
    if value < 120.0 and red_fraction < 0.22:
        return "dark"
    return UNKNOWN_TEAM


def _track_is_referee_like(summary: dict[str, Any]) -> bool:
    stats = summary.get("uniform_stats") or {}
    if not stats:
        return False
    red_fraction = float(stats.get("red_fraction", 0.0))
    dark_fraction = float(stats.get("dark_fraction", 0.0))
    hue = float(stats.get("median_hue", 0.0))
    saturation = float(stats.get("median_saturation", 0.0))
    value = float(stats.get("median_value", 0.0))
    return 75.0 <= hue <= 145.0 and red_fraction < 0.18 and dark_fraction < 0.35 and saturation < 130.0 and value > 95.0


def _track_is_team_candidate(
    summary: dict[str, Any],
    min_track_samples: int,
    min_clean_sample_ratio: float,
    min_track_saturation: float,
) -> bool:
    if summary["clean_color_samples"] < min_track_samples:
        return False
    if summary["clean_sample_ratio"] < min_clean_sample_ratio:
        return False
    median_hsv = summary.get("median_hsv")
    if not median_hsv:
        return False
    if float(median_hsv[1]) < min_track_saturation:
        return False
    return True


def _label_color(center_bgr: np.ndarray) -> str:
    sample = np.uint8([[center_bgr]])
    h, s, v = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0, 0]
    if v < 80:
        return "dark"
    if s < 45:
        return "light"
    if h < 12 or h > 168:
        return "red"
    if 12 <= h < 35:
        return "orange"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 130:
        return "blue"
    return "purple"
