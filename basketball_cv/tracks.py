from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from basketball_cv.teams import UNKNOWN_TEAM


def stitch_track_fragments(
    records: list[dict[str, Any]],
    track_report: dict[str, Any],
    fps: float,
    max_gap_s: float = 1.6,
    base_distance_m: float = 1.35,
    max_speed_mps: float = 5.8,
    min_embedding_similarity: float = 0.78,
) -> dict[str, Any]:
    """Merge raw tracker fragments into stable player IDs.

    This is a lightweight ReID pass for the common failure mode where players touch,
    ByteTrack drops one ID, and the same person reappears a few frames later.
    """

    for rec in records:
        if rec.get("class_name") == "person":
            rec["raw_track_id"] = rec.get("track_id")
            rec["player_id"] = None

    segments = _build_segments(records, track_report)
    if not segments:
        return {"enabled": True, "player_count": 0, "merged_track_count": 0, "players": []}

    max_gap_frames = max(1, int(round(max_gap_s * fps)))
    players: list[dict[str, Any]] = []
    next_player_id = 1

    for segment in sorted(segments, key=lambda s: (s["first_frame"], s["track_id"])):
        match = _best_player_match(
            segment,
            players,
            fps=fps,
            max_gap_frames=max_gap_frames,
            base_distance_m=base_distance_m,
            max_speed_mps=max_speed_mps,
            min_embedding_similarity=min_embedding_similarity,
        )
        if match is None:
            match = {
                "player_id": next_player_id,
                "team": segment["team"],
                "track_ids": [],
                "first_frame": segment["first_frame"],
                "last_frame": segment["last_frame"],
                "last_pos": segment["last_pos"],
                "embedding": segment["embedding"],
                "frames": 0,
            }
            players.append(match)
            next_player_id += 1

        segment["player_id"] = match["player_id"]
        match["track_ids"].append(segment["track_id"])
        match["first_frame"] = min(match["first_frame"], segment["first_frame"])
        match["last_frame"] = max(match["last_frame"], segment["last_frame"])
        match["last_pos"] = segment["last_pos"]
        match["frames"] += segment["frames"]
        match["embedding"] = _merge_embeddings(match["embedding"], segment["embedding"])

    track_to_player = {segment["track_id"]: segment["player_id"] for segment in segments}
    for rec in records:
        if rec.get("class_name") != "person":
            continue
        track_id = rec.get("track_id")
        if track_id is None:
            continue
        rec["player_id"] = track_to_player.get(int(track_id))

    player_report = []
    for player in sorted(players, key=lambda p: p["player_id"]):
        player_report.append(
            {
                "player_id": player["player_id"],
                "team": player["team"],
                "track_ids": player["track_ids"],
                "first_frame": player["first_frame"],
                "last_frame": player["last_frame"],
                "frames": player["frames"],
            }
        )

    return {
        "enabled": True,
        "player_count": len(players),
        "merged_track_count": sum(max(0, len(p["track_ids"]) - 1) for p in player_report),
        "players": player_report,
        "parameters": {
            "max_gap_s": max_gap_s,
            "base_distance_m": base_distance_m,
            "max_speed_mps": max_speed_mps,
            "min_embedding_similarity": min_embedding_similarity,
        },
    }


def resolve_crossing_id_switches(
    records: list[dict[str, Any]],
    fps: float,
    close_distance_m: float = 1.25,
    lookaround_s: float = 0.35,
    min_improvement_m: float = 0.45,
    min_appearance_improvement: float = 0.18,
    max_appearance_motion_penalty_m: float = 1.25,
) -> dict[str, Any]:
    """Fix identity swaps when two same-team players cross paths.

    If two player IDs get very close, compare motion continuity before and after
    the close interval. When the swapped assignment is clearly smoother, swap
    their player IDs from that point onward.
    """

    corrections: list[dict[str, Any]] = []
    handled_events: set[tuple[int, int, int, int]] = set()
    max_iterations = 5
    for _ in range(max_iterations):
        event = _find_best_crossing_switch(
            records,
            fps=fps,
            close_distance_m=close_distance_m,
            lookaround_s=lookaround_s,
            min_improvement_m=min_improvement_m,
            min_appearance_improvement=min_appearance_improvement,
            max_appearance_motion_penalty_m=max_appearance_motion_penalty_m,
            handled_events=handled_events,
        )
        if event is None:
            break
        _swap_player_ids_after(records, event["player_a"], event["player_b"], event["swap_from_frame"])
        handled_events.add(tuple(event["event_key"]))
        corrections.append(event)

    return {
        "enabled": True,
        "correction_count": len(corrections),
        "corrections": corrections,
        "parameters": {
            "close_distance_m": close_distance_m,
            "lookaround_s": lookaround_s,
            "min_improvement_m": min_improvement_m,
            "min_appearance_improvement": min_appearance_improvement,
            "max_appearance_motion_penalty_m": max_appearance_motion_penalty_m,
        },
    }


def summarize_players_from_records(
    records: list[dict[str, Any]],
    base_report: dict[str, Any] | None = None,
    crossing_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild player summary after post-processing player IDs.

    Crossing correction can split a raw tracker ID across two stable player IDs.
    This summary is record-based so it reflects that final assignment.
    """

    base_report = dict(base_report or {})
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("player_id") is None:
            continue
        by_player[int(rec["player_id"])].append(rec)

    players = []
    for player_id, player_records in sorted(by_player.items()):
        player_records = sorted(player_records, key=lambda r: int(r["frame_index"]))
        team_counts: dict[str, int] = defaultdict(int)
        for rec in player_records:
            team = rec.get("team")
            if team not in (None, UNKNOWN_TEAM):
                team_counts[str(team)] += 1
        team = max(team_counts.items(), key=lambda item: item[1])[0] if team_counts else UNKNOWN_TEAM

        track_segments = _player_track_segments(player_records)
        track_ids = sorted({segment["track_id"] for segment in track_segments})
        players.append(
            {
                "player_id": player_id,
                "team": team,
                "track_ids": track_ids,
                "track_segments": track_segments,
                "first_frame": int(player_records[0]["frame_index"]),
                "last_frame": int(player_records[-1]["frame_index"]),
                "frames": len(player_records),
                "id_switch_corrected_frames": sum(1 for rec in player_records if rec.get("id_switch_corrected")),
            }
        )

    base_report["player_count"] = len(players)
    base_report["merged_track_count"] = sum(max(0, len(player["track_ids"]) - 1) for player in players)
    base_report["players"] = players
    if crossing_report is not None:
        base_report["id_switch_correction_count"] = crossing_report.get("correction_count", 0)
        base_report["id_switch_corrections"] = crossing_report.get("corrections", [])
        base_report["id_switch_correction_parameters"] = crossing_report.get("parameters", {})
    return base_report


def _find_best_crossing_switch(
    records: list[dict[str, Any]],
    fps: float,
    close_distance_m: float,
    lookaround_s: float,
    min_improvement_m: float,
    min_appearance_improvement: float,
    max_appearance_motion_penalty_m: float,
    handled_events: set[tuple[int, int, int, int]] | None = None,
) -> dict[str, Any] | None:
    by_frame = _player_records_by_frame(records)
    if not by_frame:
        return None

    by_player = _player_records_by_id(records)
    lookaround = max(3, int(round(lookaround_s * fps)))
    close_runs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for frame_index, frame_records in by_frame.items():
        ids = sorted(frame_records)
        for i, player_a in enumerate(ids):
            rec_a = frame_records[player_a]
            for player_b in ids[i + 1 :]:
                rec_b = frame_records[player_b]
                if rec_a.get("team") != rec_b.get("team"):
                    continue
                dist = _record_distance(rec_a, rec_b)
                if dist <= close_distance_m:
                    close_runs[(player_a, player_b)].append(frame_index)

    best_event = None
    best_score = 0.0
    for (player_a, player_b), frames in close_runs.items():
        for start, end in _contiguous_runs(frames, max_gap=2):
            event_key = (min(player_a, player_b), max(player_a, player_b), start, end)
            if handled_events and event_key in handled_events:
                continue
            before_frame = start - lookaround
            after_frame = end + lookaround
            a_before = _nearest_player_record(by_frame, player_a, before_frame, max_offset=lookaround)
            b_before = _nearest_player_record(by_frame, player_b, before_frame, max_offset=lookaround)
            a_after = _nearest_player_record(by_frame, player_a, after_frame, max_offset=lookaround)
            b_after = _nearest_player_record(by_frame, player_b, after_frame, max_offset=lookaround)
            if not all([a_before, b_before, a_after, b_after]):
                continue

            current_cost = _record_distance(a_before, a_after) + _record_distance(b_before, b_after)
            swapped_cost = _record_distance(a_before, b_after) + _record_distance(b_before, a_after)
            motion_improvement = current_cost - swapped_cost
            appearance = _crossing_appearance_scores(by_player, player_a, player_b, start, end, lookaround)
            appearance_improvement = None
            if appearance is not None:
                appearance_improvement = appearance["swapped_similarity"] - appearance["current_similarity"]

            motion_trigger = motion_improvement > min_improvement_m
            appearance_trigger = (
                appearance_improvement is not None
                and appearance_improvement >= min_appearance_improvement
                and motion_improvement >= -max_appearance_motion_penalty_m
            )
            if not motion_trigger and not appearance_trigger:
                continue

            score = 0.0
            if motion_trigger:
                score = max(score, motion_improvement / max(min_improvement_m, 1e-6))
            if appearance_trigger and appearance_improvement is not None:
                score = max(score, appearance_improvement / max(min_appearance_improvement, 1e-6))
            if score > best_score:
                best_score = score
                best_event = {
                    "player_a": player_a,
                    "player_b": player_b,
                    "event_key": list(event_key),
                    "close_start_frame": start,
                    "close_end_frame": end,
                    "swap_from_frame": end + 1,
                    "current_cost_m": round(float(current_cost), 3),
                    "swapped_cost_m": round(float(swapped_cost), 3),
                    "motion_improvement_m": round(float(motion_improvement), 3),
                    "appearance_improvement": round(float(appearance_improvement), 4)
                    if appearance_improvement is not None
                    else None,
                    "current_appearance_similarity": round(float(appearance["current_similarity"]), 4)
                    if appearance is not None
                    else None,
                    "swapped_appearance_similarity": round(float(appearance["swapped_similarity"]), 4)
                    if appearance is not None
                    else None,
                    "decision": "appearance" if appearance_trigger and not motion_trigger else "motion",
                }
    return best_event


def _player_track_segments(player_records: list[dict[str, Any]]) -> list[dict[str, int]]:
    segments: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    prev_frame = None
    prev_track_id = None
    for rec in player_records:
        track_id = rec.get("raw_track_id", rec.get("track_id"))
        if track_id is None:
            continue
        track_id = int(track_id)
        frame = int(rec["frame_index"])
        starts_new = current is None or track_id != prev_track_id or (prev_frame is not None and frame - prev_frame > 1)
        if starts_new:
            if current is not None:
                segments.append(current)
            current = {"track_id": track_id, "first_frame": frame, "last_frame": frame, "frames": 1}
        else:
            current["last_frame"] = frame
            current["frames"] += 1
        prev_frame = frame
        prev_track_id = track_id
    if current is not None:
        segments.append(current)
    return segments


def _player_records_by_frame(records: list[dict[str, Any]]) -> dict[int, dict[int, dict[str, Any]]]:
    by_frame: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        if rec.get("class_name") != "person":
            continue
        if rec.get("player_id") is None or rec.get("court_x") is None or rec.get("court_y") is None:
            continue
        if rec.get("team") in (None, UNKNOWN_TEAM):
            continue
        by_frame[int(rec["frame_index"])][int(rec["player_id"])] = rec
    return by_frame


def _player_records_by_id(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("player_id") is None:
            continue
        if rec.get("team") in (None, UNKNOWN_TEAM):
            continue
        by_player[int(rec["player_id"])].append(rec)
    for player_id in by_player:
        by_player[player_id].sort(key=lambda r: int(r["frame_index"]))
    return by_player


def _crossing_appearance_scores(
    by_player: dict[int, list[dict[str, Any]]],
    player_a: int,
    player_b: int,
    close_start: int,
    close_end: int,
    lookaround: int,
) -> dict[str, float] | None:
    quiet_gap = max(3, lookaround // 3)
    a_before = _median_player_embedding(by_player, player_a, close_start - (3 * lookaround), close_start - quiet_gap)
    b_before = _median_player_embedding(by_player, player_b, close_start - (3 * lookaround), close_start - quiet_gap)
    a_after = _median_player_embedding(by_player, player_a, close_end + quiet_gap, close_end + (3 * lookaround))
    b_after = _median_player_embedding(by_player, player_b, close_end + quiet_gap, close_end + (3 * lookaround))
    if not all([a_before is not None, b_before is not None, a_after is not None, b_after is not None]):
        return None

    current_a = _cosine_similarity(a_before, a_after)
    current_b = _cosine_similarity(b_before, b_after)
    swapped_a = _cosine_similarity(a_before, b_after)
    swapped_b = _cosine_similarity(b_before, a_after)
    if None in (current_a, current_b, swapped_a, swapped_b):
        return None
    return {
        "current_similarity": float(current_a + current_b),
        "swapped_similarity": float(swapped_a + swapped_b),
    }


def _median_player_embedding(
    by_player: dict[int, list[dict[str, Any]]],
    player_id: int,
    start_frame: int,
    end_frame: int,
    min_samples: int = 4,
) -> np.ndarray | None:
    samples = []
    for rec in by_player.get(player_id, []):
        frame = int(rec["frame_index"])
        if frame < start_frame or frame > end_frame:
            continue
        if rec.get("confidence", 0.0) < 0.35 or rec.get("bottom_truncated") or rec.get("on_court") is False:
            continue
        embedding = rec.get("appearance_embedding") or rec.get("jersey_embedding")
        if not embedding:
            continue
        samples.append(np.asarray(embedding, dtype=np.float32))
    if len(samples) < min_samples:
        return None
    median = np.median(np.asarray(samples, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(median))
    return median / norm if norm > 0 else None


def _contiguous_runs(frames: list[int], max_gap: int = 2) -> list[tuple[int, int]]:
    if not frames:
        return []
    frames = sorted(set(frames))
    runs = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame - prev <= max_gap:
            prev = frame
        else:
            runs.append((start, prev))
            start = prev = frame
    runs.append((start, prev))
    return runs


def _nearest_player_record(
    by_frame: dict[int, dict[int, dict[str, Any]]],
    player_id: int,
    target_frame: int,
    max_offset: int,
) -> dict[str, Any] | None:
    best = None
    best_offset = max_offset + 1
    for offset in range(max_offset + 1):
        for frame in (target_frame - offset, target_frame + offset):
            rec = by_frame.get(frame, {}).get(player_id)
            if rec is not None and offset < best_offset:
                best = rec
                best_offset = offset
        if best is not None:
            return best
    return None


def _swap_player_ids_after(records: list[dict[str, Any]], player_a: int, player_b: int, start_frame: int) -> None:
    for rec in records:
        if rec.get("class_name") != "person" or int(rec.get("frame_index", -1)) < start_frame:
            continue
        if rec.get("player_id") == player_a:
            rec["player_id"] = player_b
            rec["id_switch_corrected"] = True
        elif rec.get("player_id") == player_b:
            rec["player_id"] = player_a
            rec["id_switch_corrected"] = True


def _record_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(np.hypot(float(a["court_x"]) - float(b["court_x"]), float(a["court_y"]) - float(b["court_y"])))


def _build_segments(records: list[dict[str, Any]], track_report: dict[str, Any]) -> list[dict[str, Any]]:
    report_by_track = {int(t["track_id"]): t for t in track_report.get("tracks", [])}
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        if rec.get("team") in (None, UNKNOWN_TEAM):
            continue
        if rec.get("court_x") is None or rec.get("court_y") is None:
            continue
        by_track[int(rec["track_id"])].append(rec)

    segments = []
    for track_id, track_records in by_track.items():
        track_records = sorted(track_records, key=lambda r: int(r["frame_index"]))
        if len(track_records) < 3:
            continue
        start = _edge_position(track_records, from_start=True)
        end = _edge_position(track_records, from_start=False)
        if start is None or end is None:
            continue
        summary = report_by_track.get(track_id, {})
        embedding = np.asarray(summary.get("median_embedding"), dtype=np.float32) if summary.get("median_embedding") else None
        if embedding is not None and float(np.linalg.norm(embedding)) > 0:
            embedding = embedding / float(np.linalg.norm(embedding))
        segments.append(
            {
                "track_id": track_id,
                "team": track_records[0].get("team"),
                "first_frame": int(track_records[0]["frame_index"]),
                "last_frame": int(track_records[-1]["frame_index"]),
                "first_pos": start,
                "last_pos": end,
                "embedding": embedding,
                "frames": len(track_records),
            }
        )
    return segments


def _edge_position(track_records: list[dict[str, Any]], from_start: bool) -> np.ndarray | None:
    edge = track_records[:8] if from_start else track_records[-8:]
    points = [[r["court_x"], r["court_y"]] for r in edge if r.get("court_x") is not None and r.get("court_y") is not None]
    if not points:
        return None
    return np.median(np.asarray(points, dtype=np.float32), axis=0)


def _best_player_match(
    segment: dict[str, Any],
    players: list[dict[str, Any]],
    fps: float,
    max_gap_frames: int,
    base_distance_m: float,
    max_speed_mps: float,
    min_embedding_similarity: float,
) -> dict[str, Any] | None:
    best_player = None
    best_score = float("inf")
    for player in players:
        if player["team"] != segment["team"]:
            continue
        if player["last_frame"] >= segment["first_frame"]:
            continue

        gap_frames = segment["first_frame"] - player["last_frame"]
        if gap_frames > max_gap_frames:
            continue

        dt = max(gap_frames / max(fps, 1e-6), 1e-6)
        distance_m = float(np.linalg.norm(segment["first_pos"] - player["last_pos"]))
        allowed_distance = base_distance_m + max_speed_mps * dt
        if distance_m > allowed_distance:
            continue

        similarity = _cosine_similarity(player.get("embedding"), segment.get("embedding"))
        if similarity is not None and similarity < min_embedding_similarity:
            # If two fragments are nearly touching in 2D, allow lower visual confidence.
            if distance_m > base_distance_m * 0.75:
                continue

        visual_penalty = 0.0 if similarity is None else max(0.0, 1.0 - similarity)
        score = (distance_m / allowed_distance) + visual_penalty * 1.4 + (gap_frames / max_gap_frames) * 0.2
        if score < best_score:
            best_score = score
            best_player = player
    return best_player


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return None
    return float(np.dot(a, b) / denom)


def _merge_embeddings(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None:
        return b
    if b is None:
        return a
    merged = (a + b) / 2.0
    norm = float(np.linalg.norm(merged))
    return merged / norm if norm > 0 else merged
