from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.court import CourtSpec, load_calibration
from basketball_cv.events import assign_ball_ownership, densify_ball_track_for_render, detect_passes, interpolate_ball_gaps
from tools.analyze_video import render_annotated_video, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an annotated video from existing tracks.json.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--tracks", required=True, help="tracks.json produced by tools/analyze_video.py.")
    parser.add_argument("--calibration", required=True, help="Court calibration JSON.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--jersey-numbers", default=None, help="jersey_numbers.json produced by tools/extract_jersey_numbers.py.")
    parser.add_argument("--identity-overrides", default=None, help="Optional JSON with manual track/frame identity overrides.")
    parser.add_argument("--events", default=None, help="Optional events.json to render pass toasts.")
    parser.add_argument("--output-events", default=None, help="Optional path to write pass events recomputed after OCR/overrides.")
    parser.add_argument("--no-detect-passes", action="store_true", help="Do not recompute ball ownership/pass events before rendering.")
    parser.add_argument("--no-dense-ball-track", action="store_true", help="Do not add render-only ball estimates for every frame.")
    parser.add_argument("--dense-ball-max-gap", type=float, default=3.0, help="Max seconds between ball detections to linearly connect in the dense render track.")
    parser.add_argument("--no-interpolate-jersey-gaps", action="store_true", help="Disable short-gap interpolation for stable jersey identities.")
    parser.add_argument("--max-interpolation-gap", type=int, default=75, help="Max missing frames to interpolate for a stable jersey identity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    if args.jersey_numbers:
        apply_jersey_numbers(records, Path(args.jersey_numbers))
    if args.identity_overrides:
        apply_identity_overrides(records, Path(args.identity_overrides))
    if args.jersey_numbers or args.identity_overrides:
        backfill_jersey_identities_across_fragments(records, float(tracks.get("fps", 60.0)))
        if not args.no_interpolate_jersey_gaps:
            interpolate_jersey_identity_gaps(records, float(tracks.get("fps", 60.0)), args.max_interpolation_gap)

    fps = float(tracks.get("fps", 60.0))
    events = load_events(Path(args.events)) if args.events else []
    if not args.no_detect_passes:
        interpolate_ball_gaps(records, fps)
        assign_ball_ownership(records, fps)
        recomputed_passes = detect_passes(records, fps)
        non_pass_events = [event for event in events if event.get("type") != "pass"]
        events = sorted(non_pass_events + recomputed_passes, key=lambda event: int(event.get("start_frame", 0)))
        if args.output_events:
            write_json(Path(args.output_events), {"video": args.video, "fps": fps, "events": events})
    if not args.no_dense_ball_track:
        frame_count = int(tracks.get("frame_count") or max((int(rec.get("frame_index", 0)) for rec in records), default=0) + 1)
        densify_ball_track_for_render(records, fps, frame_count=frame_count, max_linear_gap_s=args.dense_ball_max_gap)

    calibration = load_calibration(args.calibration)
    court_spec = CourtSpec(**calibration["court"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_annotated_video(args.video, output_path, records, fps, court_spec, events=events)
    print(output_path.resolve())


def apply_jersey_numbers(records: list[dict[str, Any]], jersey_numbers_path: Path) -> None:
    report = json.loads(jersey_numbers_path.read_text(encoding="utf-8"))
    apply_identity_segments(records, report.get("identity_segments", []))
    conflicting_player_ids = {
        int(conflict["player_id"])
        for conflict in report.get("identity_conflicts", [])
        if conflict.get("type") == "player_contains_multiple_jersey_identities" and conflict.get("player_id") is not None
    }
    numbers = {
        int(player["player_id"]): player
        for player in report.get("players", [])
        if player.get("jersey_number") is not None
    }
    for rec in records:
        player_id = rec.get("player_id")
        if player_id is None:
            continue
        if int(player_id) in conflicting_player_ids:
            rec["identity_conflict"] = "player_contains_multiple_jersey_identities"
            continue
        number_info = numbers.get(int(player_id))
        if number_info is not None:
            if rec.get("jersey_identity"):
                continue
            rec["jersey_number"] = str(number_info["jersey_number"])
            rec["jersey_identity"] = number_info.get("jersey_identity")
            rec["canonical_player_id"] = number_info.get("canonical_player_id")


def apply_identity_segments(records: list[dict[str, Any]], identity_segments: list[dict[str, Any]]) -> None:
    if not identity_segments:
        return
    segments_by_track: dict[int, list[dict[str, Any]]] = {}
    for segment in identity_segments:
        if segment.get("jersey_number") is None or segment.get("track_id") is None:
            continue
        segments_by_track.setdefault(int(segment["track_id"]), []).append(segment)

    for rec in records:
        if rec.get("class_name") != "person":
            continue
        candidate_ids = {
            int(value)
            for value in (rec.get("track_id"), rec.get("raw_track_id"), rec.get("detector_track_id"))
            if value is not None
        }
        frame = int(rec.get("frame_index", -1))
        matching = [
            segment
            for track_id in candidate_ids
            for segment in segments_by_track.get(track_id, [])
            if _segment_applies_to_frame(segment, frame)
        ]
        if not matching:
            continue
        segment = max(
            matching,
            key=lambda item: (
                float(item.get("score_by_number", {}).get(str(item.get("jersey_number")), 0.0)),
                int(item.get("vote_count", 0)),
            ),
        )
        team = segment.get("team")
        number = segment.get("jersey_number")
        identity = segment.get("jersey_identity") or (f"{team}_{number}" if team and number is not None else None)
        if team:
            rec["team"] = team
        if number is not None:
            rec["jersey_number"] = str(number)
        if identity:
            rec["jersey_identity"] = identity
        if segment.get("canonical_track_id") is not None:
            rec["canonical_track_id"] = segment["canonical_track_id"]
        rec["identity_source"] = "ocr_track_segment"


def _segment_applies_to_frame(segment: dict[str, Any], frame: int, margin_frames: int = 45) -> bool:
    if segment.get("apply_to_full_track", True):
        return True
    first = segment.get("first_vote_frame")
    last = segment.get("last_vote_frame")
    if first is None or last is None:
        return True
    return int(first) - margin_frames <= frame <= int(last) + margin_frames


def load_events(events_path: Path) -> list[dict[str, Any]]:
    data = json.loads(events_path.read_text(encoding="utf-8"))
    return data.get("events", [])


def backfill_jersey_identities_across_fragments(
    records: list[dict[str, Any]],
    fps: float,
    max_gap_frames: int = 120,
    max_speed_mps: float = 6.5,
    min_embedding_similarity: float = 0.32,
    min_candidate_frames: int = 20,
) -> int:
    segments = summarize_record_segments(records)
    labeled = [segment for segment in segments if segment.get("jersey_identity") and len(str(segment.get("jersey_number", ""))) >= 2]
    unlabeled = [segment for segment in segments if not segment.get("jersey_identity")]
    backfills: dict[int, dict[str, Any]] = {}

    for source in sorted(labeled, key=lambda item: int(item["first_frame"])):
        for candidate in unlabeled:
            if candidate["track_id"] in backfills:
                continue
            if candidate.get("team") != source.get("team"):
                continue
            if int(candidate.get("frames", 0)) < min_candidate_frames:
                continue
            if int(candidate["last_frame"]) >= int(source["first_frame"]):
                continue
            gap = int(source["first_frame"]) - int(candidate["last_frame"])
            if gap <= 0 or gap > max_gap_frames:
                continue
            distance = float(np.linalg.norm(np.asarray(source["first_pos"]) - np.asarray(candidate["last_pos"])))
            seconds = max(gap / max(fps, 1e-6), 1e-6)
            if distance / seconds > max_speed_mps:
                continue
            similarity = cosine_similarity(candidate.get("embedding"), source.get("embedding"))
            if similarity is not None and similarity < min_embedding_similarity and distance > 1.8:
                continue
            score = distance / max(max_speed_mps * seconds, 1e-6)
            if similarity is not None:
                score += max(0.0, 1.0 - similarity) * 0.35
            current = backfills.get(candidate["track_id"])
            if current is None or score < current["score"]:
                backfills[candidate["track_id"]] = {"source": source, "score": score, "gap": gap, "distance": distance, "similarity": similarity}

    if not backfills:
        return 0

    by_track = {int(segment["track_id"]): segment for segment in segments}
    changed = 0
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        track_id = int(rec["track_id"])
        backfill = backfills.get(track_id)
        if backfill is None:
            continue
        source = backfill["source"]
        rec["team"] = source["team"]
        rec["jersey_number"] = str(source["jersey_number"])
        rec["jersey_identity"] = source["jersey_identity"]
        rec["canonical_track_id"] = source["track_id"]
        rec["identity_source"] = "jersey_backfill_motion"
        rec["identity_backfill"] = {
            "from_track_id": source["track_id"],
            "gap_frames": backfill["gap"],
            "distance_m": round(float(backfill["distance"]), 3),
            "embedding_similarity": round(float(backfill["similarity"]), 4) if backfill["similarity"] is not None else None,
        }
        segment = by_track.get(track_id)
        if segment is not None and segment.get("player_id") is not None:
            rec["canonical_player_id"] = segment["player_id"]
        changed += 1
    return changed


def summarize_record_segments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_track: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("track_id") is None:
            continue
        if rec.get("court_x") is None or rec.get("court_y") is None:
            continue
        by_track.setdefault(int(rec["track_id"]), []).append(rec)

    segments = []
    for track_id, track_records in sorted(by_track.items()):
        track_records = sorted(track_records, key=lambda rec: int(rec["frame_index"]))
        first = track_records[0]
        last = track_records[-1]
        team_counts: dict[str, int] = {}
        identity_counts: dict[str, int] = {}
        number_counts: dict[str, int] = {}
        embeddings = []
        for rec in track_records:
            if rec.get("team"):
                team_counts[str(rec["team"])] = team_counts.get(str(rec["team"]), 0) + 1
            if rec.get("jersey_identity"):
                identity_counts[str(rec["jersey_identity"])] = identity_counts.get(str(rec["jersey_identity"]), 0) + 1
            if rec.get("jersey_number"):
                number_counts[str(rec["jersey_number"])] = number_counts.get(str(rec["jersey_number"]), 0) + 1
            emb = rec.get("appearance_embedding") or rec.get("jersey_embedding")
            if emb:
                embeddings.append(np.asarray(emb, dtype=np.float32))
        identity = max(identity_counts.items(), key=lambda item: item[1])[0] if identity_counts else None
        number = max(number_counts.items(), key=lambda item: item[1])[0] if number_counts else None
        team = max(team_counts.items(), key=lambda item: item[1])[0] if team_counts else None
        embedding = None
        if embeddings:
            embedding = np.median(np.asarray(embeddings, dtype=np.float32), axis=0)
            norm = float(np.linalg.norm(embedding))
            embedding = embedding / norm if norm > 0 else embedding
        segments.append(
            {
                "track_id": track_id,
                "player_id": first.get("player_id"),
                "team": team,
                "jersey_number": number,
                "jersey_identity": identity,
                "first_frame": int(first["frame_index"]),
                "last_frame": int(last["frame_index"]),
                "frames": len(track_records),
                "first_pos": np.asarray([first["court_x"], first["court_y"]], dtype=np.float32),
                "last_pos": np.asarray([last["court_x"], last["court_y"]], dtype=np.float32),
                "embedding": embedding,
            }
        )
    return segments


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return None
    return float(np.dot(a, b) / denom)


def apply_identity_overrides(records: list[dict[str, Any]], overrides_path: Path) -> None:
    data = json.loads(overrides_path.read_text(encoding="utf-8"))
    overrides = data.get("track_overrides", [])
    for rec in records:
        if rec.get("class_name") != "person":
            continue
        frame = int(rec.get("frame_index", -1))
        track_id = rec.get("track_id")
        raw_track_id = rec.get("raw_track_id")
        detector_track_id = rec.get("detector_track_id")
        for override in overrides:
            if not _override_matches(override, frame, track_id, raw_track_id, detector_track_id):
                continue
            team = override.get("team")
            number = override.get("jersey_number")
            identity = override.get("jersey_identity") or (f"{team}_{number}" if team and number else None)
            if team:
                rec["team"] = team
            if number is not None:
                rec["jersey_number"] = str(number)
            if identity:
                rec["jersey_identity"] = identity
            if override.get("canonical_player_id") is not None:
                rec["canonical_player_id"] = override["canonical_player_id"]
            rec["identity_override"] = override.get("reason", True)
            break


def _override_matches(
    override: dict[str, Any],
    frame: int,
    track_id: Any,
    raw_track_id: Any,
    detector_track_id: Any,
) -> bool:
    first_frame = int(override.get("first_frame", -1_000_000))
    last_frame = int(override.get("last_frame", 1_000_000))
    if frame < first_frame or frame > last_frame:
        return False

    ids = {int(v) for v in (track_id, raw_track_id, detector_track_id) if v is not None}
    expected_ids = set()
    if override.get("track_id") is not None:
        expected_ids.add(int(override["track_id"]))
    for key in ("track_ids", "raw_track_ids", "detector_track_ids"):
        expected_ids.update(int(v) for v in override.get(key, []))
    return bool(ids & expected_ids)


def interpolate_jersey_identity_gaps(records: list[dict[str, Any]], fps: float, max_gap_frames: int) -> None:
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if rec.get("class_name") != "person" or not rec.get("jersey_identity"):
            continue
        # Two-digit dorsals are much safer for interpolation in this video.
        if len(str(rec.get("jersey_number", ""))) < 2:
            continue
        by_identity.setdefault(str(rec["jersey_identity"]), []).append(rec)

    synthetic: list[dict[str, Any]] = []
    existing = {(int(rec["frame_index"]), str(rec.get("jersey_identity"))) for rec in records if rec.get("jersey_identity")}
    for identity, identity_records in by_identity.items():
        frame_best: dict[int, dict[str, Any]] = {}
        for rec in identity_records:
            frame = int(rec["frame_index"])
            current = frame_best.get(frame)
            if current is None or float(rec.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
                frame_best[frame] = rec
        ordered = [frame_best[frame] for frame in sorted(frame_best)]
        for prev, nxt in zip(ordered[:-1], ordered[1:]):
            start = int(prev["frame_index"])
            end = int(nxt["frame_index"])
            gap = end - start - 1
            if gap <= 0 or gap > max_gap_frames:
                continue
            if not _can_interpolate_identity_gap(prev, nxt, gap, fps):
                continue
            for frame in range(start + 1, end):
                if (frame, identity) in existing:
                    continue
                alpha = (frame - start) / float(end - start)
                synthetic.append(_interpolate_record(prev, nxt, frame, alpha, fps))
                existing.add((frame, identity))

    records.extend(synthetic)
    records.sort(key=lambda rec: (int(rec.get("frame_index", 0)), int(rec.get("player_id") or 9999), str(rec.get("source", ""))))


def _can_interpolate_identity_gap(prev: dict[str, Any], nxt: dict[str, Any], gap: int, fps: float) -> bool:
    if prev.get("team") != nxt.get("team"):
        return False
    if prev.get("court_x") is not None and nxt.get("court_x") is not None:
        distance = float(
            ((float(prev["court_x"]) - float(nxt["court_x"])) ** 2 + (float(prev["court_y"]) - float(nxt["court_y"])) ** 2)
            ** 0.5
        )
        seconds = max((gap + 1) / max(fps, 1e-6), 1e-6)
        if distance / seconds > 8.5:
            return False
    return True


def _interpolate_record(prev: dict[str, Any], nxt: dict[str, Any], frame: int, alpha: float, fps: float) -> dict[str, Any]:
    rec = dict(prev)
    rec["frame_index"] = frame
    rec["time_s"] = round(frame / fps, 4)
    rec["source"] = "interpolated_jersey_identity"
    rec["track_id"] = None
    rec["raw_track_id"] = None
    rec["confidence"] = round(min(float(prev.get("confidence") or 0.0), float(nxt.get("confidence") or 0.0)) * 0.6, 4)
    rec["bbox"] = _lerp_list(prev.get("bbox"), nxt.get("bbox"), alpha, ndigits=2)
    rec["anchor_px"] = _lerp_list(prev.get("anchor_px"), nxt.get("anchor_px"), alpha, ndigits=2)
    rec["court_x"] = _lerp_value(prev.get("court_x"), nxt.get("court_x"), alpha, ndigits=3)
    rec["court_y"] = _lerp_value(prev.get("court_y"), nxt.get("court_y"), alpha, ndigits=3)
    rec["speed_mps"] = None
    rec["interpolated_identity"] = True
    return rec


def _lerp_list(a: Any, b: Any, alpha: float, ndigits: int) -> list[float] | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return a if isinstance(a, list) else None
    return [round(float(x) * (1.0 - alpha) + float(y) * alpha, ndigits) for x, y in zip(a, b)]


def _lerp_value(a: Any, b: Any, alpha: float, ndigits: int) -> float | None:
    if a is None or b is None:
        return a
    return round(float(a) * (1.0 - alpha) + float(b) * alpha, ndigits)


if __name__ == "__main__":
    main()
