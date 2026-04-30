from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DIGIT_RE = re.compile(r"\d{1,2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract jersey-number OCR votes from tracked basketball players.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--tracks", required=True, help="tracks.json produced by tools/analyze_video.py.")
    parser.add_argument("--player-summary", default=None, help="Optional player_summary.json to enrich with OCR numbers.")
    parser.add_argument("--output-dir", default="runs/jersey_ocr", help="Output directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="EasyOCR device preference.")
    parser.add_argument("--languages", nargs="+", default=["en"], help="EasyOCR language list.")
    parser.add_argument("--max-crops-per-player", type=int, default=40, help="Max OCR crops sampled for each player.")
    parser.add_argument("--sample-step", type=int, default=6, help="Minimum frame gap between crops for the same player.")
    parser.add_argument("--min-box-height", type=float, default=115.0, help="Ignore smaller player boxes.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Ignore lower-confidence player detections.")
    parser.add_argument("--min-ocr-confidence", type=float, default=0.18, help="Minimum EasyOCR confidence for a digit vote.")
    parser.add_argument("--save-crops", action="store_true", help="Save sampled and preprocessed crops for review.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    crop_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    samples = select_samples(records, args.max_crops_per_player, args.sample_step, args.min_box_height, args.min_confidence)

    reader = build_reader(args.languages, args.device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    player_votes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for player_id, player_samples in sorted(samples.items()):
        for sample_index, rec in enumerate(player_samples):
            crop = read_jersey_crop(cap, int(rec["frame_index"]), rec["bbox"])
            if crop is None:
                continue
            variants = preprocess_crop_variants(crop)
            if args.save_crops:
                player_crop_dir = crop_dir / f"player_{player_id:02d}"
                player_crop_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(player_crop_dir / f"f{int(rec['frame_index']):04d}_{sample_index:02d}_raw.jpg"), crop)
                for variant_name, variant in variants:
                    cv2.imwrite(str(player_crop_dir / f"f{int(rec['frame_index']):04d}_{sample_index:02d}_{variant_name}.jpg"), variant)
            votes = read_digits(reader, variants, args.min_ocr_confidence)
            for vote in votes:
                vote.update(
                    {
                        "player_id": player_id,
                        "frame_index": int(rec["frame_index"]),
                        "track_id": rec.get("track_id"),
                        "raw_track_id": rec.get("raw_track_id"),
                        "detector_track_id": rec.get("detector_track_id"),
                        "team": rec.get("team"),
                        "bbox": rec.get("bbox"),
                    }
                )
                player_votes[player_id].append(vote)

    cap.release()

    report = build_report(player_votes, samples)
    write_json(output_dir / "jersey_numbers.json", report)
    if args.player_summary:
        enrich_player_summary(Path(args.player_summary), output_dir / "player_summary_with_numbers.json", report)

    print(json.dumps(report["summary"], indent=2))


def build_reader(languages: list[str], device: str) -> Any:
    try:
        import easyocr
        import torch
    except ImportError as exc:
        raise SystemExit("EasyOCR is not installed. Run: python -m pip install easyocr") from exc

    gpu = device == "cuda" and torch.cuda.is_available()
    return easyocr.Reader(languages, gpu=gpu, verbose=False)


def select_samples(
    records: list[dict[str, Any]],
    max_crops_per_player: int,
    sample_step: int,
    min_box_height: float,
    min_confidence: float,
) -> dict[int, list[dict[str, Any]]]:
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("class_name") != "person" or rec.get("player_id") is None:
            continue
        if rec.get("confidence", 0.0) < min_confidence:
            continue
        if rec.get("bottom_truncated"):
            continue
        if rec.get("on_court") is False:
            continue
        x1, y1, x2, y2 = [float(v) for v in rec.get("bbox", [0, 0, 0, 0])]
        box_h = y2 - y1
        box_w = x2 - x1
        if box_h < min_box_height or box_w < 28:
            continue
        by_player[int(rec["player_id"])].append(rec)

    selected: dict[int, list[dict[str, Any]]] = {}
    for player_id, player_records in by_player.items():
        player_records = sorted(
            player_records,
            key=lambda r: (float(r.get("bbox_area") or 0.0), float(r.get("confidence") or 0.0)),
            reverse=True,
        )
        chosen = []
        chosen_frames: list[int] = []
        for rec in player_records:
            frame = int(rec["frame_index"])
            if any(abs(frame - prev) < sample_step for prev in chosen_frames):
                continue
            chosen.append(rec)
            chosen_frames.append(frame)
            if len(chosen) >= max_crops_per_player:
                break
        selected[player_id] = sorted(chosen, key=lambda r: int(r["frame_index"]))
    return selected


def read_jersey_crop(cap: cv2.VideoCapture, frame_index: int, bbox: list[float]) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return None

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    cx1 = x1 + int(0.14 * box_w)
    cx2 = x2 - int(0.14 * box_w)
    cy1 = y1 + int(0.10 * box_h)
    cy2 = y1 + int(0.66 * box_h)
    crop = frame[max(0, cy1) : min(h, cy2), max(0, cx1) : min(w, cx2)]
    return crop if crop.size else None


def preprocess_crop_variants(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    scale = 4 if max(crop.shape[:2]) < 180 else 3
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(gray)
    sharp = cv2.addWeighted(clahe, 1.6, cv2.GaussianBlur(clahe, (0, 0), 1.2), -0.6, 0)
    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv_otsu = 255 - otsu
    adaptive = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    return [
        ("rgb", enlarged),
        ("gray", cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)),
        ("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
        ("inv", cv2.cvtColor(inv_otsu, cv2.COLOR_GRAY2BGR)),
        ("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)),
    ]


def read_digits(reader: Any, variants: list[tuple[str, np.ndarray]], min_confidence: float) -> list[dict[str, Any]]:
    votes = []
    for variant_name, image in variants:
        try:
            results = reader.readtext(
                image,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
                text_threshold=0.25,
                low_text=0.15,
                link_threshold=0.2,
                canvas_size=640,
                mag_ratio=1.0,
            )
        except Exception:
            continue
        for _box, text, confidence in results:
            for match in DIGIT_RE.findall(str(text)):
                number = normalize_number(match)
                if number is None or confidence < min_confidence:
                    continue
                votes.append(
                    {
                        "number": number,
                        "text": str(text),
                        "ocr_confidence": round(float(confidence), 4),
                        "variant": variant_name,
                    }
                )
    return votes


def normalize_number(text: str) -> str | None:
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits or len(digits) > 2:
        return None
    if len(digits) == 2 and digits.startswith("0"):
        digits = digits[1:]
    value = int(digits)
    if value < 0 or value > 99:
        return None
    return str(value)


def build_report(player_votes: dict[int, list[dict[str, Any]]], samples: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    players = []
    for player_id in sorted(samples):
        votes = player_votes.get(player_id, [])
        counts, scores = aggregate_frame_votes(votes)

        best_number = choose_jersey_number(counts, scores)
        sample_frames = [int(rec["frame_index"]) for rec in samples[player_id]]

        sample_teams = Counter(str(rec.get("team")) for rec in samples[player_id] if rec.get("team"))
        players.append(
            {
                "player_id": player_id,
                "team": sample_teams.most_common(1)[0][0] if sample_teams else None,
                "jersey_number": best_number,
                "first_sample_frame": min(sample_frames) if sample_frames else None,
                "last_sample_frame": max(sample_frames) if sample_frames else None,
                "sample_count": len(samples[player_id]),
                "vote_count": len(votes),
                "frame_votes": dict(counts),
                "score_by_number": {number: round(float(score), 4) for number, score in sorted(scores.items())},
                "raw_votes": votes[:40],
            }
        )

    identity_segments = build_identity_segments(player_votes)
    apply_canonical_jersey_identities(players)
    apply_canonical_segment_identities(identity_segments, players)
    identity_conflicts = detect_identity_conflicts(players, identity_segments)
    return {
        "summary": {
            "players_sampled": len(samples),
            "players_with_votes": sum(1 for player_id in samples if player_votes.get(player_id)),
            "players_resolved": sum(1 for player in players if player.get("jersey_number") is not None),
            "canonical_jersey_identities": len(
                {
                    player.get("jersey_identity")
                    for player in players
                    if player.get("jersey_identity") is not None
                }
            ),
            "identity_segments_resolved": sum(1 for segment in identity_segments if segment.get("jersey_number") is not None),
            "identity_conflicts": len(identity_conflicts),
        },
        "players": players,
        "identity_segments": identity_segments,
        "identity_conflicts": identity_conflicts,
    }


def aggregate_frame_votes(votes: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, float]]:
    per_frame: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for vote in votes:
        number = vote["number"]
        frame = int(vote["frame_index"])
        per_frame[frame][number] = max(per_frame[frame][number], float(vote["ocr_confidence"]))

    counts: Counter[str] = Counter()
    scores: dict[str, float] = defaultdict(float)
    for frame_scores in per_frame.values():
        for number, score in frame_scores.items():
            counts[number] += 1
            scores[number] += score
    return counts, scores


def choose_jersey_number(counts: Counter[str], scores: dict[str, float]) -> str | None:
    if not scores:
        return None

    best_any, best_any_score = max(scores.items(), key=lambda item: (item[1], counts[item[0]]))
    two_digit = {
        number: score
        for number, score in scores.items()
        if len(number) == 2 and counts[number] >= 2 and score >= 2.0
    }
    if two_digit:
        best_two, best_two_score = max(two_digit.items(), key=lambda item: (item[1], counts[item[0]]))
        # Back numbers are larger and more reliable than isolated single digits
        # produced by folds, logos, or shorts. Do not require them to beat every
        # single digit, only to have repeated support.
        if best_two_score >= best_any_score * 0.28 or counts[best_two] >= 5:
            return best_two

    second_score = max([score for number, score in scores.items() if number != best_any], default=0.0)
    if len(best_any) == 1 and counts[best_any] >= 4 and best_any_score >= 2.4 and best_any_score >= second_score * 1.35:
        return best_any
    if len(best_any) == 2 and counts[best_any] >= 2 and best_any_score >= 1.0 and best_any_score >= second_score * 1.15:
        return best_any
    return None


def apply_canonical_jersey_identities(players: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for player in players:
        number = player.get("jersey_number")
        team = player.get("team")
        if number is None or team is None:
            continue
        grouped[(team, str(number))].append(player)

    for (team, number), group in grouped.items():
        canonical = max(
            group,
            key=lambda player: (
                float(player.get("score_by_number", {}).get(number, 0.0)),
                int(player.get("sample_count", 0)),
            ),
        )
        merged_ids = sorted(int(player["player_id"]) for player in group)
        identity = f"{team}_{number}"
        for player in group:
            if len(number) == 1 and player["player_id"] != canonical["player_id"]:
                player["suppressed_jersey_number"] = player.get("jersey_number")
                player["jersey_number"] = None
                player["jersey_suppression_reason"] = "duplicate_single_digit_conflict"
                continue
            player["jersey_identity"] = identity
            player["canonical_player_id"] = canonical["player_id"]
            player["same_jersey_player_ids"] = merged_ids


def build_identity_segments(player_votes: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for player_id, votes in player_votes.items():
        for vote in votes:
            track_id = vote.get("track_id")
            team = vote.get("team")
            if track_id is None or team is None:
                continue
            vote = dict(vote)
            vote["player_id"] = player_id
            grouped[(int(track_id), str(team))].append(vote)

    segments = []
    for (track_id, team), votes in sorted(grouped.items()):
        counts, scores = aggregate_frame_votes(votes)
        best_number = choose_jersey_number(counts, scores)
        strong_numbers = strong_segment_numbers(counts, scores)
        if best_number is not None and best_number not in strong_numbers:
            strong_numbers.insert(0, best_number)

        if len(strong_numbers) <= 1:
            segments.append(make_identity_segment(track_id, team, votes, best_number, apply_to_full_track=True))
            continue

        for number in strong_numbers:
            number_votes = [vote for vote in votes if vote.get("number") == number]
            if not number_votes:
                continue
            segment = make_identity_segment(track_id, team, number_votes, number, apply_to_full_track=False)
            segment["track_conflicting_numbers"] = strong_numbers
            segments.append(segment)
    return segments


def strong_segment_numbers(counts: Counter[str], scores: dict[str, float]) -> list[str]:
    two_digit = {
        number: score
        for number, score in scores.items()
        if len(number) == 2 and counts[number] >= 2 and score >= 1.0
    }
    if two_digit:
        best_score = max(two_digit.values())
        return [
            number
            for number, score in sorted(two_digit.items(), key=lambda item: item[1], reverse=True)
            if score >= max(2.0, best_score * 0.45)
        ]

    one_digit = {
        number: score
        for number, score in scores.items()
        if len(number) == 1 and counts[number] >= 6 and score >= 3.0
    }
    if not one_digit:
        return []
    best_score = max(one_digit.values())
    return [
        number
        for number, score in sorted(one_digit.items(), key=lambda item: item[1], reverse=True)
        if score >= best_score * 0.55
    ]


def make_identity_segment(
    track_id: int,
    team: str,
    votes: list[dict[str, Any]],
    jersey_number: str | None,
    apply_to_full_track: bool,
) -> dict[str, Any]:
    counts, scores = aggregate_frame_votes(votes)
    frames = [int(vote["frame_index"]) for vote in votes]
    player_ids = sorted({int(vote["player_id"]) for vote in votes})
    return {
        "track_id": track_id,
        "team": team,
        "player_ids": player_ids,
        "jersey_number": jersey_number,
        "jersey_identity": f"{team}_{jersey_number}" if jersey_number is not None else None,
        "first_vote_frame": min(frames) if frames else None,
        "last_vote_frame": max(frames) if frames else None,
        "apply_to_full_track": apply_to_full_track,
        "vote_count": len(votes),
        "frame_votes": dict(counts),
        "score_by_number": {number: round(float(score), 4) for number, score in sorted(scores.items())},
        "raw_votes": votes[:30],
    }


def apply_canonical_segment_identities(segments: list[dict[str, Any]], players: list[dict[str, Any]]) -> None:
    player_number_by_id = {int(player["player_id"]): player.get("jersey_number") for player in players}
    canonical_player_by_identity = {
        str(player["jersey_identity"]): int(player.get("canonical_player_id", player["player_id"]))
        for player in players
        if player.get("jersey_identity") and player.get("jersey_number") is not None
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        number = segment.get("jersey_number")
        team = segment.get("team")
        if number is None or team is None:
            continue
        grouped[(str(team), str(number))].append(segment)

    for (team, number), group in grouped.items():
        canonical = max(
            group,
            key=lambda segment: (
                float(segment.get("score_by_number", {}).get(number, 0.0)),
                int(segment.get("vote_count", 0)),
            ),
        )
        merged_track_ids = sorted(int(segment["track_id"]) for segment in group)
        identity = f"{team}_{number}"
        for segment in group:
            if len(number) == 1 and segment["track_id"] != canonical["track_id"]:
                segment["suppressed_jersey_number"] = segment.get("jersey_number")
                segment["jersey_number"] = None
                segment["jersey_identity"] = None
                segment["jersey_suppression_reason"] = "duplicate_single_digit_segment_conflict"
                continue
            canonical_player_id = canonical_player_by_identity.get(identity)
            if len(number) == 1 and canonical_player_id is not None and canonical_player_id not in segment.get("player_ids", []):
                segment["suppressed_jersey_number"] = segment.get("jersey_number")
                segment["jersey_number"] = None
                segment["jersey_identity"] = None
                segment["jersey_suppression_reason"] = "single_digit_belongs_to_other_canonical_player"
                continue
            player_ids = [int(player_id) for player_id in segment.get("player_ids", [])]
            parent_numbers = [player_number_by_id.get(player_id) for player_id in player_ids]
            score = float(segment.get("score_by_number", {}).get(number, 0.0))
            frame_votes = int(segment.get("frame_votes", {}).get(number, 0))
            if len(number) == 2 and parent_numbers and all(parent_number is None for parent_number in parent_numbers):
                if score < 4.5 or frame_votes < 6:
                    segment["suppressed_jersey_number"] = segment.get("jersey_number")
                    segment["jersey_number"] = None
                    segment["jersey_identity"] = None
                    segment["jersey_suppression_reason"] = "weak_two_digit_segment_on_unresolved_player"
                    continue
            segment["jersey_identity"] = identity
            segment["canonical_track_id"] = canonical["track_id"]
            segment["same_jersey_track_ids"] = merged_track_ids


def detect_identity_conflicts(players: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    by_player: dict[int, set[str]] = defaultdict(set)
    by_track: dict[int, set[str]] = defaultdict(set)
    for segment in segments:
        identity = segment.get("jersey_identity")
        if not identity:
            continue
        if segment.get("track_id") is not None:
            by_track[int(segment["track_id"])].add(str(identity))
        for player_id in segment.get("player_ids", []):
            by_player[int(player_id)].add(str(identity))
    for track_id, identities in sorted(by_track.items()):
        if len(identities) > 1:
            conflicts.append(
                {
                    "type": "track_contains_multiple_jersey_identities",
                    "track_id": track_id,
                    "jersey_identities": sorted(identities),
                    "action": "render_only_bounded_ocr_segments_or_split_track",
                }
            )
    for player_id, identities in sorted(by_player.items()):
        if len(identities) > 1:
            conflicts.append(
                {
                    "type": "player_contains_multiple_jersey_identities",
                    "player_id": player_id,
                    "jersey_identities": sorted(identities),
                    "action": "split_or_review_player_segments",
                }
            )

    player_by_id = {int(player["player_id"]): player for player in players}
    for player_id, player in player_by_id.items():
        player_identity = player.get("jersey_identity")
        segment_identities = by_player.get(player_id, set())
        if player_identity and segment_identities and player_identity not in segment_identities:
            conflicts.append(
                {
                    "type": "player_ocr_disagrees_with_track_segment_ocr",
                    "player_id": player_id,
                    "player_jersey_identity": player_identity,
                    "segment_jersey_identities": sorted(segment_identities),
                    "action": "prefer_segment_identity_for_rendering",
                }
            )
    return conflicts


def enrich_player_summary(input_path: Path, output_path: Path, number_report: dict[str, Any]) -> None:
    summary = json.loads(input_path.read_text(encoding="utf-8"))
    numbers = {int(player["player_id"]): player for player in number_report.get("players", [])}
    for player in summary.get("players", []):
        ocr = numbers.get(int(player["player_id"]))
        if not ocr:
            continue
        player["jersey_number"] = ocr.get("jersey_number")
        player["jersey_identity"] = ocr.get("jersey_identity")
        player["canonical_player_id"] = ocr.get("canonical_player_id")
        player["same_jersey_player_ids"] = ocr.get("same_jersey_player_ids", [])
        player["jersey_number_frame_votes"] = ocr.get("frame_votes", {})
        player["jersey_number_scores"] = ocr.get("score_by_number", {})
    write_json(output_path, summary)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
