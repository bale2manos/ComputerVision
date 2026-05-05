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

from basketball_cv.jersey_identity import JerseyIdentityConfig, resolve_player_identity


DIGIT_RE = re.compile(r"\d{1,2}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract jersey-number OCR votes from tracked basketball players.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--tracks", required=True, help="tracks.json produced by tools/analyze_video.py.")
    parser.add_argument("--player-summary", default=None, help="Optional player_summary.json to enrich with OCR numbers.")
    parser.add_argument("--output-dir", default="runs/jersey_ocr", help="Output directory.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="EasyOCR device preference.")
    parser.add_argument("--languages", nargs="+", default=["en"], help="EasyOCR language list.")
    parser.add_argument("--ocr-engine", choices=["easyocr", "paddle", "both"], default="easyocr")
    parser.add_argument("--paddle-rec-model-dir", default=None, help="Exported PaddleOCR recognition inference model directory.")
    parser.add_argument("--paddle-rec-char-dict", default=None, help="PaddleOCR character dictionary, e.g. digit_dict.txt.")
    parser.add_argument("--paddle-use-gpu", action="store_true", help="Use PaddleOCR GPU inference if available.")
    parser.add_argument("--paddle-min-confidence", type=float, default=0.20)
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

    readers = build_readers(args)
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
            votes = read_digits(readers, variants, args)
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
    report.setdefault("summary", {})["ocr_engine"] = args.ocr_engine
    report["summary"]["ocr_sources"] = list(readers.keys())
    report["summary"]["paddle_model_dir"] = args.paddle_rec_model_dir
    write_json(output_dir / "jersey_numbers.json", report)
    if args.player_summary:
        enrich_player_summary(Path(args.player_summary), output_dir / "player_summary_with_numbers.json", report)

    print(json.dumps(report["summary"], indent=2))


def build_readers(args: argparse.Namespace) -> dict[str, Any]:
    readers: dict[str, Any] = {}
    if args.ocr_engine in {"easyocr", "both"}:
        readers["easyocr"] = build_easyocr_reader(args.languages, args.device)
    if args.ocr_engine in {"paddle", "both"}:
        readers["paddle"] = build_paddle_reader(args)
    if not readers:
        raise SystemExit("No OCR reader configured.")
    return readers


def build_easyocr_reader(languages: list[str], device: str) -> Any:
    try:
        import easyocr
        import torch
    except ImportError as exc:
        raise SystemExit("EasyOCR is not installed. Run: python -m pip install easyocr") from exc

    gpu = device == "cuda" and torch.cuda.is_available()
    return easyocr.Reader(languages, gpu=gpu, verbose=False)


def build_paddle_reader(args: argparse.Namespace) -> Any:
    model_dir = Path(args.paddle_rec_model_dir) if args.paddle_rec_model_dir else None
    char_dict = Path(args.paddle_rec_char_dict) if args.paddle_rec_char_dict else None
    if model_dir is None or not model_dir.exists():
        raise SystemExit("--paddle-rec-model-dir is required for --ocr-engine paddle/both and must exist.")
    if char_dict is None or not char_dict.exists():
        raise SystemExit("--paddle-rec-char-dict is required for --ocr-engine paddle/both and must exist.")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit("PaddleOCR is not installed in this Python environment.") from exc

    common = {
        "use_angle_cls": False,
        "lang": "en",
        "show_log": False,
        "use_gpu": bool(args.paddle_use_gpu),
    }
    attempts = [
        {
            **common,
            "det": False,
            "rec": True,
            "cls": False,
            "rec_model_dir": str(model_dir),
            "rec_char_dict_path": str(char_dict),
        },
        {
            **common,
            "text_recognition_model_dir": str(model_dir),
            "text_rec_char_dict_path": str(char_dict),
        },
        {
            "rec_model_dir": str(model_dir),
            "rec_char_dict_path": str(char_dict),
            "use_gpu": bool(args.paddle_use_gpu),
        },
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:  # API changed between PaddleOCR versions.
            last_exc = exc
    raise SystemExit(f"Could not initialize PaddleOCR with adapted model: {last_exc}")


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
        if rec.get("is_estimated"):
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


def read_digits(readers: dict[str, Any], variants: list[tuple[str, np.ndarray]], args: argparse.Namespace) -> list[dict[str, Any]]:
    votes = []
    if "easyocr" in readers:
        votes.extend(read_digits_easyocr(readers["easyocr"], variants, args.min_ocr_confidence))
    if "paddle" in readers:
        votes.extend(read_digits_paddle(readers["paddle"], variants, args.paddle_min_confidence))
    return votes


def read_digits_easyocr(reader: Any, variants: list[tuple[str, np.ndarray]], min_confidence: float) -> list[dict[str, Any]]:
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
                        "ocr_source": "easyocr",
                    }
                )
    return votes


def read_digits_paddle(reader: Any, variants: list[tuple[str, np.ndarray]], min_confidence: float) -> list[dict[str, Any]]:
    votes = []
    for variant_name, image in variants:
        try:
            try:
                result = reader.ocr(image, det=False, cls=False)
            except TypeError:
                result = reader.ocr(image, det=False)
        except Exception:
            continue
        for text, confidence in extract_text_conf_pairs(result):
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
                        "ocr_source": "paddle",
                    }
                )
    return votes


def extract_text_conf_pairs(result: Any) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []

    def walk(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, dict):
            text = obj.get("text") or obj.get("rec_text") or obj.get("label")
            conf = obj.get("confidence") or obj.get("score") or obj.get("rec_score")
            if text is not None and conf is not None:
                try:
                    pairs.append((str(text), float(conf)))
                except Exception:
                    pass
            for value in obj.values():
                if isinstance(value, (list, tuple, dict)):
                    walk(value)
            return
        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and isinstance(obj[0], str) and isinstance(obj[1], (float, int, np.floating)):
                pairs.append((str(obj[0]), float(obj[1])))
                return
            if len(obj) >= 2 and isinstance(obj[1], (list, tuple)) and len(obj[1]) >= 2 and isinstance(obj[1][0], str):
                try:
                    pairs.append((str(obj[1][0]), float(obj[1][1])))
                    return
                except Exception:
                    pass
            for item in obj:
                walk(item)

    walk(result)
    return pairs


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
    identity_segments = []
    resolver_config = JerseyIdentityConfig()
    for player_id in sorted(samples):
        votes = player_votes.get(player_id, [])
        sample_frames = [int(rec["frame_index"]) for rec in samples[player_id]]
        sample_teams = Counter(str(rec.get("team")) for rec in samples[player_id] if rec.get("team"))
        team = sample_teams.most_common(1)[0][0] if sample_teams else None
        resolved = resolve_player_identity(player_id=player_id, team=team, votes=votes, config=resolver_config)
        players.append(
            {
                "player_id": player_id,
                "team": team,
                "jersey_number": resolved["canonical_jersey_number"],
                "canonical_jersey_number": resolved["canonical_jersey_number"],
                "display_jersey_number": resolved["display_jersey_number"],
                "jersey_locked": resolved["jersey_locked"],
                "first_sample_frame": min(sample_frames) if sample_frames else None,
                "last_sample_frame": max(sample_frames) if sample_frames else None,
                "sample_count": len(samples[player_id]),
                "vote_count": len(votes),
                "frame_votes": resolved["frame_votes"],
                "score_by_number": resolved["score_by_number"],
                "raw_votes": votes[:40],
            }
        )
        identity_segments.extend(resolved["segments"])

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
