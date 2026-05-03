from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine EasyOCR and adapted PaddleOCR jersey reports into one render-ready jersey_numbers.json.")
    parser.add_argument("--easyocr", default=None, help="jersey_numbers.json generated with EasyOCR")
    parser.add_argument("--paddle", default=None, help="jersey_numbers.json generated with adapted PaddleOCR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--easyocr-weight", type=float, default=1.0)
    parser.add_argument("--paddle-weight", type=float, default=1.15)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument("--min-frame-votes", type=int, default=2)
    parser.add_argument("--min-margin", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports: list[tuple[str, float, dict[str, Any]]] = []
    if args.easyocr:
        reports.append(("easyocr", args.easyocr_weight, load_json(Path(args.easyocr))))
    if args.paddle:
        reports.append(("paddle", args.paddle_weight, load_json(Path(args.paddle))))
    if not reports:
        raise SystemExit("Pass at least --easyocr or --paddle")

    players = combine_players(reports, args)
    segments = combine_segments(reports, args)
    conflicts = detect_conflicts(players, segments)
    payload = {
        "summary": {
            "ocr_engine": "combined",
            "sources": [name for name, _weight, _report in reports],
            "players_sampled": len(players),
            "players_with_votes": sum(1 for p in players if p.get("vote_count", 0) > 0),
            "players_resolved": sum(1 for p in players if p.get("jersey_number") is not None),
            "identity_segments_resolved": sum(1 for s in segments if s.get("jersey_number") is not None),
            "identity_conflicts": len(conflicts),
        },
        "players": players,
        "identity_segments": segments,
        "identity_conflicts": conflicts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(output.resolve())


def combine_players(reports: list[tuple[str, float, dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[str, float, dict[str, Any]]]] = defaultdict(list)
    for source, weight, report in reports:
        for player in report.get("players", []) or []:
            if player.get("player_id") is not None:
                grouped[int(player["player_id"])].append((source, weight, player))

    players = []
    for player_id, items in sorted(grouped.items()):
        counts, scores, raw_votes = merge_votes(items)
        number = choose_number(counts, scores, args)
        team = majority([item.get("team") for _source, _weight, item in items])
        player = {
            "player_id": player_id,
            "team": team,
            "jersey_number": number,
            "jersey_identity": f"{team}_{number}" if team and number is not None else None,
            "canonical_player_id": player_id if number is not None else None,
            "sample_count": max(int(item.get("sample_count", 0)) for _s, _w, item in items),
            "vote_count": sum(counts.values()),
            "frame_votes": dict(counts),
            "score_by_number": {k: round(float(v), 4) for k, v in sorted(scores.items())},
            "raw_votes": raw_votes[:80],
            "combined_sources": sorted({source for source, _weight, _item in items}),
        }
        firsts = [item.get("first_sample_frame") for _s, _w, item in items if item.get("first_sample_frame") is not None]
        lasts = [item.get("last_sample_frame") for _s, _w, item in items if item.get("last_sample_frame") is not None]
        player["first_sample_frame"] = min(firsts) if firsts else None
        player["last_sample_frame"] = max(lasts) if lasts else None
        players.append(player)
    return players


def combine_segments(reports: list[tuple[str, float, dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[tuple[str, float, dict[str, Any]]]] = defaultdict(list)
    for source, weight, report in reports:
        for seg in report.get("identity_segments", []) or []:
            if seg.get("track_id") is not None and seg.get("team") is not None:
                grouped[(int(seg["track_id"]), str(seg["team"]))].append((source, weight, seg))

    segments = []
    for (track_id, team), items in sorted(grouped.items()):
        counts, scores, raw_votes = merge_votes(items)
        number = choose_number(counts, scores, args)
        frames_first = [item.get("first_vote_frame") for _s, _w, item in items if item.get("first_vote_frame") is not None]
        frames_last = [item.get("last_vote_frame") for _s, _w, item in items if item.get("last_vote_frame") is not None]
        player_ids = sorted({int(pid) for _s, _w, item in items for pid in item.get("player_ids", []) if pid is not None})
        segments.append(
            {
                "track_id": track_id,
                "team": team,
                "player_ids": player_ids,
                "jersey_number": number,
                "jersey_identity": f"{team}_{number}" if number is not None else None,
                "first_vote_frame": min(frames_first) if frames_first else None,
                "last_vote_frame": max(frames_last) if frames_last else None,
                "apply_to_full_track": all(bool(item.get("apply_to_full_track", True)) for _s, _w, item in items),
                "vote_count": sum(counts.values()),
                "frame_votes": dict(counts),
                "score_by_number": {k: round(float(v), 4) for k, v in sorted(scores.items())},
                "raw_votes": raw_votes[:80],
                "combined_sources": sorted({source for source, _weight, _item in items}),
            }
        )
    return segments


def merge_votes(items: list[tuple[str, float, dict[str, Any]]]) -> tuple[Counter[str], dict[str, float], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    scores: dict[str, float] = defaultdict(float)
    raw_votes: list[dict[str, Any]] = []
    for source, weight, item in items:
        for number, count in (item.get("frame_votes") or {}).items():
            counts[str(number)] += int(count)
        for number, score in (item.get("score_by_number") or {}).items():
            scores[str(number)] += float(score) * weight
        for vote in item.get("raw_votes", []) or []:
            vote = dict(vote)
            vote.setdefault("ocr_source", source)
            raw_votes.append(vote)
    return counts, scores, raw_votes


def choose_number(counts: Counter[str], scores: dict[str, float], args: argparse.Namespace) -> str | None:
    candidates = [(num, float(score), int(counts.get(num, 0))) for num, score in scores.items()]
    candidates = [(num, score, count) for num, score, count in candidates if score >= args.min_score and count >= args.min_frame_votes]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], x[2], len(x[0])), reverse=True)
    best_num, best_score, _best_count = candidates[0]
    second_score = candidates[1][1] if len(candidates) > 1 else 0.0
    if second_score > 0 and best_score < second_score * (1.0 + args.min_margin):
        return None
    return str(best_num)


def detect_conflicts(players: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    by_identity: dict[str, set[int]] = defaultdict(set)
    for player in players:
        if player.get("jersey_identity"):
            by_identity[str(player["jersey_identity"])].add(int(player["player_id"]))
    for identity, ids in sorted(by_identity.items()):
        if len(ids) > 1:
            conflicts.append({"type": "duplicate_player_jersey_identity", "jersey_identity": identity, "player_ids": sorted(ids)})
    by_track: dict[int, set[str]] = defaultdict(set)
    for segment in segments:
        if segment.get("jersey_identity"):
            by_track[int(segment["track_id"])].add(str(segment["jersey_identity"]))
    for track_id, identities in sorted(by_track.items()):
        if len(identities) > 1:
            conflicts.append({"type": "track_contains_multiple_jersey_identities", "track_id": track_id, "jersey_identities": sorted(identities)})
    return conflicts


def majority(values: list[Any]) -> Any:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
