from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JerseyIdentityConfig:
    min_global_frames: int = 2
    min_global_score: float = 1.8
    min_global_margin: float = 1.05
    min_segment_frames: int = 2
    min_segment_score: float = 1.7
    min_segment_margin: float = 1.05


def resolve_player_identity(
    player_id: int,
    team: str | None,
    votes: list[dict[str, Any]],
    config: JerseyIdentityConfig,
) -> dict[str, Any]:
    counts, scores = aggregate_frame_votes(votes)
    canonical = choose_canonical_number(counts, scores, config)
    segments = build_track_segments(player_id, votes, canonical, config)
    return {
        "player_id": player_id,
        "team": team,
        "canonical_jersey_number": canonical,
        "display_jersey_number": canonical if canonical is not None else f"P{player_id}",
        "jersey_locked": canonical is not None,
        "frame_votes": dict(counts),
        "score_by_number": {number: round(float(score), 4) for number, score in sorted(scores.items())},
        "segments": segments,
    }


def aggregate_frame_votes(votes: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, float]]:
    per_frame: dict[int, dict[str, float]] = defaultdict(dict)
    for vote in votes:
        frame = int(vote["frame_index"])
        number = str(vote["number"])
        score = float(vote.get("ocr_confidence") or 0.0)
        if str(vote.get("variant") or "") in {"rgb", "gray"}:
            score += 0.05
        if str(vote.get("ocr_source") or "") == "paddle":
            score += 0.04
        current = per_frame[frame].get(number)
        if current is None or score > current:
            per_frame[frame][number] = score

    counts: Counter[str] = Counter()
    scores: dict[str, float] = defaultdict(float)
    for frame_scores in per_frame.values():
        for number, score in frame_scores.items():
            counts[number] += 1
            scores[number] += score
    return counts, scores


def choose_canonical_number(counts: Counter[str], scores: dict[str, float], config: JerseyIdentityConfig) -> str | None:
    ranked = rank_numbers(counts, scores)
    if not ranked:
        return None
    best_number = ranked[0]
    best_score = scores[best_number]
    second_score = scores[ranked[1]] if len(ranked) > 1 else 0.0
    if counts[best_number] < config.min_global_frames:
        return None
    if best_score < config.min_global_score:
        return None
    if second_score and best_score < second_score * config.min_global_margin:
        return None
    return best_number


def build_track_segments(
    player_id: int,
    votes: list[dict[str, Any]],
    canonical: str | None,
    config: JerseyIdentityConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        track_id = vote.get("track_id")
        team = vote.get("team")
        if track_id is None or team is None:
            continue
        grouped[(int(track_id), str(team))].append(vote)

    segments: list[dict[str, Any]] = []
    for (track_id, team), track_votes in sorted(grouped.items()):
        counts, scores = aggregate_frame_votes(track_votes)
        ranked = rank_numbers(counts, scores)
        candidate_numbers = [
            number
            for number in ranked
            if counts[number] >= config.min_segment_frames and scores[number] >= config.min_segment_score
        ]
        if candidate_numbers:
            best_number = candidate_numbers[0]
            second_score = scores[candidate_numbers[1]] if len(candidate_numbers) > 1 else 0.0
            if second_score and scores[best_number] < second_score * config.min_segment_margin:
                candidate_numbers = candidate_numbers[:2]
            else:
                candidate_numbers = candidate_numbers[:1] + [
                    number
                    for number in candidate_numbers[1:]
                    if scores[number] >= scores[best_number] * 0.92
                ]
        if not candidate_numbers:
            segments.append(
                make_identity_segment(
                    player_id=player_id,
                    track_id=track_id,
                    team=team,
                    votes=track_votes,
                    jersey_number=None,
                    apply_to_full_track=True,
                    canonical=canonical,
                )
            )
            continue

        if len(candidate_numbers) == 1:
            segments.append(
                make_identity_segment(
                    player_id=player_id,
                    track_id=track_id,
                    team=team,
                    votes=track_votes,
                    jersey_number=candidate_numbers[0],
                    apply_to_full_track=True,
                    canonical=canonical,
                )
            )
            continue

        for number in candidate_numbers:
            number_votes = [vote for vote in track_votes if str(vote.get("number")) == number]
            if not number_votes:
                continue
            segment = make_identity_segment(
                player_id=player_id,
                track_id=track_id,
                team=team,
                votes=number_votes,
                jersey_number=number,
                apply_to_full_track=False,
                canonical=canonical,
            )
            segment["track_conflicting_numbers"] = candidate_numbers
            segments.append(segment)
    return segments


def make_identity_segment(
    player_id: int,
    track_id: int,
    team: str,
    votes: list[dict[str, Any]],
    jersey_number: str | None,
    apply_to_full_track: bool,
    canonical: str | None,
) -> dict[str, Any]:
    counts, scores = aggregate_frame_votes(votes)
    frames = [int(vote["frame_index"]) for vote in votes]
    return {
        "track_id": track_id,
        "team": team,
        "player_ids": [int(player_id)],
        "jersey_number": jersey_number,
        "jersey_identity": f"{team}_{jersey_number}" if jersey_number is not None else None,
        "backfill_allowed": canonical is not None and jersey_number in {None, canonical},
        "first_vote_frame": min(frames) if frames else None,
        "last_vote_frame": max(frames) if frames else None,
        "apply_to_full_track": apply_to_full_track,
        "vote_count": len(votes),
        "frame_votes": dict(counts),
        "score_by_number": {number: round(float(score), 4) for number, score in sorted(scores.items())},
        "raw_votes": votes[:30],
    }


def rank_numbers(counts: Counter[str], scores: dict[str, float]) -> list[str]:
    return [
        number
        for number, _ in sorted(
            scores.items(),
            key=lambda item: (
                item[1],
                counts[item[0]],
                item[0].isdigit() and len(item[0]) == 2,
                -int(item[0]) if item[0].isdigit() else 0,
            ),
            reverse=True,
        )
    ]
