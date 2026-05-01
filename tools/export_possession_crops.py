from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.events import interpolate_ball_gaps
from basketball_cv.possession import best_non_dense_ball
from basketball_cv.possession_balanced import assign_balanced_ball_ownership
from basketball_cv.possession_model import make_player_ball_crop, select_candidates, PossessionModelConfig
from tools.render_tracks import apply_jersey_numbers, apply_identity_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export player-ball candidate crops for training a possession/action classifier."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output-dir", default="datasets/possession_cls/raw")
    parser.add_argument("--jersey-numbers", default=None)
    parser.add_argument("--identity-overrides", default=None)
    parser.add_argument("--sample-step", type=int, default=4, help="Export at most one crop every N frames.")
    parser.add_argument("--max-crops", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--crop-margin", type=float, default=0.35)
    parser.add_argument("--candidate-radius-m", type=float, default=8.0)
    parser.add_argument("--candidate-min-contact", type=float, default=0.05)
    parser.add_argument(
        "--bootstrap-labels",
        action="store_true",
        help="Place crops into weak-label folders from the current rule-based owner. Otherwise writes all crops to unlabeled/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    fps = float(tracks.get("fps", 60.0))
    if args.jersey_numbers:
        apply_jersey_numbers(records, Path(args.jersey_numbers))
    if args.identity_overrides:
        apply_identity_overrides(records, Path(args.identity_overrides))

    interpolate_ball_gaps(records, fps)
    assign_balanced_ball_ownership(records, fps)

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        by_frame.setdefault(int(rec.get("frame_index", 0)), []).append(rec)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    config = PossessionModelConfig(
        crop_margin=args.crop_margin,
        max_candidates=args.max_candidates,
        candidate_radius_m=args.candidate_radius_m,
        candidate_min_contact=args.candidate_min_contact,
    )

    manifest_path = output_dir / "manifest.csv"
    exported = 0
    current_frame = -1
    frame_img = None
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "frame_index",
                "player_id",
                "team",
                "jersey_number",
                "weak_label",
                "ball_owner_player_id",
                "ball_owner_team",
                "ball_owner_reason",
                "ball_owner_confidence",
            ],
        )
        writer.writeheader()

        for frame_index in sorted(by_frame):
            if args.sample_step > 1 and frame_index % args.sample_step != 0:
                continue
            if args.max_crops and exported >= args.max_crops:
                break
            if frame_index != current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_img = cap.read()
                current_frame = frame_index
                if not ok:
                    frame_img = None
            if frame_img is None:
                continue

            frame_records = by_frame[frame_index]
            ball = best_non_dense_ball(frame_records)
            if ball is None:
                continue
            candidates = select_candidates(ball, frame_records, config)
            owner_id = ball.get("ball_owner_player_id")
            owner_team = ball.get("ball_owner_team")
            owner_reason = ball.get("ball_owner_assignment_reason")
            owner_conf = ball.get("ball_owner_confidence")

            for rank, player in enumerate(candidates):
                if args.max_crops and exported >= args.max_crops:
                    break
                crop = make_player_ball_crop(frame_img, player, ball, args.crop_margin)
                if crop is None or crop.size == 0:
                    continue
                player_id = player.get("player_id") or player.get("track_id")
                is_owner = str(player_id) == str(owner_id)
                weak_label = "control" if is_owner else "no_control"
                label_dir = output_dir / (weak_label if args.bootstrap_labels else "unlabeled")
                label_dir.mkdir(parents=True, exist_ok=True)
                filename = f"f{frame_index:06d}_p{player_id}_r{rank}_{weak_label}.jpg"
                out_path = label_dir / filename
                cv2.imwrite(str(out_path), crop)
                writer.writerow(
                    {
                        "file": str(out_path.relative_to(output_dir)),
                        "frame_index": frame_index,
                        "player_id": player_id,
                        "team": player.get("team"),
                        "jersey_number": player.get("jersey_number"),
                        "weak_label": weak_label,
                        "ball_owner_player_id": owner_id,
                        "ball_owner_team": owner_team,
                        "ball_owner_reason": owner_reason,
                        "ball_owner_confidence": owner_conf,
                    }
                )
                exported += 1

    cap.release()
    print(f"Exported {exported} crops to {output_dir}")
    print(f"Manifest: {manifest_path}")
    if not args.bootstrap_labels:
        print("Move crops from unlabeled/ into class folders such as control/, no_control/, dribble/, shooting/, contested/ before training.")


if __name__ == "__main__":
    main()
