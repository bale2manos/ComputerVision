from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.possession_dataset import export_manifest_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export possession datasets from the canonical manifest. "
            "Use --task ball-state for owned / air / loose ball crops, or "
            "--task owner-state for player-plus-ball crops labeled as "
            "control / dribble / shot / contested / no_control."
        )
    )
    parser.add_argument("--manifest", required=True, help="Canonical JSONL manifest created by annotate_possession_dataset.py")
    parser.add_argument("--task", required=True, choices=["ball-state", "owner-state"])
    parser.add_argument("--output-dir", default=None, help="Output dataset root. Defaults depend on --task.")
    parser.add_argument("--val-video-stems", default="", help="Comma-separated video stems forced to validation split.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Used when rows have split_hint=auto and no explicit val stems.")
    parser.add_argument("--max-negatives-per-frame", type=int, default=4, help="Only used for owner-state export.")
    parser.add_argument("--exclude-flags", default="", help="Comma-separated manifest flags to skip during export, e.g. uncertain,occluded.")
    parser.add_argument("--ball-margin", type=float, default=1.2, help="Context margin around ball crops.")
    parser.add_argument("--owner-margin", type=float, default=0.35, help="Context margin around player-plus-ball crops.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.task, args.output_dir)
    val_video_stems = {item.strip() for item in args.val_video_stems.split(",") if item.strip()}
    exclude_flags = {item.strip() for item in args.exclude_flags.split(",") if item.strip()}
    report = export_manifest_dataset(
        Path(args.manifest),
        output_dir,
        task=args.task,
        max_negatives_per_frame=args.max_negatives_per_frame,
        ball_margin=args.ball_margin,
        owner_margin=args.owner_margin,
        val_video_stems=val_video_stems,
        val_ratio=args.val_ratio,
        exclude_flags=exclude_flags,
    )
    print(json.dumps(report, indent=2))


def resolve_output_dir(task: str, value: str | None) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else (ROOT / path).resolve()
    default_name = "datasets/possession_ball_state" if task == "ball-state" else "datasets/possession_owner_state"
    return (ROOT / default_name).resolve()


if __name__ == "__main__":
    main()
