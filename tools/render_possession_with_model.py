from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.court import CourtSpec, load_calibration
from basketball_cv.events import densify_ball_track_for_render, detect_passes, interpolate_ball_gaps
from basketball_cv.possession import build_possession_timeline
from basketball_cv.possession_balanced import assign_balanced_ball_ownership
from basketball_cv.possession_model import PossessionClassifier, PossessionModelConfig
from tools.analyze_video import write_json
from tools.render_possession import render_annotated_video_with_possession
from tools.render_tracks import (
    apply_identity_overrides,
    apply_jersey_numbers,
    backfill_jersey_identities_across_fragments,
    interpolate_jersey_identity_gaps,
    load_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render tracks using balanced possession plus an optional learned possession classifier."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--tracks", required=True, help="tracks.json produced by analyze_video.py.")
    parser.add_argument("--calibration", required=True, help="Court calibration JSON.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--possession-model", default=None, help="Optional YOLO classification .pt for player-ball state.")
    parser.add_argument("--possession-imgsz", type=int, default=224, help="Reserved for future model export/inference tuning.")
    parser.add_argument("--possession-min-confidence", type=float, default=0.58)
    parser.add_argument("--possession-min-margin", type=float, default=0.10)
    parser.add_argument("--possession-max-candidates", type=int, default=6)
    parser.add_argument("--jersey-numbers", default=None)
    parser.add_argument("--identity-overrides", default=None)
    parser.add_argument("--events", default=None)
    parser.add_argument("--output-events", default=None)
    parser.add_argument("--output-possession", default=None)
    parser.add_argument("--no-detect-passes", action="store_true")
    parser.add_argument("--no-dense-ball-track", action="store_true")
    parser.add_argument("--dense-ball-max-gap", type=float, default=3.0)
    parser.add_argument("--no-interpolate-jersey-gaps", action="store_true")
    parser.add_argument("--max-interpolation-gap", type=int, default=75)
    parser.add_argument("--debug-possession", action="store_true")
    parser.add_argument("--no-possession-hud", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    fps = float(tracks.get("fps", 60.0))

    if args.jersey_numbers:
        apply_jersey_numbers(records, Path(args.jersey_numbers))
    if args.identity_overrides:
        apply_identity_overrides(records, Path(args.identity_overrides))
    if args.jersey_numbers or args.identity_overrides:
        backfill_jersey_identities_across_fragments(records, fps)
        if not args.no_interpolate_jersey_gaps:
            interpolate_jersey_identity_gaps(records, fps, args.max_interpolation_gap)

    events = load_events(Path(args.events)) if args.events else []
    interpolate_ball_gaps(records, fps)
    ownership_report = assign_balanced_ball_ownership(records, fps)

    if args.possession_model:
        config = PossessionModelConfig(
            max_candidates=args.possession_max_candidates,
            min_model_confidence=args.possession_min_confidence,
            min_control_margin=args.possession_min_margin,
        )
        classifier = PossessionClassifier(args.possession_model, config=config)
        model_report = classifier.apply_to_records(records, args.video, fps)
        ownership_report["learned_model"] = model_report

    if not args.no_detect_passes:
        recomputed_passes = detect_passes(records, fps)
        non_pass_events = [event for event in events if event.get("type") != "pass"]
        events = sorted(non_pass_events + recomputed_passes, key=lambda event: int(event.get("start_frame", 0)))
        if args.output_events:
            write_json(Path(args.output_events), {"video": args.video, "fps": fps, "events": events})

    timeline = build_possession_timeline(records, fps)
    if args.output_possession:
        write_json(Path(args.output_possession), {"video": args.video, "fps": fps, "summary": ownership_report, "timeline": timeline})

    if not args.no_dense_ball_track:
        frame_count = int(tracks.get("frame_count") or max((int(rec.get("frame_index", 0)) for rec in records), default=0) + 1)
        densify_ball_track_for_render(records, fps, frame_count=frame_count, max_linear_gap_s=args.dense_ball_max_gap)

    calibration = load_calibration(args.calibration)
    court_spec = CourtSpec(**calibration["court"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_annotated_video_with_possession(
        args.video,
        output_path,
        records,
        fps,
        court_spec,
        events=events,
        show_possession_hud=not args.no_possession_hud,
        debug_possession=args.debug_possession,
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
