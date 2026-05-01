from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pipeline and render with optional learned possession models."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-root", default="runs/pipeline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-frame", type=int, default=120)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--tracker", default="trackers/bytetrack_basketball.yaml")
    parser.add_argument("--ball-model", default="auto")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--ball-imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--ball-conf", type=float, default=0.15)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--ocr-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--identity-overrides", default=None)
    parser.add_argument("--possession-detector-model", default=None)
    parser.add_argument("--possession-detector-conf", type=float, default=0.35)
    parser.add_argument("--possession-detector-imgsz", type=int, default=960)
    parser.add_argument("--possession-detector-device", default=None)
    parser.add_argument("--possession-model", default=None)
    parser.add_argument("--possession-min-confidence", type=float, default=0.58)
    parser.add_argument("--possession-min-margin", type=float, default=0.10)
    parser.add_argument("--debug-possession", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = resolve_path(args.video)
    run_name = args.run_name or f"{video.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = resolve_path(args.output_root) / run_name

    base_cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_full_pipeline_enhanced.py"),
        "--video",
        str(video),
        "--output-root",
        str(resolve_path(args.output_root)),
        "--run-name",
        run_name,
        "--model",
        args.model,
        "--tracker",
        args.tracker,
        "--ball-model",
        args.ball_model,
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--ball-imgsz",
        str(args.ball_imgsz),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--ball-conf",
        str(args.ball_conf),
        "--ocr-device",
        args.ocr_device,
        "--skip-render",
    ]
    if args.calibration:
        base_cmd += ["--calibration", str(resolve_path(args.calibration))]
    if args.reuse_calibration:
        base_cmd.append("--reuse-calibration")
    if args.skip_calibration:
        base_cmd.append("--skip-calibration")
    if args.calibration_frame:
        base_cmd += ["--calibration-frame", str(args.calibration_frame)]
    if args.max_frames > 0:
        base_cmd += ["--max-frames", str(args.max_frames)]
    if args.skip_ocr:
        base_cmd.append("--skip-ocr")
    if args.save_crops:
        base_cmd.append("--save-crops")
    if args.identity_overrides:
        base_cmd += ["--identity-overrides", str(resolve_path(args.identity_overrides))]

    print("[pipeline-model] Running base pipeline without final render")
    subprocess.run(base_cmd, cwd=ROOT, check=True)

    render_cmd = [
        sys.executable,
        str(ROOT / "tools" / "render_possession_with_model.py"),
        "--video",
        str(video),
        "--tracks",
        str(run_dir / "tracks.json"),
        "--calibration",
        str(run_dir / "court_calibration.json" if not args.calibration else resolve_path(args.calibration)),
        "--events",
        str(run_dir / "events.json"),
        "--output",
        str(run_dir / "final_annotated.mp4"),
        "--output-events",
        str(run_dir / "events_final.json"),
        "--output-possession",
        str(run_dir / "possession_timeline_final.json"),
    ]
    jersey_numbers = run_dir / "jerseys" / "jersey_numbers.json"
    if jersey_numbers.exists():
        render_cmd += ["--jersey-numbers", str(jersey_numbers)]
    if args.identity_overrides:
        render_cmd += ["--identity-overrides", str(resolve_path(args.identity_overrides))]
    if args.possession_detector_model:
        render_cmd += [
            "--possession-detector-model",
            str(resolve_path(args.possession_detector_model)),
            "--possession-detector-conf",
            str(args.possession_detector_conf),
            "--possession-detector-imgsz",
            str(args.possession_detector_imgsz),
        ]
        if args.possession_detector_device:
            render_cmd += ["--possession-detector-device", args.possession_detector_device]
    if args.possession_model:
        render_cmd += [
            "--possession-model",
            str(resolve_path(args.possession_model)),
            "--possession-min-confidence",
            str(args.possession_min_confidence),
            "--possession-min-margin",
            str(args.possession_min_margin),
        ]
    if args.debug_possession:
        render_cmd.append("--debug-possession")

    print("[pipeline-model] Rendering final video with learned possession layer")
    subprocess.run(render_cmd, cwd=ROOT, check=True)
    print(f"[pipeline-model] Done: {run_dir / 'final_annotated.mp4'}")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
