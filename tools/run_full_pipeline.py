from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete basketball CV pipeline: court calibration, tracking, "
            "ball tracking, jersey OCR, pass detection and final rendering."
        )
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--output-root",
        default="runs/pipeline",
        help="Root directory where this pipeline run will be written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to video stem plus timestamp.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional calibration JSON path. If omitted, one is created inside the run folder.",
    )
    parser.add_argument("--calibration-frame", type=int, default=120, help="Frame used by the interactive calibration tool.")
    parser.add_argument("--court-length", type=float, default=28.0, help="Court length in metres.")
    parser.add_argument("--court-width", type=float, default=15.0, help="Court width in metres.")
    parser.add_argument("--ppm", type=int, default=40, help="Top-down pixels per metre used in calibration UI.")
    parser.add_argument(
        "--reuse-calibration",
        action="store_true",
        help="If the calibration file already exists, reuse it instead of opening the calibration UI.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Do not run the calibration UI. The calibration file must already exist.",
    )
    parser.add_argument("--skip-review", action="store_true", help="Do not generate calibration_review.jpg.")

    parser.add_argument("--model", default="yolo11m.pt", help="Main YOLO model for people and COCO sports ball.")
    parser.add_argument(
        "--tracker",
        default="trackers/bytetrack_basketball.yaml",
        help="Ultralytics tracker YAML.",
    )
    parser.add_argument(
        "--ball-model",
        default="auto",
        help=(
            "Optional trained ball detector. Use 'auto' to pick the newest "
            "runs/ball_train/**/weights/best.pt if present, or 'none' to disable."
        ),
    )
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=768, help="Main model image size.")
    parser.add_argument("--ball-imgsz", type=int, default=960, help="Ball model image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Main model confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="Main model IoU threshold.")
    parser.add_argument("--ball-conf", type=float, default=0.15, help="Ball model confidence threshold.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for quick tests.")
    parser.add_argument("--court-margin", type=float, default=0.9, help="Default court margin in metres.")
    parser.add_argument("--court-near-margin", type=float, default=1.2, help="Near sideline margin in metres.")
    parser.add_argument("--court-far-margin", type=float, default=0.5, help="Far sideline margin in metres.")
    parser.add_argument("--ball-color-fallback", action="store_true", help="Enable experimental orange-blob ball fallback.")
    parser.add_argument("--skip-analysis", action="store_true", help="Do not run tools/analyze_video.py.")

    parser.add_argument("--skip-ocr", action="store_true", help="Do not run jersey OCR.")
    parser.add_argument("--ocr-device", default="cuda", choices=["cuda", "cpu"], help="EasyOCR device preference.")
    parser.add_argument("--max-crops-per-player", type=int, default=40, help="OCR crops sampled per player.")
    parser.add_argument("--ocr-sample-step", type=int, default=6, help="Minimum frame gap between OCR crops.")
    parser.add_argument("--save-crops", action="store_true", help="Save OCR crops for manual review.")
    parser.add_argument(
        "--strict-ocr",
        action="store_true",
        help="Fail the whole pipeline if OCR fails. By default, the final render continues without numbers.",
    )

    parser.add_argument(
        "--identity-overrides",
        default=None,
        help="Optional manual identity override JSON passed to render_tracks.py.",
    )
    parser.add_argument("--skip-render", action="store_true", help="Do not render the final annotated video.")
    parser.add_argument(
        "--no-detect-passes-on-render",
        action="store_true",
        help="Do not recompute pass events after OCR/identity overrides during final render.",
    )
    parser.add_argument(
        "--no-dense-ball-track",
        action="store_true",
        help="Disable render-only dense ball interpolation in analysis and final render.",
    )
    parser.add_argument("--dense-ball-max-gap", type=float, default=3.0, help="Max seconds to visually connect ball detections.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = resolve_input_path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    run_dir = build_run_dir(video, args.output_root, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    calibration = resolve_output_path(args.calibration, run_dir / "court_calibration.json")
    review_image = run_dir / "calibration_review.jpg"
    final_video = run_dir / "final_annotated.mp4"
    command_log = run_dir / "pipeline_commands.txt"
    summary_path = run_dir / "pipeline_summary.json"

    command_history: list[dict[str, Any]] = []
    print_header(run_dir, calibration)

    if args.skip_calibration:
        require_file(calibration, "Calibration JSON")
    elif calibration.exists() and args.reuse_calibration:
        print(f"[pipeline] Reusing calibration: {calibration}")
    else:
        run_step(
            "court_calibration",
            [
                sys.executable,
                str(ROOT / "tools" / "calibrate_court.py"),
                "--video",
                str(video),
                "--output",
                str(calibration),
                "--frame",
                str(args.calibration_frame),
                "--court-length",
                str(args.court_length),
                "--court-width",
                str(args.court_width),
                "--ppm",
                str(args.ppm),
            ],
            command_history,
        )
    require_file(calibration, "Calibration JSON")

    if not args.skip_review:
        run_step(
            "calibration_review",
            [
                sys.executable,
                str(ROOT / "tools" / "review_calibration.py"),
                "--video",
                str(video),
                "--calibration",
                str(calibration),
                "--output",
                str(review_image),
            ],
            command_history,
        )

    ball_model = resolve_ball_model(args.ball_model)
    if ball_model:
        print(f"[pipeline] Using ball model: {ball_model}")
    else:
        print("[pipeline] No custom ball model found/enabled. Analysis will rely on the main detector/fallbacks.")

    if not args.skip_analysis:
        analysis_cmd = [
            sys.executable,
            str(ROOT / "tools" / "analyze_video.py"),
            "--video",
            str(video),
            "--calibration",
            str(calibration),
            "--output-dir",
            str(run_dir),
            "--model",
            args.model,
            "--tracker",
            args.tracker,
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--device",
            args.device,
            "--court-margin",
            str(args.court_margin),
            "--court-near-margin",
            str(args.court_near_margin),
            "--court-far-margin",
            str(args.court_far_margin),
            "--write-video",
        ]
        if args.max_frames > 0:
            analysis_cmd.extend(["--max-frames", str(args.max_frames)])
        if ball_model:
            analysis_cmd.extend(
                [
                    "--ball-model",
                    str(ball_model),
                    "--ball-conf",
                    str(args.ball_conf),
                    "--ball-imgsz",
                    str(args.ball_imgsz),
                ]
            )
        if args.ball_color_fallback:
            analysis_cmd.append("--ball-color-fallback")
        if args.no_dense_ball_track:
            analysis_cmd.append("--no-dense-ball-track")
        else:
            analysis_cmd.extend(["--dense-ball-max-gap", str(args.dense_ball_max_gap)])
        run_step("analysis_tracking_ball_events", analysis_cmd, command_history)

    tracks_path = run_dir / "tracks.json"
    player_summary_path = run_dir / "player_summary.json"
    events_path = run_dir / "events.json"
    require_file(tracks_path, "tracks.json")

    jersey_numbers_path: Path | None = None
    if not args.skip_ocr:
        ocr_dir = run_dir / "jerseys"
        ocr_cmd = [
            sys.executable,
            str(ROOT / "tools" / "extract_jersey_numbers.py"),
            "--video",
            str(video),
            "--tracks",
            str(tracks_path),
            "--output-dir",
            str(ocr_dir),
            "--device",
            args.ocr_device,
            "--max-crops-per-player",
            str(args.max_crops_per_player),
            "--sample-step",
            str(args.ocr_sample_step),
        ]
        if player_summary_path.exists():
            ocr_cmd.extend(["--player-summary", str(player_summary_path)])
        if args.save_crops:
            ocr_cmd.append("--save-crops")

        try:
            run_step("jersey_ocr", ocr_cmd, command_history)
            candidate = ocr_dir / "jersey_numbers.json"
            if candidate.exists():
                jersey_numbers_path = candidate
        except subprocess.CalledProcessError:
            if args.strict_ocr:
                raise
            print("[pipeline] OCR failed. Continuing without jersey numbers. Use --strict-ocr to make this fatal.")
    else:
        print("[pipeline] Skipping jersey OCR by request.")

    if not args.skip_render:
        render_cmd = [
            sys.executable,
            str(ROOT / "tools" / "render_tracks.py"),
            "--video",
            str(video),
            "--tracks",
            str(tracks_path),
            "--calibration",
            str(calibration),
            "--output",
            str(final_video),
        ]
        if jersey_numbers_path is not None:
            render_cmd.extend(["--jersey-numbers", str(jersey_numbers_path)])
        if args.identity_overrides:
            render_cmd.extend(["--identity-overrides", str(resolve_input_path(args.identity_overrides))])
        if events_path.exists():
            render_cmd.extend(["--events", str(events_path), "--output-events", str(run_dir / "events_final.json")])
        if args.no_detect_passes_on_render:
            render_cmd.append("--no-detect-passes")
        if args.no_dense_ball_track:
            render_cmd.append("--no-dense-ball-track")
        else:
            render_cmd.extend(["--dense-ball-max-gap", str(args.dense_ball_max_gap)])
        run_step("final_render", render_cmd, command_history)

    write_command_log(command_log, command_history)
    summary = build_summary(run_dir, video, calibration, review_image, jersey_numbers_path, final_video, command_history)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[pipeline] Done")
    print(f"[pipeline] Run folder: {run_dir}")
    print(f"[pipeline] Final video: {final_video if final_video.exists() else 'not generated'}")
    print(f"[pipeline] Summary: {summary_path}")


def print_header(run_dir: Path, calibration: Path) -> None:
    print("\n=== Basketball CV full pipeline ===")
    print(f"Run directory: {run_dir}")
    print(f"Calibration file: {calibration}")
    print("During calibration: click matching points in the video frame and the 2D court, then press Enter to save.")
    print("==================================\n")


def build_run_dir(video: Path, output_root: str, run_name: str | None) -> Path:
    root = resolve_output_path(output_root, ROOT / output_root)
    name = run_name or f"{safe_stem(video.stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return root / name


def resolve_input_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_output_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


def resolve_ball_model(value: str | None) -> Path | None:
    if value is None or value.lower() in {"", "none", "false", "no"}:
        return None
    if value.lower() != "auto":
        path = resolve_input_path(value)
        return path if path.exists() else None

    candidates = sorted(
        (ROOT / "runs" / "ball_train").glob("**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_step(name: str, cmd: list[str], history: list[dict[str, Any]]) -> None:
    print(f"\n[pipeline] Starting: {name}")
    print(format_command(cmd))
    started = datetime.now().isoformat(timespec="seconds")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        history.append(
            {
                "step": name,
                "status": "failed",
                "returncode": exc.returncode,
                "started_at": started,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "command": cmd,
            }
        )
        print(f"[pipeline] Failed: {name} (exit code {exc.returncode})")
        raise
    history.append(
        {
            "step": name,
            "status": "ok",
            "returncode": 0,
            "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "command": cmd,
        }
    )
    print(f"[pipeline] Finished: {name}")


def write_command_log(path: Path, history: list[dict[str, Any]]) -> None:
    lines = []
    for item in history:
        lines.append(f"# {item['step']} [{item['status']}] {item['started_at']} -> {item['finished_at']}")
        lines.append(format_command(item["command"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    run_dir: Path,
    video: Path,
    calibration: Path,
    review_image: Path,
    jersey_numbers: Path | None,
    final_video: Path,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    outputs = {
        "run_dir": str(run_dir),
        "video": str(video),
        "calibration": str(calibration) if calibration.exists() else None,
        "calibration_review": str(review_image) if review_image.exists() else None,
        "analysis_video": str(run_dir / "annotated.mp4") if (run_dir / "annotated.mp4").exists() else None,
        "final_video": str(final_video) if final_video.exists() else None,
        "tracks": str(run_dir / "tracks.json") if (run_dir / "tracks.json").exists() else None,
        "ball_tracks": str(run_dir / "ball_tracks.json") if (run_dir / "ball_tracks.json").exists() else None,
        "events": str(run_dir / "events.json") if (run_dir / "events.json").exists() else None,
        "events_final": str(run_dir / "events_final.json") if (run_dir / "events_final.json").exists() else None,
        "player_summary": str(run_dir / "player_summary.json") if (run_dir / "player_summary.json").exists() else None,
        "track_summary": str(run_dir / "track_summary.json") if (run_dir / "track_summary.json").exists() else None,
        "jersey_numbers": str(jersey_numbers) if jersey_numbers and jersey_numbers.exists() else None,
    }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": outputs,
        "steps": history,
    }


def format_command(cmd: list[str]) -> str:
    return " ".join(quote_arg(part) for part in cmd)


def quote_arg(value: str) -> str:
    if re.search(r"\s", value):
        return f'"{value}"'
    return value


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "video"


if __name__ == "__main__":
    main()
