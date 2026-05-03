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
            "One-command basketball CV pipeline: court calibration, tracking, ball model, "
            "optional team calibration, role classifier, OCR, possession classifier, and final render."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-root", default="runs/pipeline")
    parser.add_argument("--run-name", default=None)

    # Court calibration / tracking
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--calibration-frame", type=int, default=120)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--tracker", default="trackers/bytetrack_basketball.yaml")
    parser.add_argument("--ball-model", default="auto", help="auto, none, or explicit .pt path")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--ball-imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--ball-conf", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--save-crops", action="store_true")

    # OCR
    parser.add_argument("--with-ocr", action="store_true", help="Run jersey OCR and pass jersey_numbers.json into final render.")
    parser.add_argument("--ocr-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--jersey-numbers", default="auto", help="auto, none, or explicit jersey_numbers.json path for final render")
    parser.add_argument("--identity-overrides", default="auto", help="auto, none, or explicit identity_overrides.json path for final render")

    # Role and team calibration
    parser.add_argument("--role-model", default="auto", help="auto, none, or explicit .pt path")
    parser.add_argument("--role-min-confidence", type=float, default=0.55)
    parser.add_argument("--role-sample-step", type=int, default=3)
    parser.add_argument("--team-calibration", default="auto", help="auto, none, or explicit JSON path")
    parser.add_argument("--team-calibration-frame", type=int, default=120)
    parser.add_argument("--skip-team-calibration", action="store_true")
    parser.add_argument("--team-switch-min-frames", type=int, default=12)
    parser.add_argument("--team-switch-max-gap", type=int, default=5)
    parser.add_argument("--team-switch-strong-margin", type=float, default=0.045)

    # Possession classifier
    parser.add_argument("--possession-model", default="none", help="auto, none, or explicit .pt path")
    parser.add_argument("--possession-max-candidates", type=int, default=7)
    parser.add_argument("--possession-min-confidence", type=float, default=0.45)
    parser.add_argument("--possession-min-margin", type=float, default=0.05)

    # Render
    parser.add_argument("--debug-possession", action="store_true")
    parser.add_argument("--no-possession-hud", action="store_true")
    parser.add_argument("--no-detect-passes", action="store_true")
    parser.add_argument("--no-dense-ball-track", action="store_true")
    parser.add_argument("--no-minimap", action="store_true")
    parser.add_argument("--dense-ball-max-gap", type=float, default=3.0)
    parser.add_argument("--clean-render", action="store_true", help="Alias for --no-possession-hud --no-detect-passes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = resolve_path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    run_name = args.run_name or f"{safe_stem(video.stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = resolve_path(args.output_root)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video": str(video),
        "run_dir": str(run_dir),
        "steps": [],
    }

    calibration = resolve_optional_path(args.calibration) if args.calibration else run_dir / "court_calibration.json"
    role_model = resolve_model(args.role_model, role_model_candidates())
    possession_model = resolve_model(args.possession_model, possession_model_candidates())
    team_calibration = resolve_team_calibration(args.team_calibration, run_dir)

    print_header(run_dir, calibration, role_model, team_calibration, possession_model)

    run_base_pipeline(args, video, run_dir, calibration, summary)

    tracks = run_dir / "tracks.json"
    events = run_dir / "events.json"
    require_file(tracks, "tracks.json")
    require_file(calibration, "court calibration")

    if should_create_team_calibration(args, team_calibration):
        run_step(
            "manual_team_calibration",
            [
                sys.executable,
                str(ROOT / "tools" / "calibrate_teams.py"),
                "--video",
                str(video),
                "--tracks",
                str(tracks),
                "--frame",
                str(args.team_calibration_frame),
                "--output",
                str(team_calibration),
            ],
            summary,
        )

    jersey_numbers = resolve_jersey_numbers(args.jersey_numbers, run_dir)
    identity_overrides = resolve_identity_overrides(args.identity_overrides, run_dir)

    final_video = run_dir / "final_game_pipeline.mp4"
    possession_json = run_dir / "possession_timeline_game_pipeline.json"
    render_cmd = [
        sys.executable,
        str(ROOT / "tools" / "render_possession_with_model.py"),
        "--video",
        str(video),
        "--tracks",
        str(tracks),
        "--calibration",
        str(calibration),
        "--output",
        str(final_video),
        "--output-possession",
        str(possession_json),
        "--team-switch-min-frames",
        str(args.team_switch_min_frames),
        "--team-switch-max-gap",
        str(args.team_switch_max_gap),
        "--team-switch-strong-margin",
        str(args.team_switch_strong_margin),
    ]
    if events.exists():
        render_cmd += ["--events", str(events), "--output-events", str(run_dir / "events_game_pipeline.json")]
    if role_model:
        render_cmd += [
            "--role-model",
            str(role_model),
            "--role-min-confidence",
            str(args.role_min_confidence),
            "--role-sample-step",
            str(args.role_sample_step),
        ]
    if team_calibration and team_calibration.exists():
        render_cmd += ["--team-calibration", str(team_calibration)]
    if jersey_numbers and jersey_numbers.exists():
        render_cmd += ["--jersey-numbers", str(jersey_numbers)]
    if identity_overrides and identity_overrides.exists():
        render_cmd += ["--identity-overrides", str(identity_overrides)]
    if possession_model:
        render_cmd += [
            "--possession-model",
            str(possession_model),
            "--possession-max-candidates",
            str(args.possession_max_candidates),
            "--possession-min-confidence",
            str(args.possession_min_confidence),
            "--possession-min-margin",
            str(args.possession_min_margin),
        ]
    if args.clean_render:
        args.no_possession_hud = True
        args.no_detect_passes = True
    if args.no_possession_hud:
        render_cmd.append("--no-possession-hud")
    if args.no_detect_passes:
        render_cmd.append("--no-detect-passes")
    if args.no_dense_ball_track:
        render_cmd.append("--no-dense-ball-track")
    else:
        render_cmd += ["--dense-ball-max-gap", str(args.dense_ball_max_gap)]
    if args.no_minimap:
        render_cmd.append("--no-minimap")
    if args.debug_possession:
        render_cmd.append("--debug-possession")

    run_step("final_render_roles_teams_possession", render_cmd, summary)

    summary["outputs"] = {
        "final_video": str(final_video) if final_video.exists() else None,
        "tracks": str(tracks),
        "events": str(events) if events.exists() else None,
        "court_calibration": str(calibration),
        "team_calibration": str(team_calibration) if team_calibration and team_calibration.exists() else None,
        "role_model": str(role_model) if role_model else None,
        "possession_model": str(possession_model) if possession_model else None,
        "jersey_numbers": str(jersey_numbers) if jersey_numbers and jersey_numbers.exists() else None,
        "identity_overrides": str(identity_overrides) if identity_overrides and identity_overrides.exists() else None,
        "possession_timeline": str(possession_json) if possession_json.exists() else None,
        "command_log": str(run_dir / "game_pipeline_commands.txt"),
    }
    write_command_log(run_dir / "game_pipeline_commands.txt", summary["steps"])
    (run_dir / "game_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[game-pipeline] Done")
    print(f"[game-pipeline] Run folder: {run_dir}")
    print(f"[game-pipeline] Final video: {final_video}")
    print(f"[game-pipeline] Summary: {run_dir / 'game_pipeline_summary.json'}")


def run_base_pipeline(args: argparse.Namespace, video: Path, run_dir: Path, calibration: Path, summary: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_full_pipeline_enhanced.py"),
        "--video",
        str(video),
        "--output-root",
        str(run_dir.parent),
        "--run-name",
        run_dir.name,
        "--calibration",
        str(calibration),
        "--calibration-frame",
        str(args.calibration_frame),
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
        "--skip-render",
    ]
    if args.reuse_calibration:
        cmd.append("--reuse-calibration")
    if args.skip_calibration:
        cmd.append("--skip-calibration")
    if args.skip_analysis:
        cmd.append("--skip-analysis")
    if args.max_frames > 0:
        cmd += ["--max-frames", str(args.max_frames)]
    if not args.with_ocr:
        cmd.append("--skip-ocr")
    else:
        cmd += ["--ocr-device", args.ocr_device]
        if args.save_crops:
            cmd.append("--save-crops")
    if args.no_dense_ball_track:
        cmd.append("--no-dense-ball-track")
    else:
        cmd += ["--dense-ball-max-gap", str(args.dense_ball_max_gap)]
    run_step("base_tracking_ball", cmd, summary)


def resolve_jersey_numbers(value: str | None, run_dir: Path) -> Path | None:
    if value is None or str(value).lower() in {"none", "false", "no"}:
        return None
    if str(value).lower() == "auto":
        candidate = run_dir / "jerseys" / "jersey_numbers.json"
        return candidate if candidate.exists() else None
    return resolve_path(value)


def resolve_identity_overrides(value: str | None, run_dir: Path) -> Path | None:
    if value is None or str(value).lower() in {"none", "false", "no"}:
        return None
    if str(value).lower() == "auto":
        candidate = run_dir / "identity_overrides.json"
        return candidate if candidate.exists() else None
    return resolve_path(value)


def should_create_team_calibration(args: argparse.Namespace, team_calibration: Path | None) -> bool:
    if args.skip_team_calibration:
        return False
    if args.team_calibration is None or str(args.team_calibration).lower() in {"none", "false", "no"}:
        return False
    if team_calibration is None:
        return False
    if team_calibration.exists():
        print(f"[game-pipeline] Reusing team calibration: {team_calibration}")
        return False
    return True


def resolve_team_calibration(value: str | None, run_dir: Path) -> Path | None:
    if value is None or str(value).lower() in {"none", "false", "no"}:
        return None
    if str(value).lower() == "auto":
        return run_dir / "team_calibration.json"
    return resolve_path(value)


def role_model_candidates() -> list[Path]:
    roots = [
        ROOT / "runs" / "classify" / "runs" / "person_roles",
        ROOT / "runs" / "person_roles",
        ROOT / "runs" / "classify" / "person_roles",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/weights/best.pt"))
    return candidates


def possession_model_candidates() -> list[Path]:
    roots = [
        ROOT / "runs" / "classify" / "runs" / "possession_cls",
        ROOT / "runs" / "possession_cls",
        ROOT / "runs" / "classify" / "possession_cls",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/weights/best.pt"))
    return candidates


def resolve_model(value: str | None, candidates: list[Path]) -> Path | None:
    if value is None or str(value).lower() in {"", "none", "false", "no"}:
        return None
    if str(value).lower() != "auto":
        path = resolve_path(value)
        return path if path.exists() else None
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def run_step(name: str, cmd: list[str], summary: dict[str, Any]) -> None:
    print(f"\n[game-pipeline] Starting: {name}")
    print(format_command(cmd))
    started = datetime.now().isoformat(timespec="seconds")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        summary["steps"].append(
            {
                "step": name,
                "status": "failed",
                "returncode": exc.returncode,
                "started_at": started,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "command": cmd,
            }
        )
        raise
    summary["steps"].append(
        {
            "step": name,
            "status": "ok",
            "returncode": 0,
            "started_at": started,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "command": cmd,
        }
    )
    print(f"[game-pipeline] Finished: {name}")


def write_command_log(path: Path, steps: list[dict[str, Any]]) -> None:
    lines = []
    for step in steps:
        lines.append(f"# {step['step']} [{step['status']}] {step['started_at']} -> {step['finished_at']}")
        lines.append(format_command(step["command"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_header(
    run_dir: Path,
    calibration: Path,
    role_model: Path | None,
    team_calibration: Path | None,
    possession_model: Path | None,
) -> None:
    print("\n=== Basketball CV game pipeline ===")
    print(f"Run directory: {run_dir}")
    print(f"Court calibration: {calibration}")
    print(f"Role model: {role_model if role_model else 'disabled/not found'}")
    print(f"Team calibration: {team_calibration if team_calibration else 'disabled'}")
    print(f"Possession model: {possession_model if possession_model else 'disabled/not found'}")
    print("==================================\n")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def resolve_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return resolve_path(value)


def format_command(cmd: list[str]) -> str:
    return " ".join(quote_arg(part) for part in cmd)


def quote_arg(value: str) -> str:
    return f'"{value}"' if re.search(r"\s", value) else value


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "video"


if __name__ == "__main__":
    main()
