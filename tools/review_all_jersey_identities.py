from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch human-in-the-loop jersey review across all pipeline runs that have a matching video_xx_game folder. "
            "For each run, optionally extracts OCR crops, opens the assisted review UI, writes per-run identity_overrides.json, "
            "and appends corrected crops to a shared OCR fine-tuning dataset."
        )
    )
    parser.add_argument("--runs-root", default="runs/pipeline")
    parser.add_argument("--video-root", default=".")
    parser.add_argument("--run-glob", default="video*_game*", help="Glob for run directories under --runs-root.")
    parser.add_argument("--only", default="", help="Comma-separated run names or video stems to process, e.g. video_3,video_8_game.")
    parser.add_argument("--skip", default="", help="Comma-separated run names or video stems to skip.")
    parser.add_argument("--labeled-crops-dir", default="datasets/jersey_ocr_labeled")
    parser.add_argument("--group-by", choices=["player", "track"], default="player")
    parser.add_argument("--samples-per-group", type=int, default=12)

    parser.add_argument("--ocr-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-crops-per-player", type=int, default=80)
    parser.add_argument("--sample-step", type=int, default=5)
    parser.add_argument("--save-crops", action="store_true", default=True)
    parser.add_argument("--no-save-crops", action="store_false", dest="save_crops")
    parser.add_argument("--extract-missing-ocr", action="store_true", default=True)
    parser.add_argument("--no-extract-missing-ocr", action="store_false", dest="extract_missing_ocr")
    parser.add_argument("--force-extract-ocr", action="store_true")

    parser.add_argument("--skip-reviewed", action="store_true", default=True)
    parser.add_argument("--no-skip-reviewed", action="store_false", dest="skip_reviewed")
    parser.add_argument("--append", action="store_true", default=True)
    parser.add_argument("--no-append", action="store_false", dest="append")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = resolve_path(args.runs_root)
    video_root = resolve_path(args.video_root)
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")

    selected = split_set(args.only)
    skipped = split_set(args.skip)
    runs = discover_runs(runs_root, args.run_glob, selected, skipped)
    if not runs:
        raise SystemExit("No matching runs found.")

    print(f"[batch-review] Found {len(runs)} candidate runs")
    for run_dir in runs:
        process_run(run_dir, video_root, args)

    print("\n[batch-review] Done")
    print(f"Shared labeled OCR dataset: {resolve_path(args.labeled_crops_dir)}")


def process_run(run_dir: Path, video_root: Path, args: argparse.Namespace) -> None:
    print(f"\n=== {run_dir.name} ===")
    tracks = run_dir / "tracks.json"
    player_summary = run_dir / "player_summary.json"
    jerseys_dir = run_dir / "jerseys"
    jersey_report = jerseys_dir / "jersey_numbers.json"
    crops_dir = jerseys_dir / "crops"
    overrides = run_dir / "identity_overrides.json"

    if not tracks.exists():
        print(f"[skip] Missing tracks.json: {tracks}")
        return

    video = resolve_video_for_run(run_dir, video_root)
    if video is None or not video.exists():
        print(f"[skip] Could not resolve source video for {run_dir.name}")
        return
    print(f"[video] {video}")

    if args.skip_reviewed and overrides.exists():
        existing = load_json(overrides)
        reviewed_count = len(existing.get("reviewed_groups", []))
        if reviewed_count > 0:
            print(f"[skip] Already reviewed ({reviewed_count} groups): {overrides}")
            return

    if args.force_extract_ocr or should_extract_ocr(jersey_report, crops_dir, args):
        if not player_summary.exists():
            print(f"[warn] Missing player_summary.json: {player_summary}. OCR extraction may fail depending on extract_jersey_numbers.py requirements.")
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "extract_jersey_numbers.py"),
            "--video",
            str(video),
            "--tracks",
            str(tracks),
            "--player-summary",
            str(player_summary),
            "--output-dir",
            str(jerseys_dir),
            "--device",
            args.ocr_device,
            "--max-crops-per-player",
            str(args.max_crops_per_player),
            "--sample-step",
            str(args.sample_step),
        ]
        if args.save_crops:
            cmd.append("--save-crops")
        run_command("extract_jersey_numbers", cmd, args.dry_run)

    if not jersey_report.exists():
        print(f"[skip] Missing jersey report after OCR: {jersey_report}")
        return
    if not crops_dir.exists():
        print(f"[warn] Missing crops dir: {crops_dir}. Review can still generate crops from video, but saved OCR crops are preferable.")

    review_cmd = [
        sys.executable,
        str(ROOT / "tools" / "review_jersey_identities.py"),
        "--video",
        str(video),
        "--tracks",
        str(tracks),
        "--jersey-report",
        str(jersey_report),
        "--output-overrides",
        str(overrides),
        "--labeled-crops-dir",
        str(resolve_path(args.labeled_crops_dir)),
        "--group-by",
        args.group_by,
        "--samples-per-group",
        str(args.samples_per_group),
    ]
    if crops_dir.exists():
        review_cmd += ["--crops-dir", str(crops_dir)]
    if args.append:
        review_cmd.append("--append")
    run_command("review_jersey_identities", review_cmd, args.dry_run)


def should_extract_ocr(jersey_report: Path, crops_dir: Path, args: argparse.Namespace) -> bool:
    if not args.extract_missing_ocr:
        return False
    if not jersey_report.exists():
        return True
    if args.save_crops and not crops_dir.exists():
        return True
    if args.save_crops and crops_dir.exists() and not any(crops_dir.rglob("*.jpg")):
        return True
    return False


def discover_runs(runs_root: Path, pattern: str, selected: set[str], skipped: set[str]) -> list[Path]:
    candidates = [path for path in runs_root.glob(pattern) if path.is_dir()]
    output = []
    for run in sorted(candidates, key=lambda p: natural_key(p.name)):
        video_stem = infer_video_stem_from_run(run.name)
        aliases = {run.name.lower(), video_stem.lower()}
        if selected and not aliases & selected:
            continue
        if aliases & skipped:
            continue
        output.append(run)
    return output


def resolve_video_for_run(run_dir: Path, video_root: Path) -> Path | None:
    summary = run_dir / "game_pipeline_summary.json"
    if summary.exists():
        data = load_json(summary)
        video = data.get("video")
        if video:
            path = Path(video)
            if path.exists():
                return path
            candidate = ROOT / path
            if candidate.exists():
                return candidate

    stem = infer_video_stem_from_run(run_dir.name)
    for ext in VIDEO_EXTS:
        candidate = video_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(video_root.glob(f"{stem}.*"))
    for match in matches:
        if match.suffix.lower() in VIDEO_EXTS:
            return match
    return None


def infer_video_stem_from_run(run_name: str) -> str:
    # video_8_game, video_8_game_ocr, video_8_ball_mixed -> video_8
    match = re.match(r"(video[_-]?\d+)", run_name, re.IGNORECASE)
    if match:
        return match.group(1)
    for suffix in ("_game", "-game"):
        if suffix in run_name:
            return run_name.split(suffix)[0]
    return run_name


def run_command(name: str, cmd: list[str], dry_run: bool) -> None:
    print(f"\n[{name}]")
    print(format_command(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def split_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def format_command(cmd: list[str]) -> str:
    return " ".join(quote(part) for part in cmd)


def quote(value: str) -> str:
    return f'"{value}"' if re.search(r"\s", str(value)) else str(value)


if __name__ == "__main__":
    main()
