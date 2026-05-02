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

from basketball_cv.role_classifier import person_crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-body person crops for role classification: player/referee/other.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output-dir", default="datasets/person_roles/curation")
    parser.add_argument("--sample-step", type=int, default=8)
    parser.add_argument("--max-crops", type=int, default=3000)
    parser.add_argument("--crop-margin", type=float, default=0.08)
    parser.add_argument("--include-unknown", action="store_true", help="Also export tracks currently marked unknown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    unlabeled_dir = output_dir / "unlabeled"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    records = tracks.get("records", [])
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        if rec.get("class_name") != "person":
            continue
        if not args.include_unknown and rec.get("team") in (None, "unknown"):
            continue
        by_frame.setdefault(int(rec.get("frame_index", 0)), []).append(rec)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    manifest = output_dir / "manifest.csv"
    exported = 0
    current_frame = -1
    frame_img = None
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "frame_index", "track_id", "player_id", "team", "bbox", "confidence"],
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
            for rec in by_frame[frame_index]:
                if args.max_crops and exported >= args.max_crops:
                    break
                crop = person_crop(frame_img, rec, args.crop_margin)
                if crop is None or crop.size == 0:
                    continue
                track_id = rec.get("track_id")
                filename = f"f{frame_index:06d}_t{track_id}_p{rec.get('player_id')}_{rec.get('team')}.jpg"
                out_path = unlabeled_dir / filename
                cv2.imwrite(str(out_path), crop)
                writer.writerow(
                    {
                        "file": str(out_path.relative_to(output_dir)),
                        "frame_index": frame_index,
                        "track_id": track_id,
                        "player_id": rec.get("player_id"),
                        "team": rec.get("team"),
                        "bbox": rec.get("bbox"),
                        "confidence": rec.get("confidence"),
                    }
                )
                exported += 1
    cap.release()
    print(f"Exported {exported} crops to {unlabeled_dir}")
    print("Move images into datasets/person_roles/train|val/player, referee, other before training.")


if __name__ == "__main__":
    main()
