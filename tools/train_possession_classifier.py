from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


CLASS_HINTS = """
Expected folder layout, compatible with YOLO classification:

  datasets/possession_cls/
    train/
      control/
      no_control/
      dribble/
      shooting/
      contested/
    val/
      control/
      no_control/
      dribble/
      shooting/
      contested/

Minimal first version:

  datasets/possession_cls/
    train/control
    train/no_control
    val/control
    val/no_control

Class names interpreted as control by the runtime:
  control, has_ball, player_with_ball, dribble, dribbling, shoot, shooting, pass, passing

Class names interpreted as loose/no-control:
  loose, loose_ball, contested, divided, no_control, none
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO classification model for player-ball possession/action state.")
    parser.add_argument("--data", default="datasets/possession_cls", help="Classification dataset root.")
    parser.add_argument("--base-model", default="yolo11s-cls.pt", help="YOLO classification base model.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/possession_cls")
    parser.add_argument("--name", default="basketball_possession_cls")
    parser.add_argument("--show-layout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_layout:
        print(CLASS_HINTS)
        return
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"Dataset not found: {data}\n{CLASS_HINTS}")
    model = YOLO(args.base_model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
