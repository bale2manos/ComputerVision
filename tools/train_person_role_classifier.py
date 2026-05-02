from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


LAYOUT = """
Expected dataset layout:

  datasets/person_roles/
    train/
      player/
      referee/
      other/
    val/
      player/
      referee/
      other/

Minimal useful classes:
  player  -> on-court basketball players only
  referee -> referees/officials with your local clothing
  other   -> coaches, spectators, bench players, partial bodies, bad crops
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train player/referee/other role classifier.")
    parser.add_argument("--data", default="datasets/person_roles")
    parser.add_argument("--base-model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/person_roles")
    parser.add_argument("--name", default="person_role_cls")
    parser.add_argument("--show-layout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_layout:
        print(LAYOUT)
        return
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"Dataset not found: {data}\n{LAYOUT}")
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
