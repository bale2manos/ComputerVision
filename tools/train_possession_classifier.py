from __future__ import annotations

import argparse
from pathlib import Path


TASK_DEFAULTS = {
    "ball-state": {
        "data": "datasets/possession_ball_state",
        "base_model": "yolo11s-cls.pt",
    },
    "owner-state": {
        "data": "datasets/possession_owner_state",
        "base_model": "yolo11s-cls.pt",
    },
}

CLASS_HINTS = """
Ball-state dataset:
  datasets/possession_ball_state/
    train/owned
    train/air
    train/loose
    val/owned
    val/air
    val/loose

Owner-state dataset:
  datasets/possession_owner_state/
    train/control
    train/dribble
    train/shot
    train/contested
    train/no_control
    val/control
    val/dribble
    val/shot
    val/contested
    val/no_control
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO classification model for ball-state or owner-state possession tasks.")
    parser.add_argument("--task", required=True, choices=["ball-state", "owner-state"])
    parser.add_argument("--data", default=None, help="Classification dataset root. Defaults depend on --task.")
    parser.add_argument("--base-model", default=None, help="YOLO classification base model. Defaults depend on --task.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/possession_cls")
    parser.add_argument("--name", default=None)
    parser.add_argument("--show-layout", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = TASK_DEFAULTS[args.task]
    if args.show_layout:
        print(CLASS_HINTS.strip())
        return

    data = Path(args.data or defaults["data"])
    base_model = args.base_model or defaults["base_model"]
    run_name = args.name or f"basketball_{args.task.replace('-', '_')}_cls"
    if not data.exists():
        raise SystemExit(f"Dataset not found: {data}\n\n{CLASS_HINTS.strip()}")

    from ultralytics import YOLO

    model = YOLO(base_model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=run_name,
    )


if __name__ == "__main__":
    main()
