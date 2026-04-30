from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick manual ball labeler for YOLO training.")
    parser.add_argument("--video", required=True, nargs="+", help="Input video path(s).")
    parser.add_argument("--video-glob", action="append", default=[], help="Optional glob for adding more videos, e.g. *.mp4.")
    parser.add_argument("--output", default="datasets/ball", help="YOLO dataset output directory.")
    parser.add_argument("--start", type=int, default=0, help="Start frame.")
    parser.add_argument("--end", type=int, default=0, help="End frame, 0 means video end.")
    parser.add_argument("--step", type=int, default=10, help="Frame step.")
    parser.add_argument("--box-size", type=int, default=34, help="Initial square box size in pixels.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for train/val split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output = Path(args.output)

    ensure_dataset(output)
    write_data_yaml(output)

    video_paths = collect_video_paths(args.video, args.video_glob)

    saved = 0

    # Importante: WINDOW_AUTOSIZE evita que OpenCV redimensione la ventana manualmente
    # y descuadre las coordenadas del ratón.
    cv2.namedWindow("label_ball", cv2.WINDOW_AUTOSIZE)

    for video_path in video_paths:
        saved = label_video(args, video_path, output, rng, saved)

    cv2.destroyAllWindows()
    print(f"Saved {saved} labelled frames to {output}")


def collect_video_paths(videos: list[str], globs: list[str]) -> list[Path]:
    paths = [Path(video) for video in videos]

    for pattern in globs:
        paths.extend(Path(".").glob(pattern))

    unique: list[Path] = []
    seen = set()

    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    return unique


def label_video(
    args: argparse.Namespace,
    video_path: Path,
    output: Path,
    rng: random.Random,
    saved: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = args.end if args.end > 0 else total_frames - 1
    frame_index = max(0, args.start)
    box_size = args.box_size

    state = {
        "center": None,
        "scale": 1.0,
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = float(state["scale"])

            # Convertimos el clic de la imagen reescalada a coordenadas del frame original.
            original_x = int(x / scale)
            original_y = int(y / scale)

            state["center"] = (original_x, original_y)

    cv2.setMouseCallback("label_ball", on_mouse)

    while frame_index <= end:
        frame = read_frame(cap, frame_index)

        if frame is None:
            break

        state["center"] = None

        while True:
            view = frame.copy()

            draw_overlay(
                view,
                str(video_path.name),
                frame_index,
                total_frames,
                state["center"],
                box_size,
                saved,
            )

            display, scale = resize_for_screen(view)
            state["scale"] = scale

            cv2.imshow("label_ball", display)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                print(f"Saved {saved} labelled frames to {output}")
                raise SystemExit(0)

            if key in (ord("+"), ord("=")):
                box_size += 2

            elif key in (ord("-"), ord("_")):
                box_size = max(8, box_size - 2)

            elif key in (ord("u"), ord("U")):
                state["center"] = None

            elif key in (ord("n"), ord("N")):
                break

            elif key in (ord("e"), ord("E")):
                split = "val" if rng.random() < args.val_ratio else "train"
                save_example(output, split, frame, video_path, frame_index, None, box_size)
                saved += 1
                break

            elif key in (ord(" "), 13, 10):
                split = "val" if rng.random() < args.val_ratio else "train"
                save_example(output, split, frame, video_path, frame_index, state["center"], box_size)
                saved += 1
                break

        frame_index += max(1, args.step)

    cap.release()
    return saved


def ensure_dataset(output: Path) -> None:
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output: Path) -> None:
    data = f"""path: {output.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: ball
"""

    (output / "data.yaml").write_text(data, encoding="utf-8")


def read_frame(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def draw_overlay(
    frame,
    video_name: str,
    frame_index: int,
    total_frames: int,
    center,
    box_size: int,
    saved: int,
) -> None:
    if center:
        x, y = center
        half = box_size // 2

        cv2.rectangle(
            frame,
            (x - half, y - half),
            (x + half, y + half),
            (0, 140, 255),
            2,
        )

        cv2.circle(frame, center, 4, (0, 140, 255), -1)

    text = (
        f"{video_name} | frame {frame_index}/{total_frames} | "
        f"click ball | Space=save | E=empty | N=skip | "
        f"+/- size {box_size} | U=undo | Q=quit | saved {saved}"
    )

    cv2.putText(
        frame,
        text,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def resize_for_screen(frame, max_width: int = 1400, max_height: int = 820):
    h, w = frame.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)

    if scale >= 0.999:
        return frame, 1.0

    resized = cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def save_example(
    output: Path,
    split: str,
    frame,
    video_path: Path,
    frame_index: int,
    center,
    box_size: int,
) -> None:
    stem = f"{safe_stem(video_path.stem)}_frame_{frame_index:06d}"

    image_path = output / "images" / split / f"{stem}.jpg"
    label_path = output / "labels" / split / f"{stem}.txt"

    cv2.imwrite(str(image_path), frame)

    if center is None:
        label_path.write_text("", encoding="utf-8")
        return

    h, w = frame.shape[:2]
    x, y = center

    bw = min(box_size, w)
    bh = min(box_size, h)

    cx = min(max(x / w, 0.0), 1.0)
    cy = min(max(y / h, 0.0), 1.0)

    nw = min(max(bw / w, 0.0), 1.0)
    nh = min(max(bh / h, 0.0), 1.0)

    label_path.write_text(
        f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n",
        encoding="utf-8",
    )


def safe_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "video"


if __name__ == "__main__":
    main()