from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.court import (
    CourtSpec,
    canvas_to_court,
    compute_homography,
    draw_topdown_court,
    save_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual court homography calibration.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", default="config/court_calibration.json", help="Calibration JSON output.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to calibrate.")
    parser.add_argument("--court-length", type=float, default=28.0, help="Court length in metres.")
    parser.add_argument("--court-width", type=float, default=15.0, help="Court width in metres.")
    parser.add_argument("--ppm", type=int, default=40, help="Top-down pixels per metre.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = CourtSpec(args.court_length, args.court_width)
    frame = read_frame(args.video, args.frame)
    if frame is None:
        raise SystemExit(f"Could not read frame {args.frame} from {args.video}")

    image_points: list[tuple[float, float]] = []
    court_points: list[tuple[float, float]] = []
    state = {"waiting_for": "image"}

    topdown = draw_topdown_court(spec, pixels_per_meter=args.ppm)
    image_view = resize_for_screen(frame, max_width=1200, max_height=760)
    scale_x = frame.shape[1] / image_view.shape[1]
    scale_y = frame.shape[0] / image_view.shape[0]

    def on_image(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or state["waiting_for"] != "image":
            return
        image_points.append((float(x * scale_x), float(y * scale_y)))
        state["waiting_for"] = "court"

    def on_court(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or state["waiting_for"] != "court":
            return
        court_points.append(canvas_to_court((x, y), spec, pixels_per_meter=args.ppm))
        state["waiting_for"] = "image"

    try:
        cv2.namedWindow("video_frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("topdown_court", cv2.WINDOW_NORMAL)
    except cv2.error:
        sys.stderr.write(
            "\nOpenCV appears to be built without GUI/highgui support.\n"
            "If you installed 'opencv-python-headless', uninstall it and install 'opencv-python'.\n"
            "Run:\n  python -m pip uninstall -y opencv-python-headless\n  python -m pip install --upgrade --force-reinstall opencv-python\n\n"
        )
        raise SystemExit(1)

    cv2.setMouseCallback("video_frame", on_image)
    cv2.setMouseCallback("topdown_court", on_court)

    while True:
        frame_draw = image_view.copy()
        court_draw = topdown.copy()
        draw_points(frame_draw, image_points, sx=1.0 / scale_x, sy=1.0 / scale_y)
        draw_points(court_draw, court_points, court=True, spec=spec, ppm=args.ppm)
        status = f"Pairs: {len(court_points)} | click {state['waiting_for']} | Enter=save, U=undo, Q=quit"
        cv2.putText(frame_draw, status, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_draw, status, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("video_frame", frame_draw)
        cv2.imshow("topdown_court", court_draw)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("u"), ord("U")):
            if state["waiting_for"] == "court" and image_points:
                image_points.pop()
                state["waiting_for"] = "image"
            elif court_points:
                court_points.pop()
                image_points.pop()
                state["waiting_for"] = "image"
        if key in (13, 10):
            if len(image_points) != len(court_points) or len(image_points) < 4:
                print("Need at least four complete image/court point pairs before saving.")
                continue
            h = compute_homography(np.asarray(image_points), np.asarray(court_points))
            save_calibration(
                args.output,
                {
                    "video": str(Path(args.video)),
                    "frame_index": args.frame,
                    "court": {"length_m": spec.length_m, "width_m": spec.width_m},
                    "points": [
                        {"image": [float(ix), float(iy)], "court": [float(cx), float(cy)]}
                        for (ix, iy), (cx, cy) in zip(image_points, court_points)
                    ],
                    "homography": h,
                },
            )
            print(f"Saved calibration to {args.output}")
            break

    cv2.destroyAllWindows()


def read_frame(video_path: str, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def resize_for_screen(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_points(
    image: np.ndarray,
    points: list[tuple[float, float]],
    sx: float = 1.0,
    sy: float = 1.0,
    court: bool = False,
    spec: CourtSpec | None = None,
    ppm: int = 40,
) -> None:
    from basketball_cv.court import court_to_canvas

    for i, point in enumerate(points, start=1):
        if court:
            px = court_to_canvas(np.asarray([point], dtype=np.float32), spec or CourtSpec(), ppm)[0]
            x, y = int(px[0]), int(px[1])
        else:
            x, y = int(point[0] * sx), int(point[1] * sy)
        cv2.circle(image, (x, y), 6, (0, 80, 255), -1)
        cv2.putText(image, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
