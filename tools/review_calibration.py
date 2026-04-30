from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basketball_cv.court import CourtSpec, court_marking_polylines, load_calibration, project_court_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the calibrated court over a video frame.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--calibration", required=True, help="Calibration JSON from tools/calibrate_court.py.")
    parser.add_argument("--output", default="previews/calibration_review.jpg", help="Output preview image.")
    parser.add_argument("--frame", type=int, default=None, help="Frame index. Defaults to calibration frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    frame_index = args.frame if args.frame is not None else int(calibration.get("frame_index", 0))
    frame = read_frame(args.video, frame_index)
    if frame is None:
        raise SystemExit(f"Could not read frame {frame_index} from {args.video}")

    spec = CourtSpec(**calibration["court"])
    homography = calibration["homography"]
    draw_calibrated_court(frame, spec, homography)
    draw_point_pairs(frame, calibration)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    print(f"Saved calibration review to {output}")


def read_frame(video_path: str, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_calibrated_court(frame: np.ndarray, spec: CourtSpec, homography: np.ndarray) -> None:
    for line_m in court_marking_polylines(spec):
        line_px = project_court_to_image(line_m, homography)
        pts = line_px.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)


def draw_point_pairs(frame: np.ndarray, calibration: dict) -> None:
    for i, point in enumerate(calibration.get("points", []), start=1):
        x, y = [int(round(v)) for v in point["image"]]
        cv2.circle(frame, (x, y), 7, (0, 80, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, str(i), (x + 9, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, str(i), (x + 9, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
