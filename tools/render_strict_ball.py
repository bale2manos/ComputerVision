from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render only high-confidence ball detections. No interpolation or possession.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-area", type=float, default=18.0)
    parser.add_argument("--max-area-ratio", type=float, default=0.004)
    parser.add_argument("--show-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_out = Path(args.json_output) if args.json_output else out.with_suffix(".json")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_area = float(max(width * height, 1))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = YOLO(args.model)

    records: list[dict[str, Any]] = []
    frame_index = 0
    detected_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_index >= args.max_frames:
            break
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        detections = parse_detections(result, frame_index, fps, frame_area, args.min_area, args.max_area_ratio)
        if detections:
            detected_frames += 1
            selected = detections if args.show_all else [max(detections, key=lambda item: item["confidence"])]
            for det in selected:
                draw_ball(frame, det)
                records.append(det)
        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()
    payload = {
        "video": args.video,
        "model": args.model,
        "fps": fps,
        "frames_processed": frame_index,
        "detected_frames": detected_frames,
        "detection_rate": round(detected_frames / max(frame_index, 1), 4),
        "parameters": vars(args),
        "records": records,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"frames_processed": frame_index, "detected_frames": detected_frames, "detection_rate": payload["detection_rate"]}, indent=2))
    print(out.resolve())
    print(json_out.resolve())


def parse_detections(result: Any, frame_index: int, fps: float, frame_area: float, min_area: float, max_area_ratio: float) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    detections = []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    for box, score in zip(xyxy, conf):
        x1, y1, x2, y2 = [float(v) for v in box]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area or area / frame_area > max_area_ratio:
            continue
        detections.append({
            "frame_index": frame_index,
            "time_s": round(frame_index / max(fps, 1e-6), 4),
            "class_name": "sports ball",
            "source": "strict_ball_model",
            "confidence": round(float(score), 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "center_px": [round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
            "bbox_area": round(float(area), 2),
        })
    return detections


def draw_ball(frame: np.ndarray, det: dict[str, Any]) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in det["bbox"]]
    cx, cy = [int(round(float(v))) for v in det["center_px"]]
    radius = max(8, int(round(max(x2 - x1, y2 - y1) * 0.75)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 190, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius + 2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, (0, 190, 255), 3, cv2.LINE_AA)
    text = f"ball {float(det['confidence']):.2f}"
    cv2.putText(frame, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 190, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
