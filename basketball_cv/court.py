from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CourtSpec:
    """Top-down court coordinates in metres."""

    length_m: float = 28.0
    width_m: float = 15.0


def topdown_size(spec: CourtSpec, pixels_per_meter: int = 40) -> tuple[int, int]:
    return int(round(spec.length_m * pixels_per_meter)), int(round(spec.width_m * pixels_per_meter))


def court_to_canvas(
    points_m: np.ndarray,
    spec: CourtSpec,
    pixels_per_meter: int = 40,
    margin_px: int = 30,
) -> np.ndarray:
    pts = np.asarray(points_m, dtype=np.float32).reshape(-1, 2)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] * pixels_per_meter + margin_px
    out[:, 1] = (spec.width_m - pts[:, 1]) * pixels_per_meter + margin_px
    return out


def canvas_to_court(
    point_px: tuple[int, int],
    spec: CourtSpec,
    pixels_per_meter: int = 40,
    margin_px: int = 30,
) -> tuple[float, float]:
    x_px, y_px = point_px
    x_m = (x_px - margin_px) / pixels_per_meter
    y_m = spec.width_m - ((y_px - margin_px) / pixels_per_meter)
    return float(x_m), float(y_m)


def draw_topdown_court(
    spec: CourtSpec | None = None,
    pixels_per_meter: int = 40,
    margin_px: int = 30,
) -> np.ndarray:
    spec = spec or CourtSpec()
    width_px, height_px = topdown_size(spec, pixels_per_meter)
    canvas = np.full((height_px + margin_px * 2, width_px + margin_px * 2, 3), 245, dtype=np.uint8)

    line = (35, 35, 35)
    red = (40, 40, 210)
    gray = (170, 170, 170)

    def p(x_m: float, y_m: float) -> tuple[int, int]:
        return tuple(court_to_canvas(np.array([[x_m, y_m]], dtype=np.float32), spec, pixels_per_meter, margin_px)[0].astype(int))

    cv2.rectangle(canvas, p(0, 0), p(spec.length_m, spec.width_m), line, 2)
    cv2.line(canvas, p(spec.length_m / 2, 0), p(spec.length_m / 2, spec.width_m), gray, 1)
    cv2.circle(canvas, p(spec.length_m / 2, spec.width_m / 2), int(round(1.8 * pixels_per_meter)), gray, 1)

    # FIBA-ish lane and restricted area markings, enough for visual calibration.
    lane_w = 4.9
    lane_half = lane_w / 2
    ft_dist = 5.8
    hoop_xs = (1.575, spec.length_m - 1.575)
    hoop_y = spec.width_m / 2
    for side in ("left", "right"):
        if side == "left":
            base_x = 0.0
            ft_x = ft_dist
            lane_x0, lane_x1 = 0.0, ft_dist
            hoop_x = hoop_xs[0]
            sign = 1
        else:
            base_x = spec.length_m
            ft_x = spec.length_m - ft_dist
            lane_x0, lane_x1 = spec.length_m, spec.length_m - ft_dist
            hoop_x = hoop_xs[1]
            sign = -1

        cv2.rectangle(canvas, p(lane_x0, hoop_y - lane_half), p(lane_x1, hoop_y + lane_half), gray, 1)
        cv2.circle(canvas, p(ft_x, hoop_y), int(round(1.8 * pixels_per_meter)), gray, 1)
        cv2.circle(canvas, p(hoop_x, hoop_y), int(round(0.225 * pixels_per_meter)), red, 2)
        cv2.line(canvas, p(base_x, hoop_y - 0.9), p(hoop_x - sign * 0.4, hoop_y), red, 1)
        cv2.line(canvas, p(base_x, hoop_y + 0.9), p(hoop_x - sign * 0.4, hoop_y), red, 1)

    return canvas


def compute_homography(image_points: np.ndarray, court_points: np.ndarray) -> np.ndarray:
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    court_points = np.asarray(court_points, dtype=np.float32).reshape(-1, 2)
    if len(image_points) < 4 or len(court_points) < 4:
        raise ValueError("Need at least 4 point pairs to compute a homography.")
    h, mask = cv2.findHomography(image_points, court_points, method=cv2.RANSAC, ransacReprojThreshold=0.4)
    if h is None:
        raise ValueError("Could not compute homography from the selected points.")
    return h


def project_points(points_px: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, np.asarray(homography, dtype=np.float64))
    return projected.reshape(-1, 2)


def project_court_to_image(points_m: np.ndarray, homography: np.ndarray) -> np.ndarray:
    inv_h = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    pts = np.asarray(points_m, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, inv_h)
    return projected.reshape(-1, 2)


def court_marking_polylines(spec: CourtSpec | None = None, samples: int = 80) -> list[np.ndarray]:
    spec = spec or CourtSpec()
    lines: list[np.ndarray] = []

    def segment(a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
        return np.asarray([a, b], dtype=np.float32)

    def rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        return np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], dtype=np.float32)

    def circle(cx: float, cy: float, r: float, start: float = 0.0, end: float = 2 * np.pi) -> np.ndarray:
        ts = np.linspace(start, end, samples)
        return np.stack([cx + np.cos(ts) * r, cy + np.sin(ts) * r], axis=1).astype(np.float32)

    lines.append(rect(0.0, 0.0, spec.length_m, spec.width_m))
    lines.append(segment((spec.length_m / 2, 0.0), (spec.length_m / 2, spec.width_m)))
    lines.append(circle(spec.length_m / 2, spec.width_m / 2, 1.8))

    lane_w = 4.9
    lane_half = lane_w / 2
    ft_dist = 5.8
    hoop_y = spec.width_m / 2
    hoop_xs = (1.575, spec.length_m - 1.575)
    for side in ("left", "right"):
        if side == "left":
            lane_x0, lane_x1 = 0.0, ft_dist
            ft_x = ft_dist
            hoop_x = hoop_xs[0]
        else:
            lane_x0, lane_x1 = spec.length_m, spec.length_m - ft_dist
            ft_x = spec.length_m - ft_dist
            hoop_x = hoop_xs[1]
        lines.append(rect(lane_x0, hoop_y - lane_half, lane_x1, hoop_y + lane_half))
        lines.append(circle(ft_x, hoop_y, 1.8))
        lines.append(circle(hoop_x, hoop_y, 0.225))

    return lines

def inside_court(points_m: np.ndarray, spec: CourtSpec, margin_m: float = 0.75) -> np.ndarray:
    pts = np.asarray(points_m, dtype=np.float32).reshape(-1, 2)
    return (
        (pts[:, 0] >= -margin_m)
        & (pts[:, 0] <= spec.length_m + margin_m)
        & (pts[:, 1] >= -margin_m)
        & (pts[:, 1] <= spec.width_m + margin_m)
    )


def load_calibration(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["homography"] = np.asarray(data["homography"], dtype=np.float64)
    return data


def save_calibration(path: str | Path, data: dict[str, Any]) -> None:
    serializable = dict(data)
    h = serializable.get("homography")
    if isinstance(h, np.ndarray):
        serializable["homography"] = h.tolist()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
