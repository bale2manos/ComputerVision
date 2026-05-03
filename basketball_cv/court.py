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
    *,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw a readable FIBA-like top-down court used by calibration and minimap.

    The previous schematic was intentionally minimal, but it made calibration hard
    because the 3-point line and visual landmarks were missing. This version adds
    the main FIBA markings while keeping coordinates in metres and the same origin.
    """

    spec = spec or CourtSpec()
    width_px, height_px = topdown_size(spec, pixels_per_meter)
    canvas = np.full((height_px + margin_px * 2, width_px + margin_px * 2, 3), (242, 238, 229), dtype=np.uint8)

    line = (38, 38, 38)
    soft = (120, 120, 120)
    paint = (218, 226, 232)
    hoop_color = (30, 60, 220)
    arc_color = (55, 55, 55)

    def p(x_m: float, y_m: float) -> tuple[int, int]:
        return tuple(court_to_canvas(np.array([[x_m, y_m]], dtype=np.float32), spec, pixels_per_meter, margin_px)[0].astype(int))

    def poly(points: np.ndarray, color: tuple[int, int, int], thickness: int = 2, closed: bool = False) -> None:
        pts = court_to_canvas(points.astype(np.float32), spec, pixels_per_meter, margin_px).round().astype(np.int32)
        cv2.polylines(canvas, [pts], closed, color, thickness, cv2.LINE_AA)

    # Outer court and centre markings.
    cv2.rectangle(canvas, p(0, 0), p(spec.length_m, spec.width_m), line, max(2, pixels_per_meter // 16), cv2.LINE_AA)
    cv2.line(canvas, p(spec.length_m / 2, 0), p(spec.length_m / 2, spec.width_m), soft, max(1, pixels_per_meter // 28), cv2.LINE_AA)
    cv2.circle(canvas, p(spec.length_m / 2, spec.width_m / 2), int(round(1.8 * pixels_per_meter)), soft, max(1, pixels_per_meter // 28), cv2.LINE_AA)
    cv2.circle(canvas, p(spec.length_m / 2, spec.width_m / 2), int(round(0.16 * pixels_per_meter)), soft, -1, cv2.LINE_AA)

    lane_w = 4.9
    lane_half = lane_w / 2.0
    ft_dist = 5.8
    restricted_r = 1.25
    hoop_y = spec.width_m / 2.0
    hoop_xs = (1.575, spec.length_m - 1.575)

    # Light fill inside the paint for easier visual orientation.
    for left in (True, False):
        if left:
            x0, x1 = 0.0, ft_dist
        else:
            x0, x1 = spec.length_m - ft_dist, spec.length_m
        pts = court_to_canvas(
            np.asarray([(x0, hoop_y - lane_half), (x1, hoop_y - lane_half), (x1, hoop_y + lane_half), (x0, hoop_y + lane_half)], dtype=np.float32),
            spec,
            pixels_per_meter,
            margin_px,
        ).round().astype(np.int32)
        cv2.fillPoly(canvas, [pts], paint, cv2.LINE_AA)

    for side in ("left", "right"):
        if side == "left":
            base_x = 0.0
            lane_x0, lane_x1 = 0.0, ft_dist
            ft_x = ft_dist
            hoop_x = hoop_xs[0]
            backboard_x = hoop_x - 1.2
            restricted_start, restricted_end = -np.pi / 2, np.pi / 2
        else:
            base_x = spec.length_m
            lane_x0, lane_x1 = spec.length_m, spec.length_m - ft_dist
            ft_x = spec.length_m - ft_dist
            hoop_x = hoop_xs[1]
            backboard_x = hoop_x + 1.2
            restricted_start, restricted_end = np.pi / 2, 3 * np.pi / 2

        # Lane, free-throw circle, hoop/backboard.
        poly(rect_points(lane_x0, hoop_y - lane_half, lane_x1, hoop_y + lane_half), line, max(1, pixels_per_meter // 22), closed=True)
        cv2.circle(canvas, p(ft_x, hoop_y), int(round(1.8 * pixels_per_meter)), soft, max(1, pixels_per_meter // 30), cv2.LINE_AA)
        poly(arc_points(hoop_x, hoop_y, restricted_r, restricted_start, restricted_end), soft, max(1, pixels_per_meter // 30))
        cv2.circle(canvas, p(hoop_x, hoop_y), max(3, int(round(0.225 * pixels_per_meter))), hoop_color, 2, cv2.LINE_AA)
        cv2.line(canvas, p(backboard_x, hoop_y - 0.9), p(backboard_x, hoop_y + 0.9), hoop_color, max(1, pixels_per_meter // 25), cv2.LINE_AA)

        # FIBA 3-point line: two straight lines plus arc around the basket.
        draw_three_point_line(canvas, spec, pixels_per_meter, margin_px, side=side, color=arc_color)

    if show_labels:
        label_color = (80, 80, 80)
        cv2.putText(canvas, "2D COURT - click matching landmarks", (margin_px, max(22, margin_px - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1, cv2.LINE_AA)
        for name, xy in (
            ("baseline", (0.3, 0.45)),
            ("center", (spec.length_m / 2 + 0.25, spec.width_m / 2 + 0.35)),
            ("3PT", (2.7, 1.55)),
            ("3PT", (spec.length_m - 3.25, 1.55)),
        ):
            cv2.putText(canvas, name, p(*xy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)

    return canvas


def rect_points(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.asarray([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], dtype=np.float32)


def arc_points(cx: float, cy: float, r: float, start: float, end: float, samples: int = 96) -> np.ndarray:
    ts = np.linspace(start, end, samples)
    return np.stack([cx + np.cos(ts) * r, cy + np.sin(ts) * r], axis=1).astype(np.float32)


def draw_three_point_line(
    canvas: np.ndarray,
    spec: CourtSpec,
    pixels_per_meter: int,
    margin_px: int,
    *,
    side: str,
    color: tuple[int, int, int],
) -> None:
    hoop_y = spec.width_m / 2.0
    hoop_x = 1.575 if side == "left" else spec.length_m - 1.575
    radius = 6.75
    side_y_low = 0.9
    side_y_high = spec.width_m - 0.9
    dy = hoop_y - side_y_low
    dx = float(np.sqrt(max(radius * radius - dy * dy, 0.0)))

    def p(x_m: float, y_m: float) -> tuple[int, int]:
        return tuple(court_to_canvas(np.array([[x_m, y_m]], dtype=np.float32), spec, pixels_per_meter, margin_px)[0].astype(int))

    if side == "left":
        x_join = hoop_x + dx
        cv2.line(canvas, p(0.0, side_y_low), p(x_join, side_y_low), color, max(1, pixels_per_meter // 22), cv2.LINE_AA)
        cv2.line(canvas, p(0.0, side_y_high), p(x_join, side_y_high), color, max(1, pixels_per_meter // 22), cv2.LINE_AA)
        start_angle = -np.arccos(dx / radius)
        end_angle = np.arccos(dx / radius)
        pts = arc_points(hoop_x, hoop_y, radius, start_angle, end_angle)
    else:
        x_join = hoop_x - dx
        cv2.line(canvas, p(spec.length_m, side_y_low), p(x_join, side_y_low), color, max(1, pixels_per_meter // 22), cv2.LINE_AA)
        cv2.line(canvas, p(spec.length_m, side_y_high), p(x_join, side_y_high), color, max(1, pixels_per_meter // 22), cv2.LINE_AA)
        start_angle = np.pi - np.arccos(dx / radius)
        end_angle = np.pi + np.arccos(dx / radius)
        pts = arc_points(hoop_x, hoop_y, radius, start_angle, end_angle)

    canvas_pts = court_to_canvas(pts, spec, pixels_per_meter, margin_px).round().astype(np.int32)
    cv2.polylines(canvas, [canvas_pts], False, color, max(1, pixels_per_meter // 22), cv2.LINE_AA)


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

    def circle(cx: float, cy: float, r: float, start: float = 0.0, end: float = 2 * np.pi) -> np.ndarray:
        return arc_points(cx, cy, r, start, end, samples=samples)

    lines.append(rect_points(0.0, 0.0, spec.length_m, spec.width_m))
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
            restricted_start, restricted_end = -np.pi / 2, np.pi / 2
            three_segments = three_point_polylines(spec, side="left", samples=samples)
        else:
            lane_x0, lane_x1 = spec.length_m, spec.length_m - ft_dist
            ft_x = spec.length_m - ft_dist
            hoop_x = hoop_xs[1]
            restricted_start, restricted_end = np.pi / 2, 3 * np.pi / 2
            three_segments = three_point_polylines(spec, side="right", samples=samples)
        lines.append(rect_points(lane_x0, hoop_y - lane_half, lane_x1, hoop_y + lane_half))
        lines.append(circle(ft_x, hoop_y, 1.8))
        lines.append(circle(hoop_x, hoop_y, 0.225))
        lines.append(circle(hoop_x, hoop_y, 1.25, restricted_start, restricted_end))
        lines.extend(three_segments)

    return lines


def three_point_polylines(spec: CourtSpec, side: str, samples: int = 80) -> list[np.ndarray]:
    hoop_y = spec.width_m / 2.0
    hoop_x = 1.575 if side == "left" else spec.length_m - 1.575
    radius = 6.75
    side_y_low = 0.9
    side_y_high = spec.width_m - 0.9
    dy = hoop_y - side_y_low
    dx = float(np.sqrt(max(radius * radius - dy * dy, 0.0)))
    if side == "left":
        x_join = hoop_x + dx
        return [
            np.asarray([(0.0, side_y_low), (x_join, side_y_low)], dtype=np.float32),
            np.asarray([(0.0, side_y_high), (x_join, side_y_high)], dtype=np.float32),
            arc_points(hoop_x, hoop_y, radius, -np.arccos(dx / radius), np.arccos(dx / radius), samples=samples),
        ]
    x_join = hoop_x - dx
    return [
        np.asarray([(spec.length_m, side_y_low), (x_join, side_y_low)], dtype=np.float32),
        np.asarray([(spec.length_m, side_y_high), (x_join, side_y_high)], dtype=np.float32),
        arc_points(hoop_x, hoop_y, radius, np.pi - np.arccos(dx / radius), np.pi + np.arccos(dx / radius), samples=samples),
    ]


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
