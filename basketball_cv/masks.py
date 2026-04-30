from __future__ import annotations

import cv2
import numpy as np


def cleanup_disconnected_mask(
    mask: np.ndarray,
    max_center_distance_px: float = 80.0,
    min_area_px: int = 24,
) -> np.ndarray:
    """Keep the main body and nearby mask islands, dropping distant fragments.

    Mirrors the cleanup idea described in Roboflow's RF-DETR + SAM2 workflow:
    find connected components, treat the largest one as the player body, and remove
    smaller components whose center is too far from the main component center.
    """

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    component_ids = [i for i in range(1, num_labels) if int(stats[i, cv2.CC_STAT_AREA]) >= min_area_px]
    if not component_ids:
        return np.zeros_like(binary)

    main_id = max(component_ids, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    main_center = centroids[main_id]
    cleaned = np.zeros_like(binary)

    for component_id in component_ids:
        center = centroids[component_id]
        distance = float(np.linalg.norm(center - main_center))
        if component_id == main_id or distance <= max_center_distance_px:
            cleaned[labels == component_id] = 1
    return cleaned


def mask_to_xyxy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
