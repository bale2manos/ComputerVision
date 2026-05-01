from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None  # type: ignore[assignment]

from basketball_cv.events import _distance, _eligible_players, _player_key, _record_identity
from basketball_cv.possession import ball_player_image_contact, best_non_dense_ball


POSSESSION_STATE_CLASSES = {
    "player-in-possession",
    "player_in_possession",
    "player possession",
    "player_with_ball",
    "player-with-ball",
    "has_ball",
    "control",
}

SHOT_STATE_CLASSES = {
    "player-jump-shot",
    "player_jump_shot",
    "jump-shot",
    "jumpshot",
    "shooting",
    "player-layup-dunk",
    "player_layup_dunk",
    "layup",
    "dunk",
}

CONTESTED_STATE_CLASSES = {
    "player-shot-block",
    "player_shot_block",
    "shot-block",
    "block",
    "contested",
    "divided",
}

IGNORED_STATE_CLASSES = {"player", "ball", "number", "referee", "rim", "ball-in-basket", "ball_in_basket"}


@dataclass
class PossessionDetectorConfig:
    conf: float = 0.35
    imgsz: int = 960
    device: str | None = None
    min_match_iou: float = 0.12
    min_center_score: float = 0.20
    max_center_distance_ratio: float = 0.65
    apply_shot_as_possession: bool = True
    apply_contested_as_possession: bool = False


class PossessionStateDetector:
    """Local YOLO detector for Roboflow-style basketball state classes.

    Expected useful detection classes include:
    - player-in-possession
    - player-jump-shot
    - player-layup-dunk
    - player-shot-block

    Detections are matched to already tracked player boxes and then written back
    into the existing records, so minimap, HUD and pass detection keep working.
    """

    def __init__(self, model_path: str | Path, config: PossessionDetectorConfig | None = None) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is required to use --possession-detector-model")
        self.model_path = Path(model_path)
        self.config = config or PossessionDetectorConfig()
        self.model = YOLO(str(self.model_path))
        self.names = getattr(self.model, "names", {}) or {}

    def apply_to_records(self, records: list[dict[str, Any]], video_path: str | Path, fps: float) -> dict[str, Any]:
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for rec in records:
            by_frame.setdefault(int(rec.get("frame_index", 0)), []).append(rec)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for possession detector: {video_path}")

        applied = 0
        possession_detections = 0
        shot_detections = 0
        contested_detections = 0
        unmatched_detections = 0
        current_frame = -1
        frame_img = None

        for frame_index in sorted(by_frame):
            if frame_index != current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_img = cap.read()
                current_frame = frame_index
                if not ok:
                    frame_img = None
            if frame_img is None:
                continue

            frame_records = by_frame[frame_index]
            players = _eligible_players(frame_records)
            if not players:
                continue

            state_detections = self.predict_state_detections(frame_img)
            if not state_detections:
                continue

            best_assignment = None
            for det in state_detections:
                state_type = state_type_for_class(det["class_name"])
                if state_type == "ignored":
                    continue
                if state_type == "shot" and not self.config.apply_shot_as_possession:
                    continue
                if state_type == "contested" and not self.config.apply_contested_as_possession:
                    continue
                match = match_detection_to_player(det, players, self.config)
                if match is None:
                    unmatched_detections += 1
                    continue
                candidate = {**det, **match, "state_type": state_type}
                if best_assignment is None or assignment_rank(candidate) > assignment_rank(best_assignment):
                    best_assignment = candidate

            if best_assignment is None:
                continue

            if best_assignment["state_type"] == "possession":
                possession_detections += 1
            elif best_assignment["state_type"] == "shot":
                shot_detections += 1
            elif best_assignment["state_type"] == "contested":
                contested_detections += 1

            reassign_owner_from_detector(frame_records, best_assignment)
            applied += 1

        cap.release()
        return {
            "possession_detector_model": str(self.model_path),
            "frames_applied": applied,
            "possession_detections": possession_detections,
            "shot_detections": shot_detections,
            "contested_detections": contested_detections,
            "unmatched_detections": unmatched_detections,
            "classes": self.names,
            "parameters": self.config.__dict__,
        }

    def predict_state_detections(self, frame: np.ndarray) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"conf": self.config.conf, "imgsz": self.config.imgsz, "verbose": False}
        if self.config.device is not None:
            kwargs["device"] = self.config.device
        results = self.model.predict(frame, **kwargs)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = normalize_class_name(str(self.names.get(cls_id, cls_id)))
            if class_name in IGNORED_STATE_CLASSES:
                continue
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            conf = float(box.conf[0])
            detections.append(
                {
                    "bbox": xyxy,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                }
            )
        return detections


def normalize_class_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def state_type_for_class(class_name: str) -> str:
    name = normalize_class_name(class_name)
    if name in POSSESSION_STATE_CLASSES:
        return "possession"
    if name in SHOT_STATE_CLASSES:
        return "shot"
    if name in CONTESTED_STATE_CLASSES:
        return "contested"
    return "ignored"


def match_detection_to_player(
    detection: dict[str, Any],
    players: list[dict[str, Any]],
    config: PossessionDetectorConfig,
) -> dict[str, Any] | None:
    det_box = detection["bbox"]
    scored = []
    for player in players:
        player_box = player.get("bbox")
        if not isinstance(player_box, list) or len(player_box) != 4:
            continue
        iou = bbox_iou(det_box, player_box)
        center_score = center_inside_score(det_box, player_box)
        distance_score = normalized_center_distance_score(det_box, player_box, config.max_center_distance_ratio)
        score = max(iou, center_score, distance_score) * float(detection.get("confidence", 0.0))
        scored.append((score, iou, center_score, distance_score, player))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    score, iou, center_score, distance_score, player = scored[0]
    if iou < config.min_match_iou and center_score < config.min_center_score and distance_score < config.min_center_score:
        return None
    return {
        "player": player,
        "match_score": round(float(score), 4),
        "match_iou": round(float(iou), 4),
        "match_center_score": round(float(center_score), 4),
        "match_distance_score": round(float(distance_score), 4),
    }


def reassign_owner_from_detector(frame_records: list[dict[str, Any]], assignment: dict[str, Any]) -> None:
    player = assignment["player"]
    owner_id = _player_key(player)
    if owner_id is None:
        return

    for rec in frame_records:
        if rec.get("class_name") == "person":
            clear_player_possession(rec)

    ball = best_non_dense_ball(frame_records)
    contact = ball_player_image_contact(ball, player) if ball is not None else None
    distance = _distance(ball, player) if ball is not None else None
    identity = _record_identity(player)
    class_name = assignment.get("class_name")
    reason = f"detector_{class_name}"

    if ball is not None:
        ball["ball_state"] = "owned"
        ball["ball_owner_player_id"] = owner_id
        ball["ball_owner_identity"] = identity
        ball["ball_owner_jersey_number"] = player.get("jersey_number")
        ball["ball_owner_team"] = player.get("team")
        if distance is not None:
            ball["ball_owner_distance_m"] = round(float(distance), 3)
        if contact is not None:
            ball["ball_owner_image_contact"] = round(float(contact), 3)
        ball["ball_owner_confidence"] = round(float(assignment.get("confidence", 0.0)), 3)
        ball["ball_owner_assignment_reason"] = reason
        ball["possession_detector"] = serialize_assignment(assignment)
        if player.get("court_x") is not None and player.get("court_y") is not None:
            ball["possession_court_x"] = player.get("court_x")
            ball["possession_court_y"] = player.get("court_y")

    player["has_ball"] = True
    player["ball_state"] = "owned"
    player["ball_owner_player_id"] = owner_id
    player["ball_owner_identity"] = identity
    player["ball_owner_jersey_number"] = player.get("jersey_number")
    player["ball_owner_team"] = player.get("team")
    if distance is not None:
        player["ball_owner_distance_m"] = round(float(distance), 3)
    if contact is not None:
        player["ball_owner_image_contact"] = round(float(contact), 3)
    player["ball_owner_confidence"] = round(float(assignment.get("confidence", 0.0)), 3)
    player["ball_owner_assignment_reason"] = reason
    player["ball_owner_source"] = ball.get("source") if ball is not None else "state_detector"
    player["possession_detector"] = serialize_assignment(assignment)


def clear_player_possession(rec: dict[str, Any]) -> None:
    for key in (
        "has_ball",
        "ball_state",
        "ball_owner_player_id",
        "ball_owner_identity",
        "ball_owner_jersey_number",
        "ball_owner_team",
        "ball_owner_distance_m",
        "ball_owner_confidence",
        "ball_owner_source",
        "ball_owner_image_contact",
        "ball_owner_assignment_reason",
        "possession_detector",
    ):
        rec.pop(key, None)


def assignment_rank(assignment: dict[str, Any]) -> tuple[float, float, float]:
    state_priority = {"possession": 3.0, "shot": 2.4, "contested": 1.2}.get(str(assignment.get("state_type")), 0.0)
    return (
        state_priority,
        float(assignment.get("confidence", 0.0)),
        float(assignment.get("match_score", 0.0)),
    )


def serialize_assignment(assignment: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in assignment.items() if k != "player"}


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def center_inside_score(det_box: list[float], player_box: list[float]) -> float:
    dx1, dy1, dx2, dy2 = [float(v) for v in det_box]
    px1, py1, px2, py2 = [float(v) for v in player_box]
    dcx, dcy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
    if px1 <= dcx <= px2 and py1 <= dcy <= py2:
        det_area = max(1.0, (dx2 - dx1) * (dy2 - dy1))
        player_area = max(1.0, (px2 - px1) * (py2 - py1))
        return min(1.0, det_area / player_area) ** 0.5
    return 0.0


def normalized_center_distance_score(det_box: list[float], player_box: list[float], max_ratio: float) -> float:
    dx1, dy1, dx2, dy2 = [float(v) for v in det_box]
    px1, py1, px2, py2 = [float(v) for v in player_box]
    dcx, dcy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    diag = max(1.0, float(np.hypot(px2 - px1, py2 - py1)))
    dist = float(np.hypot(dcx - pcx, dcy - pcy)) / diag
    if dist > max_ratio:
        return 0.0
    return 1.0 - dist / max(max_ratio, 1e-6)
