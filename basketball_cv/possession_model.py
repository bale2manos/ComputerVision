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


CONTROL_CLASSES = {"control", "has_ball", "player_with_ball", "dribble", "dribbling", "shoot", "shooting", "pass", "passing"}
LOOSE_CLASSES = {"loose", "loose_ball", "contested", "divided", "no_control", "none"}


@dataclass
class PossessionModelConfig:
    crop_margin: float = 0.35
    max_candidates: int = 6
    min_model_confidence: float = 0.58
    min_control_margin: float = 0.10
    candidate_radius_m: float = 8.0
    candidate_min_contact: float = 0.05
    image_candidate_radius_px: float = 260.0
    min_ball_confidence: float = 0.04
    suppress_non_active_balls: bool = True


class PossessionClassifier:
    """Optional classifier for player-ball interaction states.

    The expected model is a YOLO classification model trained on crops containing
    a candidate player and the detected ball. Supported class names are flexible:
    control/has_ball/player_with_ball/dribble/shoot/pass are treated as control;
    loose/contested/no_control are treated as non-control.
    """

    def __init__(self, model_path: str | Path, config: PossessionModelConfig | None = None) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is required to use --possession-model")
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.config = config or PossessionModelConfig()
        self.names = getattr(self.model, "names", {}) or {}

    def apply_to_records(self, records: list[dict[str, Any]], video_path: str | Path, fps: float) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for possession model: {video_path}")

        by_frame: dict[int, list[dict[str, Any]]] = {}
        for rec in records:
            by_frame.setdefault(int(rec.get("frame_index", 0)), []).append(rec)

        applied = 0
        model_control_frames = 0
        model_loose_frames = 0
        no_candidate_frames = 0
        active_ball_replacements = 0
        current_frame = -1
        frame_img: np.ndarray | None = None

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
            ball = choose_active_ball(frame_records, self.config)
            if ball is None:
                continue
            if self.config.suppress_non_active_balls:
                suppress_other_balls(frame_records, ball)
            if ball.get("active_ball_selected_by_model"):
                active_ball_replacements += 1
            candidates = select_candidates(ball, frame_records, self.config)
            if not candidates:
                no_candidate_frames += 1
                continue

            predictions = []
            for player in candidates:
                crop = make_player_ball_crop(frame_img, player, ball, self.config.crop_margin)
                if crop is None or crop.size == 0:
                    continue
                pred = self.predict_crop(crop)
                if pred is None:
                    continue
                pred["player"] = player
                pred["player_id"] = _player_key(player)
                pred["team"] = player.get("team")
                pred["jersey_number"] = player.get("jersey_number")
                pred["image_contact"] = round(float(ball_player_image_contact(ball, player)), 3)
                pred["distance_m"] = round(float(_distance(ball, player)), 3)
                pred["image_distance_px"] = round(float(ball_player_image_distance_px(ball, player)), 2)
                predictions.append(pred)

            if not predictions:
                continue
            ball["possession_model_candidates"] = serialize_predictions(predictions)
            winner = choose_model_winner(predictions, self.config)
            if winner is None:
                best = max(predictions, key=lambda item: float(item.get("confidence", 0.0)))
                if str(best.get("state", "")).lower() in LOOSE_CLASSES and float(best.get("confidence", 0.0)) >= self.config.min_model_confidence:
                    clear_frame_possession(frame_records)
                    ball["ball_state"] = "loose"
                    ball["ball_owner_assignment_reason"] = "model_loose"
                    model_loose_frames += 1
                    applied += 1
                continue
            reassign_owner_from_model(frame_records, ball, winner)
            applied += 1
            model_control_frames += 1

        cap.release()
        return {
            "possession_model": str(self.model_path),
            "frames_with_model_output": applied,
            "model_control_frames": model_control_frames,
            "model_loose_frames": model_loose_frames,
            "no_candidate_frames": no_candidate_frames,
            "active_ball_replacements": active_ball_replacements,
            "classes": self.names,
            "parameters": self.config.__dict__,
        }

    def predict_crop(self, crop: np.ndarray) -> dict[str, Any] | None:
        results = self.model.predict(crop, verbose=False)
        if not results:
            return None
        probs = getattr(results[0], "probs", None)
        if probs is None:
            return None
        top1 = int(probs.top1)
        conf = float(probs.top1conf)
        name = str(self.names.get(top1, top1))
        return {"state": name, "class_id": top1, "confidence": round(conf, 4)}


def choose_active_ball(frame_records: list[dict[str, Any]], config: PossessionModelConfig) -> dict[str, Any] | None:
    balls = [
        rec
        for rec in frame_records
        if rec.get("class_name") == "sports ball"
        and not str(rec.get("source", "")).startswith("dense_ball_")
        and float(rec.get("confidence") or 0.0) >= config.min_ball_confidence
    ]
    if not balls:
        return best_non_dense_ball(frame_records)
    players = _eligible_players(frame_records)
    if not players:
        return max(balls, key=lambda rec: float(rec.get("confidence") or 0.0))

    def score_ball(ball: dict[str, Any]) -> float:
        best_contact = max((ball_player_image_contact(ball, player) for player in players), default=0.0)
        best_img = min((ball_player_image_distance_px(ball, player) for player in players), default=9999.0)
        owner_bonus = 0.25 if ball.get("ball_owner_player_id") is not None else 0.0
        source_bonus = 0.2 if ball.get("source") == "ball_model" else 0.0
        conf = float(ball.get("confidence") or 0.0)
        return 1.45 * best_contact + 0.55 * max(0.0, 1.0 - best_img / max(config.image_candidate_radius_px, 1.0)) + 0.20 * conf + owner_bonus + source_bonus

    selected = max(balls, key=score_ball)
    if selected is not best_non_dense_ball(frame_records):
        selected["active_ball_selected_by_model"] = True
    selected["active_ball_score"] = round(float(score_ball(selected)), 4)
    return selected


def suppress_other_balls(frame_records: list[dict[str, Any]], active_ball: dict[str, Any]) -> None:
    for rec in frame_records:
        if rec is active_ball or rec.get("class_name") != "sports ball":
            continue
        if str(rec.get("source", "")).startswith("dense_ball_"):
            continue
        rec["suppressed_ball_candidate"] = True
        rec["ball_state"] = "suppressed_duplicate"
        rec.pop("ball_owner_player_id", None)
        rec.pop("ball_owner_identity", None)
        rec.pop("ball_owner_jersey_number", None)
        rec.pop("ball_owner_team", None)
        rec.pop("ball_owner_confidence", None)
        rec.pop("ball_owner_assignment_reason", None)


def select_candidates(ball: dict[str, Any], frame_records: list[dict[str, Any]], config: PossessionModelConfig) -> list[dict[str, Any]]:
    candidates = []
    for player in _eligible_players(frame_records):
        dist = _distance(ball, player)
        contact = ball_player_image_contact(ball, player)
        image_dist = ball_player_image_distance_px(ball, player)
        if dist <= config.candidate_radius_m or contact >= config.candidate_min_contact or image_dist <= config.image_candidate_radius_px:
            candidates.append((contact, -image_dist, -dist, player))
    candidates.sort(key=lambda item: item[:3], reverse=True)
    return [item[3] for item in candidates[: config.max_candidates]]


def ball_player_image_distance_px(ball: dict[str, Any], player: dict[str, Any]) -> float:
    ball_box = ball.get("bbox")
    player_box = player.get("bbox")
    if not isinstance(ball_box, list) or not isinstance(player_box, list):
        return 9999.0
    bx1, by1, bx2, by2 = [float(v) for v in ball_box]
    px1, py1, px2, py2 = [float(v) for v in player_box]
    bc_x = (bx1 + bx2) / 2.0
    bc_y = (by1 + by2) / 2.0
    nearest_x = min(max(bc_x, px1), px2)
    nearest_y = min(max(bc_y, py1), py2)
    return float(np.hypot(bc_x - nearest_x, bc_y - nearest_y))


def make_player_ball_crop(frame: np.ndarray, player: dict[str, Any], ball: dict[str, Any], margin: float = 0.35) -> np.ndarray | None:
    boxes = []
    for rec in (player, ball):
        box = rec.get("bbox")
        if isinstance(box, list) and len(box) == 4:
            boxes.append([float(v) for v in box])
    if not boxes:
        return None
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    pad = margin * max(w, h)
    x1 = max(0, int(round(x1 - pad)))
    y1 = max(0, int(round(y1 - pad)))
    x2 = min(frame.shape[1], int(round(x2 + pad)))
    y2 = min(frame.shape[0], int(round(y2 + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def choose_model_winner(predictions: list[dict[str, Any]], config: PossessionModelConfig) -> dict[str, Any] | None:
    control = [pred for pred in predictions if str(pred.get("state", "")).lower() in CONTROL_CLASSES]
    if not control:
        return None
    control.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            float(item.get("image_contact", 0.0)),
            -float(item.get("image_distance_px", 9999.0)),
        ),
        reverse=True,
    )
    best = control[0]
    if float(best.get("confidence", 0.0)) < config.min_model_confidence:
        return None
    if len(control) > 1:
        second = control[1]
        if float(best["confidence"]) < float(second["confidence"]) + config.min_control_margin:
            if float(best.get("image_contact", 0.0)) < float(second.get("image_contact", 0.0)) + 0.12:
                if float(best.get("image_distance_px", 9999.0)) > float(second.get("image_distance_px", 9999.0)) + 35.0:
                    return None
    return best


def clear_frame_possession(frame_records: list[dict[str, Any]]) -> None:
    for rec in frame_records:
        if rec.get("class_name") == "person":
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
            ):
                rec.pop(key, None)


def reassign_owner_from_model(frame_records: list[dict[str, Any]], ball: dict[str, Any], prediction: dict[str, Any]) -> None:
    player = prediction["player"]
    clear_frame_possession(frame_records)
    owner_id = _player_key(player)
    if owner_id is None:
        return
    contact = ball_player_image_contact(ball, player)
    distance = _distance(ball, player)
    identity = _record_identity(player)
    confidence = float(prediction.get("confidence", 0.0))
    state = str(prediction.get("state", "control"))

    ball["ball_state"] = "owned"
    ball["ball_owner_player_id"] = owner_id
    ball["ball_owner_identity"] = identity
    ball["ball_owner_jersey_number"] = player.get("jersey_number")
    ball["ball_owner_team"] = player.get("team")
    ball["ball_owner_distance_m"] = round(float(distance), 3)
    ball["ball_owner_image_contact"] = round(float(contact), 3)
    ball["ball_owner_confidence"] = round(confidence, 3)
    ball["ball_owner_assignment_reason"] = f"model_{state}"
    ball["ball_owner_image_distance_px"] = prediction.get("image_distance_px")
    if player.get("court_x") is not None and player.get("court_y") is not None:
        ball["possession_court_x"] = player.get("court_x")
        ball["possession_court_y"] = player.get("court_y")

    player["has_ball"] = True
    player["ball_state"] = "owned"
    player["ball_owner_player_id"] = owner_id
    player["ball_owner_identity"] = identity
    player["ball_owner_jersey_number"] = player.get("jersey_number")
    player["ball_owner_team"] = player.get("team")
    player["ball_owner_distance_m"] = round(float(distance), 3)
    player["ball_owner_image_contact"] = round(float(contact), 3)
    player["ball_owner_confidence"] = round(confidence, 3)
    player["ball_owner_assignment_reason"] = f"model_{state}"
    player["ball_owner_source"] = ball.get("source")
    player["ball_owner_image_distance_px"] = prediction.get("image_distance_px")


def serialize_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for pred in predictions:
        item = {k: v for k, v in pred.items() if k != "player"}
        output.append(item)
    return output
