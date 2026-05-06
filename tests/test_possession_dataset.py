from pathlib import Path

import cv2
import numpy as np

from basketball_cv.possession_dataset import (
    append_manifest_row,
    assign_manifest_split,
    build_manifest_row,
    export_manifest_dataset,
    load_manifest_rows,
)


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (80, 60))
    assert writer.isOpened()
    frame = np.full((60, 80, 3), 220, dtype=np.uint8)
    cv2.rectangle(frame, (28, 18), (36, 26), (0, 140, 255), -1)
    cv2.rectangle(frame, (12, 12), (32, 54), (255, 0, 0), 2)
    cv2.rectangle(frame, (42, 10), (65, 54), (0, 255, 0), 2)
    writer.write(frame)
    writer.release()


def test_build_manifest_row_rejects_owner_for_air():
    try:
        build_manifest_row(
            video="clip.mp4",
            frame_index=10,
            split_hint="train",
            ball_state="air",
            ball_bbox=[28, 18, 36, 26],
            candidate_players=[],
            owner_player_id=7,
        )
    except ValueError as exc:
        assert "owner" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_manifest_row_requires_owned_state_to_reference_candidate():
    try:
        build_manifest_row(
            video="clip.mp4",
            frame_index=10,
            split_hint="train",
            ball_state="owned",
            ball_bbox=[28, 18, 36, 26],
            candidate_players=[{"player_id": 11, "track_id": 11, "team": "dark", "bbox": [12, 12, 32, 54], "rank": 0}],
            owner_player_id=7,
            owner_track_id=7,
            owner_team="dark",
            owner_state="control",
        )
    except ValueError as exc:
        assert "candidate" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_assign_manifest_split_prefers_split_hint():
    row = {"video": "F:/ComputerVision/videos/subra.mp4", "split_hint": "val"}
    assert assign_manifest_split(row, set(), 0.15) == "val"


def test_export_manifest_dataset_writes_ball_and_owner_tasks(tmp_path: Path):
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path)
    manifest_path = tmp_path / "manifest.jsonl"
    append_manifest_row(
        manifest_path,
        build_manifest_row(
            video=str(video_path),
            frame_index=0,
            split_hint="train",
            ball_state="owned",
            ball_bbox=[28, 18, 36, 26],
            candidate_players=[
                {"player_id": 11, "track_id": 11, "team": "dark", "bbox": [12, 12, 32, 54], "rank": 0, "jersey_number": "11"},
                {"player_id": 22, "track_id": 22, "team": "red", "bbox": [42, 10, 65, 54], "rank": 1, "jersey_number": "22"},
            ],
            owner_player_id=11,
            owner_track_id=11,
            owner_team="dark",
            owner_state="control",
        ),
    )

    ball_out = tmp_path / "ball_state"
    owner_out = tmp_path / "owner_state"
    ball_report = export_manifest_dataset(manifest_path, ball_out, task="ball-state")
    owner_report = export_manifest_dataset(manifest_path, owner_out, task="owner-state", max_negatives_per_frame=2)

    assert ball_report["exported"] == 1
    assert owner_report["exported"] == 2
    assert any((ball_out / "train" / "owned").glob("*.jpg"))
    assert any((owner_out / "train" / "control").glob("*.jpg"))
    assert any((owner_out / "train" / "no_control").glob("*.jpg"))
    exported_rows = load_manifest_rows(owner_out / "manifest.jsonl")
    assert exported_rows[0]["source_manifest_row"] == 1


def test_export_manifest_dataset_can_skip_flagged_rows(tmp_path: Path):
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path)
    manifest_path = tmp_path / "manifest.jsonl"
    append_manifest_row(
        manifest_path,
        build_manifest_row(
            video=str(video_path),
            frame_index=0,
            split_hint="train",
            ball_state="air",
            ball_bbox=[28, 18, 36, 26],
            candidate_players=[],
            flags=["uncertain"],
        ),
    )

    output_dir = tmp_path / "ball_state"
    report = export_manifest_dataset(manifest_path, output_dir, task="ball-state", exclude_flags={"uncertain"})

    assert report["exported"] == 0
    assert not any(output_dir.rglob("*.jpg"))
