# Two-Model Possession Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the possession weak-label workflow with a manifest-driven two-step annotation flow, two dataset export tasks, and task-aware training for separate ball-state and owner-state models.

**Architecture:** Keep one canonical annotation truth source in a JSONL manifest under `datasets/possession_labels`, move validation/export logic into a pure `basketball_cv.possession_dataset` module, and keep the CLI scripts thin wrappers around those pure helpers. The first phase stops at `annotate -> export -> train` and does not integrate the new models into the runtime possession stack yet.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, Ultralytics YOLO classification, pytest, JSONL/CSV manifests.

---

## File Structure

**Create**
- `F:/ComputerVision/basketball_cv/possession_dataset.py`
- `F:/ComputerVision/tests/test_possession_dataset.py`
- `F:/ComputerVision/tests/test_possession_dataset_cli.py`

**Modify**
- `F:/ComputerVision/tools/annotate_possession_dataset.py`
- `F:/ComputerVision/tools/export_possession_crops.py`
- `F:/ComputerVision/tools/train_possession_classifier.py`

The new module owns manifest validation, split assignment, and crop export helpers. The scripts stay small and mostly parse args, call helpers, and report results.

### Task 1: Add a Pure Manifest and Export Helper Module

**Files:**
- Create: `F:/ComputerVision/tests/test_possession_dataset.py`
- Create: `F:/ComputerVision/basketball_cv/possession_dataset.py`

- [ ] **Step 1: Write the failing manifest/export tests**

Create `F:/ComputerVision/tests/test_possession_dataset.py`:

```python
import json
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_possession_dataset.py -v
```

Expected:

```text
FAILED tests/test_possession_dataset.py::test_build_manifest_row_rejects_owner_for_air
E   ModuleNotFoundError: No module named 'basketball_cv.possession_dataset'
```

- [ ] **Step 3: Implement the pure helper module**

Create `F:/ComputerVision/basketball_cv/possession_dataset.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


BALL_STATE_CLASSES = ("owned", "air", "loose")
OWNER_STATE_CLASSES = ("control", "dribble", "shot", "contested")
QUALITY_FLAGS = {
    "uncertain",
    "occluded",
    "ball_not_visible_cleanly",
    "candidate_missing",
    "reviewed_in_cvat",
}


def build_manifest_row(...):
    row = {...}
    validate_manifest_row(row)
    return row


def validate_manifest_row(row: dict[str, Any]) -> None:
    ...


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    ...


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    ...


def assign_manifest_split(row: dict[str, Any], val_video_stems: set[str], val_ratio: float) -> str:
    ...


def export_manifest_dataset(path: Path, output_dir: Path, task: str, max_negatives_per_frame: int = 4, ball_margin: float = 1.2, owner_margin: float = 0.35) -> dict[str, Any]:
    ...
```

Implementation details:
- `validate_manifest_row()` enforces:
  - `ball_state` is one of `owned/air/loose`
  - `owner_*` fields are all `null` unless `ball_state == "owned"`
  - `owner_state` is one of `control/dribble/shot/contested` when owned
  - owned rows must reference a candidate by `player_id` or `track_id`
  - `flags` only uses known labels
- `append_manifest_row()` writes one JSON line with UTF-8 and trailing newline.
- `load_manifest_rows()` returns rows in file order and injects `source_manifest_row` starting at `1` when missing.
- `export_manifest_dataset()` opens each source video, seeks to `frame_index`, and writes:
  - `ball-state`: one ball-centered crop per manifest row
  - `owner-state`: one positive crop for owner plus nearby negative candidates labeled `no_control`
- The export writes its own derived `manifest.jsonl` under `output_dir`.

- [ ] **Step 4: Re-run the tests**

Run:

```powershell
python -m pytest tests/test_possession_dataset.py -v
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

Run:

```powershell
git add tests/test_possession_dataset.py basketball_cv/possession_dataset.py
git commit -m "feat: add possession dataset manifest helpers"
```

### Task 2: Refactor the Annotator to Write Canonical Manifest Rows

**Files:**
- Modify: `F:/ComputerVision/tools/annotate_possession_dataset.py`
- Create: `F:/ComputerVision/tests/test_possession_dataset_cli.py`

- [ ] **Step 1: Write the failing annotator CLI test**

Add to `F:/ComputerVision/tests/test_possession_dataset_cli.py`:

```python
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_annotate_possession_dataset_help_mentions_manifest_and_ball_states():
    script = ROOT / "tools" / "annotate_possession_dataset.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
    assert "owned / air / loose" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests/test_possession_dataset_cli.py::test_annotate_possession_dataset_help_mentions_manifest_and_ball_states -v
```

Expected:

```text
FAILED tests/test_possession_dataset_cli.py::test_annotate_possession_dataset_help_mentions_manifest_and_ball_states
```

- [ ] **Step 3: Rewrite the annotator around a two-step state machine**

In `F:/ComputerVision/tools/annotate_possession_dataset.py`:

```python
parser.add_argument("--manifest", default="datasets/possession_labels/manifest.jsonl")
parser.add_argument("--split-hint", choices=["train", "val", "auto"], default="auto")
```

Replace the old class-folder workflow with:

```python
state = {
    ...,
    "pending_ball_state": None,
    "pending_owner_index": None,
    "pending_flags": set(),
}
```

Implement these handlers:

```python
def save_ball_state_row(state: dict[str, Any], ball_state: str) -> None:
    row = build_manifest_row(
        video=str(state["video"]),
        frame_index=int(state["last_frame"]),
        split_hint=choose_split_hint(state),
        ball_state=ball_state,
        ball_bbox=state["last_ball"].get("bbox"),
        candidate_players=serialize_candidates(state["last_candidates"]),
        flags=sorted(state["pending_flags"]),
    )
    append_manifest_row(state["manifest_path"], row)


def save_owned_row(state: dict[str, Any], candidate_idx: int, owner_state: str) -> None:
    player = state["last_candidates"][candidate_idx]
    row = build_manifest_row(
        video=str(state["video"]),
        frame_index=int(state["last_frame"]),
        split_hint=choose_split_hint(state),
        ball_state="owned",
        ball_bbox=state["last_ball"].get("bbox"),
        candidate_players=serialize_candidates(state["last_candidates"]),
        owner_player_id=player.get("player_id"),
        owner_track_id=player.get("track_id"),
        owner_team=player.get("team"),
        owner_state=owner_state,
        flags=sorted(state["pending_flags"]),
    )
    append_manifest_row(state["manifest_path"], row)
```

Key behavior:
- `o` enters owned flow.
- `a` and `l` save immediately as `air` and `loose`.
- `1-9` or click chooses owner candidate only after `o`.
- `c/d/s/t` finalizes the owned row after a candidate is selected.
- `f/g/b/m` toggle `uncertain/occluded/ball_not_visible_cleanly/candidate_missing`.
- panel text reflects the current sub-step.

- [ ] **Step 4: Re-run the test**

Run:

```powershell
python -m pytest tests/test_possession_dataset_cli.py::test_annotate_possession_dataset_help_mentions_manifest_and_ball_states -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

Run:

```powershell
git add tools/annotate_possession_dataset.py tests/test_possession_dataset_cli.py
git commit -m "feat: add two-step possession manifest annotator"
```

### Task 3: Make Export Task-Aware and Manifest-Driven

**Files:**
- Modify: `F:/ComputerVision/tools/export_possession_crops.py`
- Modify: `F:/ComputerVision/tests/test_possession_dataset_cli.py`

- [ ] **Step 1: Write the failing exporter CLI test**

Add to `F:/ComputerVision/tests/test_possession_dataset_cli.py`:

```python
def test_export_possession_crops_help_mentions_task_and_manifest():
    script = ROOT / "tools" / "export_possession_crops.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
    assert "--task" in result.stdout
    assert "ball-state" in result.stdout
    assert "owner-state" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests/test_possession_dataset_cli.py::test_export_possession_crops_help_mentions_task_and_manifest -v
```

Expected:

```text
FAILED tests/test_possession_dataset_cli.py::test_export_possession_crops_help_mentions_task_and_manifest
```

- [ ] **Step 3: Replace track-based export with manifest-driven export**

In `F:/ComputerVision/tools/export_possession_crops.py`:

```python
parser.add_argument("--manifest", required=True)
parser.add_argument("--task", required=True, choices=["ball-state", "owner-state"])
parser.add_argument("--output-dir", default=None)
parser.add_argument("--val-video-stems", default="")
parser.add_argument("--val-ratio", type=float, default=0.15)
parser.add_argument("--max-negatives-per-frame", type=int, default=4)
```

Then call:

```python
report = export_manifest_dataset(
    Path(args.manifest),
    output_dir,
    task=args.task,
    max_negatives_per_frame=args.max_negatives_per_frame,
    val_video_stems={item.strip() for item in args.val_video_stems.split(",") if item.strip()},
    val_ratio=args.val_ratio,
)
```

Default output dirs:
- `datasets/possession_ball_state`
- `datasets/possession_owner_state`

Print:

```python
print(json.dumps(report, indent=2))
```

- [ ] **Step 4: Re-run the CLI tests**

Run:

```powershell
python -m pytest tests/test_possession_dataset.py tests/test_possession_dataset_cli.py::test_export_possession_crops_help_mentions_task_and_manifest -v
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit**

Run:

```powershell
git add tools/export_possession_crops.py tests/test_possession_dataset_cli.py basketball_cv/possession_dataset.py tests/test_possession_dataset.py
git commit -m "feat: export possession datasets from manifest"
```

### Task 4: Make the Trainer Task-Aware

**Files:**
- Modify: `F:/ComputerVision/tools/train_possession_classifier.py`
- Modify: `F:/ComputerVision/tests/test_possession_dataset_cli.py`

- [ ] **Step 1: Write the failing training-layout test**

Add to `F:/ComputerVision/tests/test_possession_dataset_cli.py`:

```python
def test_train_possession_classifier_show_layout_mentions_both_tasks():
    script = ROOT / "tools" / "train_possession_classifier.py"
    result = subprocess.run(
        [sys.executable, str(script), "--task", "ball-state", "--show-layout"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "datasets/possession_ball_state" in result.stdout
    assert "datasets/possession_owner_state" in result.stdout
    assert "owned" in result.stdout
    assert "no_control" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests/test_possession_dataset_cli.py::test_train_possession_classifier_show_layout_mentions_both_tasks -v
```

Expected:

```text
FAILED tests/test_possession_dataset_cli.py::test_train_possession_classifier_show_layout_mentions_both_tasks
```

- [ ] **Step 3: Extend the trainer CLI**

In `F:/ComputerVision/tools/train_possession_classifier.py`:

```python
parser.add_argument("--task", required=True, choices=["ball-state", "owner-state"])
parser.add_argument("--data", default=None)
parser.add_argument("--base-model", default=None)
```

Resolve defaults:

```python
TASK_DEFAULTS = {
    "ball-state": {"data": "datasets/possession_ball_state", "base_model": "yolo11s-cls.pt"},
    "owner-state": {"data": "datasets/possession_owner_state", "base_model": "yolo11s-cls.pt"},
}
```

Update `--show-layout` text to print both dataset layouts and the classes for each task.

- [ ] **Step 4: Re-run the CLI tests**

Run:

```powershell
python -m pytest tests/test_possession_dataset_cli.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

Run:

```powershell
git add tools/train_possession_classifier.py tests/test_possession_dataset_cli.py
git commit -m "feat: add task-aware possession classifier training"
```

## Self-Review

- Spec coverage:
  - canonical manifest: Task 1 and Task 2
  - two-step annotator: Task 2
  - two export tasks: Task 1 and Task 3
  - task-aware trainer: Task 4
  - validation and split isolation: Task 1 and Task 3
- Placeholder scan:
  - no `TODO/TBD` markers
  - each task includes exact files, commands, and concrete code targets
- Type consistency:
  - `ball-state` and `owner-state` task names are used consistently
  - `owned/air/loose` and `control/dribble/shot/contested/no_control` stay aligned with the spec
