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


def test_export_possession_crops_help_mentions_task_and_manifest():
    script = ROOT / "tools" / "export_possession_crops.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout
    assert "--task" in result.stdout
    assert "ball-state" in result.stdout
    assert "owner-state" in result.stdout


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
