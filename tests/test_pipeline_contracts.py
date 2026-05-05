import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

from basketball_cv.court import CourtSpec, court_to_canvas


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
tools_pkg = types.ModuleType("tools")
tools_pkg.__path__ = [str(ROOT / "tools")]
sys.modules.setdefault("tools", tools_pkg)

from tools.render_possession import draw_person_record, draw_record_on_minimap, person_label
from tools.run_game_pipeline import collect_pipeline_metrics


def test_combined_wrapper_is_disabled():
    script = (ROOT / "tools" / "run_game_pipeline_combined_ocr.ps1").read_text(encoding="utf-8")
    assert "Legacy combined OCR pipeline has been removed." in script
    assert "tools\\run_game_pipeline.py" in script


def test_readme_points_to_single_official_pipeline():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python tools/run_game_pipeline.py" in readme
    assert "run_game_pipeline_combined_ocr" not in readme
    assert "continuidad visual" in readme
    assert "no cuentan como evidencia OCR" in readme


def test_render_possession_marks_estimated_players_in_frame_label_and_minimap():
    estimated = {
        "class_name": "person",
        "player_id": 7,
        "team": "team_a",
        "bbox": [40, 60, 120, 160],
        "court_x": 5.0,
        "court_y": 5.0,
        "is_estimated": True,
    }
    detected = {**estimated, "is_estimated": False}
    assert "est" in person_label(estimated)

    base = np.full((220, 180, 3), 32, dtype=np.uint8)
    estimated_frame = base.copy()
    detected_frame = base.copy()
    draw_person_record(estimated_frame, estimated)
    draw_person_record(detected_frame, detected)

    estimated_fill = estimated_frame[90:130, 70:100]
    detected_fill = detected_frame[90:130, 70:100]
    base_fill = base[90:130, 70:100]

    assert not np.array_equal(estimated_frame, base)
    assert np.array_equal(detected_fill, base_fill)
    assert not np.array_equal(estimated_fill, base_fill)

    minimap = np.zeros((420, 760, 3), dtype=np.uint8)
    spec = CourtSpec()
    draw_record_on_minimap(minimap, estimated, spec, (20, 190, 235), possession=None)

    pt = court_to_canvas(np.asarray([[5.0, 5.0]], dtype=np.float32), spec, pixels_per_meter=24, margin_px=18)[0]
    x, y = int(pt[0]), int(pt[1])
    assert int(minimap.sum()) > 0
    assert minimap[y, x].tolist() == [0, 0, 0]


def test_collect_pipeline_metrics_counts_unique_locked_players(tmp_path: Path):
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(
        json.dumps(
            [
                {"class_name": "person", "player_id": 2, "is_estimated": False},
                {"class_name": "person", "player_id": 2, "is_estimated": True},
                {"class_name": "person", "player_id": 5, "is_estimated": False},
            ]
        ),
        encoding="utf-8",
    )
    jersey_path = tmp_path / "jersey_numbers.json"
    jersey_path.write_text(
        json.dumps(
            {
                "players": [
                    {"player_id": 2, "jersey_locked": True},
                    {"player_id": 5, "jersey_locked": True},
                    {"player_id": 7, "jersey_locked": False},
                ],
                "identity_conflicts": [{"player_id": 5}],
            }
        ),
        encoding="utf-8",
    )

    metrics = collect_pipeline_metrics(tracks_path, jersey_path)
    assert metrics["players_with_locked_jersey"] == 2
    assert metrics["jersey_conflicts"] == 1
    assert metrics["estimated_player_frames"] == 1
    assert metrics["short_gap_fills"] == 1


def test_collect_pipeline_metrics_ignores_locked_players_without_renderable_jersey(tmp_path: Path):
    tracks_path = tmp_path / "tracks.json"
    tracks_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    jersey_path = tmp_path / "jersey_numbers.json"
    jersey_path.write_text(
        json.dumps(
            {
                "players": [
                    {"player_id": 2, "jersey_locked": True, "jersey_number": 23},
                    {"player_id": 5, "jersey_locked": True, "jersey_number": None},
                    {"player_id": 8, "jersey_number": 11},
                ],
                "identity_conflicts": [{"player_id": 5}],
            }
        ),
        encoding="utf-8",
    )

    metrics = collect_pipeline_metrics(tracks_path, jersey_path)

    assert metrics["players_with_locked_jersey"] == 2
    assert metrics["jersey_conflicts"] == 1


def test_render_possession_with_model_help_runs_as_script():
    script = ROOT / "tools" / "render_possession_with_model.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Render tracks using role classification" in result.stdout


def test_render_possession_with_model_help_accepts_no_minimap_flag():
    script = ROOT / "tools" / "render_possession_with_model.py"
    result = subprocess.run(
        [sys.executable, str(script), "--no-minimap", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--no-minimap" in result.stdout
