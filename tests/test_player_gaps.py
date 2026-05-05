import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from basketball_cv.player_gaps import interpolate_player_gaps, smooth_player_positions


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
tools_pkg = types.ModuleType("tools")
tools_pkg.__path__ = [str(ROOT / "tools")]
sys.modules.setdefault("tools", tools_pkg)


@contextmanager
def _load_analyze_video_module():
    sentinel = object()
    original_modules = {
        name: sys.modules.get(name, sentinel)
        for name in ("cv2", "ultralytics", "tools.analyze_video")
    }
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules.setdefault("ultralytics", ultralytics)
    sys.modules.pop("tools.analyze_video", None)
    try:
        yield importlib.import_module("tools.analyze_video")
    finally:
        for name, original_module in original_modules.items():
            if original_module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_interpolate_player_gaps_creates_estimated_records_for_short_gap():
    records = [
        {"class_name": "person", "player_id": 4, "frame_index": 10, "team": "dark", "bbox": [100, 100, 140, 220], "anchor_px": [120, 220], "court_x": 2.0, "court_y": 3.0, "confidence": 0.91},
        {"class_name": "person", "player_id": 4, "frame_index": 13, "team": "dark", "bbox": [112, 104, 152, 224], "anchor_px": [132, 224], "court_x": 2.3, "court_y": 3.4, "confidence": 0.93},
    ]
    report = interpolate_player_gaps(records, fps=25.0, max_gap_frames=4)
    estimated = [rec for rec in records if rec.get("is_estimated")]
    assert report["estimated_player_frames"] == 2
    assert [rec["frame_index"] for rec in estimated] == [11, 12]


def test_smooth_player_positions_writes_smoothed_court_fields():
    records = [
        {"class_name": "person", "player_id": 7, "frame_index": 0, "court_x": 1.0, "court_y": 1.0},
        {"class_name": "person", "player_id": 7, "frame_index": 1, "court_x": 1.6, "court_y": 1.4},
        {"class_name": "person", "player_id": 7, "frame_index": 2, "court_x": 1.2, "court_y": 1.1},
    ]
    smooth_player_positions(records, window=3)
    assert all("court_x_smooth" in rec and "court_y_smooth" in rec for rec in records)
    assert records[1]["court_x_smooth"] == 1.267
    assert records[1]["court_y_smooth"] == 1.167


def test_interpolate_player_gaps_skips_when_competitor_is_too_close():
    records = [
        {"class_name": "person", "player_id": 4, "frame_index": 10, "team": "dark", "bbox": [100, 100, 140, 220], "anchor_px": [120, 220], "court_x": 2.0, "court_y": 3.0, "confidence": 0.91},
        {"class_name": "person", "player_id": 9, "frame_index": 11, "team": "dark", "bbox": [102, 102, 142, 222], "anchor_px": [122, 222], "court_x": 2.12, "court_y": 3.08, "confidence": 0.95},
        {"class_name": "person", "player_id": 4, "frame_index": 13, "team": "dark", "bbox": [112, 104, 152, 224], "anchor_px": [132, 224], "court_x": 2.3, "court_y": 3.4, "confidence": 0.93},
    ]
    report = interpolate_player_gaps(records, fps=25.0, max_gap_frames=4)
    assert report["estimated_player_frames"] == 0
    assert not any(rec.get("is_estimated") for rec in records)


def test_smooth_player_positions_does_not_blend_across_long_gaps():
    records = [
        {"class_name": "person", "player_id": 8, "frame_index": 0, "court_x": 0.0, "court_y": 0.0},
        {"class_name": "person", "player_id": 8, "frame_index": 1, "court_x": 0.0, "court_y": 0.0},
        {"class_name": "person", "player_id": 8, "frame_index": 100, "court_x": 10.0, "court_y": 10.0},
        {"class_name": "person", "player_id": 8, "frame_index": 101, "court_x": 10.0, "court_y": 10.0},
    ]
    smooth_player_positions(records, window=3)
    assert [rec["court_x_smooth"] for rec in records] == [0.0, 0.0, 10.0, 10.0]
    assert [rec["court_y_smooth"] for rec in records] == [0.0, 0.0, 10.0, 10.0]


def test_load_analyze_video_module_restores_sys_modules(monkeypatch):
    sentinel = object()
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    monkeypatch.delitem(sys.modules, "tools.analyze_video", raising=False)
    helper_entry_modules = {
        name: sys.modules.get(name, sentinel)
        for name in ("cv2", "ultralytics", "tools.analyze_video")
    }

    with _load_analyze_video_module() as analyze_video:
        assert analyze_video is sys.modules["tools.analyze_video"]
        assert "cv2" in sys.modules
        assert "ultralytics" in sys.modules

    for name, original_module in helper_entry_modules.items():
        if original_module is sentinel:
            assert name not in sys.modules
        else:
            assert sys.modules.get(name) is original_module


def test_refresh_player_summary_preserves_fallback_when_no_player_ids_exist():
    with _load_analyze_video_module() as analyze_video:
        fallback_report = {
            "enabled": False,
            "player_count": 2,
            "merged_track_count": 0,
            "players": [
                {"player_id": 1, "track_ids": [11]},
                {"player_id": 2, "track_ids": [14]},
            ],
        }
        records = [
            {"class_name": "person", "track_id": 11, "frame_index": 0, "team": "dark"},
            {"class_name": "person", "track_id": 14, "frame_index": 1, "team": "light"},
        ]

        refreshed = analyze_video.refresh_player_summary(records, fallback_report, crossing_report={"correction_count": 0})

    assert refreshed["player_count"] == 2
    assert refreshed["players"] == fallback_report["players"]
