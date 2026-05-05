import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
tools_pkg = types.ModuleType("tools")
tools_pkg.__path__ = [str(ROOT / "tools")]
sys.modules.setdefault("tools", tools_pkg)

from tools.render_tracks import apply_jersey_numbers, backfill_jersey_identities_across_fragments


def test_apply_jersey_numbers_backfills_only_compatible_segments(tmp_path: Path):
    report = {
        "players": [
            {
                "player_id": 2,
                "team": "dark",
                "jersey_number": "93",
                "canonical_jersey_number": "93",
                "jersey_locked": True,
            }
        ],
        "identity_segments": [
            {"track_id": 2, "team": "dark", "jersey_number": "93", "jersey_identity": "dark_93", "backfill_allowed": True},
            {"track_id": 6002, "team": "dark", "jersey_number": "2", "jersey_identity": "dark_2", "backfill_allowed": False},
        ],
        "identity_conflicts": [],
    }
    report_path = tmp_path / "jersey_numbers.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    records = [
        {"class_name": "person", "player_id": 2, "track_id": 6002, "frame_index": 0, "team": "dark", "court_x": 2.0, "court_y": 3.0},
        {"class_name": "person", "player_id": 2, "track_id": 2, "frame_index": 6, "team": "dark", "court_x": 2.2, "court_y": 3.1},
    ]
    apply_jersey_numbers(records, report_path)
    backfills = backfill_jersey_identities_across_fragments(records, fps=25.0, min_candidate_frames=1)

    by_track = {rec["track_id"]: rec for rec in records}
    assert backfills == 0
    assert by_track[2]["jersey_number"] == "93"
    assert by_track[6002].get("jersey_number") == "2"
    assert by_track[6002].get("identity_conflict") is not None
    assert by_track[6002].get("identity_source") == "ocr_track_segment"
    assert by_track[6002].get("identity_source") != "jersey_backfill_motion"


def test_apply_jersey_numbers_keeps_legacy_render_ready_reports(tmp_path: Path):
    report = {
        "players": [
            {
                "player_id": 7,
                "team": "dark",
                "jersey_number": "42",
                "jersey_identity": "dark_42",
            }
        ],
        "identity_segments": [],
        "identity_conflicts": [],
    }
    report_path = tmp_path / "legacy_jersey_numbers.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    records = [{"class_name": "person", "player_id": 7, "track_id": 77, "frame_index": 10, "team": "dark"}]
    apply_jersey_numbers(records, report_path)

    assert records[0]["jersey_number"] == "42"
    assert records[0]["jersey_identity"] == "dark_42"


def test_conflicting_segment_cannot_become_backfill_source(tmp_path: Path):
    report = {
        "players": [],
        "identity_segments": [
            {"track_id": 91, "team": "dark", "jersey_number": "72", "jersey_identity": "dark_72", "backfill_allowed": False},
        ],
        "identity_conflicts": [],
    }
    report_path = tmp_path / "conflict_source.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    records = [
        {"class_name": "person", "player_id": 3, "track_id": 12, "frame_index": 0, "team": "dark", "court_x": 2.0, "court_y": 3.0},
        {"class_name": "person", "player_id": 3, "track_id": 91, "frame_index": 6, "team": "dark", "court_x": 2.1, "court_y": 3.1},
    ]
    apply_jersey_numbers(records, report_path)
    backfills = backfill_jersey_identities_across_fragments(records, fps=25.0, min_candidate_frames=1)

    by_track = {rec["track_id"]: rec for rec in records}
    assert backfills == 0
    assert by_track[91]["jersey_number"] == "72"
    assert by_track[91].get("identity_conflict") == "segment_backfill_blocked"
    assert by_track[12].get("identity_source") != "jersey_backfill_motion"
    assert by_track[12].get("jersey_number") is None
