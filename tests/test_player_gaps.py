from basketball_cv.player_gaps import interpolate_player_gaps, smooth_player_positions


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


def test_interpolate_player_gaps_skips_when_competitor_is_too_close():
    records = [
        {"class_name": "person", "player_id": 4, "frame_index": 10, "team": "dark", "bbox": [100, 100, 140, 220], "anchor_px": [120, 220], "court_x": 2.0, "court_y": 3.0, "confidence": 0.91},
        {"class_name": "person", "player_id": 9, "frame_index": 11, "team": "dark", "bbox": [102, 102, 142, 222], "anchor_px": [122, 222], "court_x": 2.12, "court_y": 3.08, "confidence": 0.95},
        {"class_name": "person", "player_id": 4, "frame_index": 13, "team": "dark", "bbox": [112, 104, 152, 224], "anchor_px": [132, 224], "court_x": 2.3, "court_y": 3.4, "confidence": 0.93},
    ]
    report = interpolate_player_gaps(records, fps=25.0, max_gap_frames=4)
    assert report["estimated_player_frames"] == 0
    assert not any(rec.get("is_estimated") for rec in records)
