from basketball_cv.events import assign_ball_ownership, detect_pick_and_rolls


def test_assign_ball_ownership_ignores_estimated_players():
    records = [
        {"class_name": "sports ball", "frame_index": 10, "court_x": 5.0, "court_y": 5.0, "confidence": 0.95},
        {"class_name": "person", "frame_index": 10, "player_id": 4, "track_id": 4, "team": "dark", "court_x": 5.05, "court_y": 5.02, "confidence": 0.98, "in_play_player": True, "is_estimated": True},
    ]
    report = assign_ball_ownership(records, fps=25.0)

    assert report["owned_ball_frames"] == 0
    assert records[0].get("ball_owner_player_id") is None


def test_detect_pick_and_rolls_ignores_estimated_players():
    records = [
        {"class_name": "sports ball", "frame_index": 0, "court_x": 5.0, "court_y": 5.0, "confidence": 0.95},
        {"class_name": "person", "frame_index": 0, "player_id": 1, "track_id": 1, "team": "dark", "court_x": 5.0, "court_y": 5.0, "confidence": 0.98, "in_play_player": True, "is_estimated": True},
        {"class_name": "person", "frame_index": 0, "player_id": 2, "track_id": 2, "team": "dark", "court_x": 5.8, "court_y": 5.0, "confidence": 0.98, "in_play_player": True},
        {"class_name": "person", "frame_index": 0, "player_id": 3, "track_id": 3, "team": "red", "court_x": 5.9, "court_y": 5.0, "confidence": 0.98, "in_play_player": True},
    ]

    events = detect_pick_and_rolls(records, fps=4.0)
    assert events == []
