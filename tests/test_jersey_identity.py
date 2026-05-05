from basketball_cv.jersey_identity import JerseyIdentityConfig, resolve_player_identity


def test_two_digit_identity_locks_when_global_evidence_is_strong():
    votes = [
        {"frame_index": 0, "track_id": 2, "team": "dark", "number": "93", "ocr_confidence": 0.99, "variant": "rgb"},
        {"frame_index": 12, "track_id": 2, "team": "dark", "number": "93", "ocr_confidence": 0.96, "variant": "gray"},
        {"frame_index": 50, "track_id": 6002, "team": "dark", "number": "93", "ocr_confidence": 0.97, "variant": "rgb"},
        {"frame_index": 19, "track_id": 2, "team": "dark", "number": "0", "ocr_confidence": 0.81, "variant": "otsu"},
    ]
    result = resolve_player_identity(player_id=2, team="dark", votes=votes, config=JerseyIdentityConfig())
    assert result["canonical_jersey_number"] == "93"
    assert result["jersey_locked"] is True
    assert result["display_jersey_number"] == "93"


def test_conflicting_segment_does_not_overwrite_locked_global_number():
    votes = [
        {"frame_index": 0, "track_id": 6001, "team": "red", "number": "5", "ocr_confidence": 0.99, "variant": "rgb"},
        {"frame_index": 33, "track_id": 6001, "team": "red", "number": "5", "ocr_confidence": 0.96, "variant": "gray"},
        {"frame_index": 39, "track_id": 6001, "team": "red", "number": "15", "ocr_confidence": 0.92, "variant": "rgb"},
        {"frame_index": 61, "track_id": 6001, "team": "red", "number": "15", "ocr_confidence": 0.91, "variant": "gray"},
    ]
    result = resolve_player_identity(player_id=10, team="red", votes=votes, config=JerseyIdentityConfig())
    assert result["canonical_jersey_number"] == "5"
    assert any(segment["jersey_number"] == "15" for segment in result["segments"])
    assert any(segment["backfill_allowed"] is False for segment in result["segments"] if segment["jersey_number"] == "15")


def test_unresolved_identity_falls_back_to_player_id_display():
    votes = [
        {"frame_index": 0, "track_id": 14, "team": "dark", "number": "5", "ocr_confidence": 0.45, "variant": "rgb"},
        {"frame_index": 9, "track_id": 14, "team": "dark", "number": "7", "ocr_confidence": 0.44, "variant": "gray"},
    ]
    result = resolve_player_identity(player_id=14, team="dark", votes=votes, config=JerseyIdentityConfig())
    assert result["canonical_jersey_number"] is None
    assert result["jersey_locked"] is False
    assert result["display_jersey_number"] == "P14"
    assert all(segment["jersey_number"] is None for segment in result["segments"])
