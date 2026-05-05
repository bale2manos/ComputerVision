from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_combined_wrapper_is_disabled():
    script = (ROOT / "tools" / "run_game_pipeline_combined_ocr.ps1").read_text(encoding="utf-8")
    assert "Legacy combined OCR pipeline has been removed." in script
    assert "tools\\run_game_pipeline.py" in script


def test_readme_points_to_single_official_pipeline():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python tools/run_game_pipeline.py" in readme
    assert "run_game_pipeline_combined_ocr" not in readme
