from __future__ import annotations

"""PaddleOCR v3-compatible wrapper for adapted jersey OCR.

This script reuses the existing jersey crop sampling/reporting logic from
``tools.extract_jersey_numbers`` but replaces PaddleOCR initialization and
inference with a version that works with newer PaddleOCR releases, where args
such as ``use_gpu``/``show_log``/``det`` are no longer accepted by ``PaddleOCR``.

It also auto-adds ``external/PaddleOCR`` to ``sys.path``. This matters when
PaddleOCR was cloned locally for training but not installed as a Python package
in the active environment.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PADDLEOCR = ROOT / "external" / "PaddleOCR"
if LOCAL_PADDLEOCR.exists() and str(LOCAL_PADDLEOCR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PADDLEOCR))

import extract_jersey_numbers as base


def build_paddle_reader_v3(args: argparse.Namespace) -> Any:
    model_dir = Path(args.paddle_rec_model_dir) if args.paddle_rec_model_dir else None
    char_dict = Path(args.paddle_rec_char_dict) if args.paddle_rec_char_dict else None
    if model_dir is None or not model_dir.exists():
        raise SystemExit("--paddle-rec-model-dir is required and must exist.")
    if char_dict is None or not char_dict.exists():
        raise SystemExit("--paddle-rec-char-dict is required and must exist.")

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        env = sys.executable
        raise SystemExit(
            "PaddleOCR is not importable in this Python environment.\n"
            f"Python: {env}\n"
            f"Tried local clone: {LOCAL_PADDLEOCR}\n"
            "Fix: activate paddle_venv and either run from the repo root after git pull, "
            "or install the local clone with: python -m pip install -e external\\PaddleOCR"
        ) from exc

    device = "gpu:0" if getattr(args, "paddle_use_gpu", False) else "cpu"

    # PaddleOCR changed parameter names across versions. Try modern v3 args first,
    # then compatibility variants. v3 rejects use_gpu/show_log/det/rec/cls.
    attempts = [
        {
            "text_recognition_model_dir": str(model_dir),
            "text_rec_char_dict_path": str(char_dict),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": device,
        },
        {
            "text_recognition_model_dir": str(model_dir),
            "text_recognition_char_dict_path": str(char_dict),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": device,
        },
        {
            "rec_model_dir": str(model_dir),
            "rec_char_dict_path": str(char_dict),
            "use_angle_cls": False,
            "lang": "en",
        },
        {
            "rec_model_dir": str(model_dir),
            "rec_char_dict_path": str(char_dict),
        },
    ]

    errors: list[str] = []
    for kwargs in attempts:
        try:
            reader = PaddleOCR(**kwargs)
            print(f"[paddle-v3] Python: {sys.executable}")
            print(f"[paddle-v3] Initialized PaddleOCR with args: {sorted(kwargs.keys())}")
            return reader
        except Exception as exc:
            errors.append(f"{sorted(kwargs.keys())}: {exc}")

    raise SystemExit("Could not initialize PaddleOCR v3 with adapted model. Attempts:\n" + "\n".join(errors))


def read_digits_paddle_v3(reader: Any, variants: list[tuple[str, np.ndarray]], min_confidence: float) -> list[dict[str, Any]]:
    votes: list[dict[str, Any]] = []
    for variant_name, image in variants:
        result = None
        calls = [
            ("predict", lambda: reader.predict(image)),
            ("ocr_det_false_cls_false", lambda: reader.ocr(image, det=False, cls=False)),
            ("ocr_det_false", lambda: reader.ocr(image, det=False)),
            ("ocr", lambda: reader.ocr(image)),
        ]
        for _name, call in calls:
            try:
                result = call()
                break
            except Exception:
                continue
        if result is None:
            continue

        for text, confidence in extract_text_conf_pairs_v3(result):
            for match in base.DIGIT_RE.findall(str(text)):
                number = base.normalize_number(match)
                if number is None or float(confidence) < min_confidence:
                    continue
                votes.append(
                    {
                        "number": number,
                        "text": str(text),
                        "ocr_confidence": round(float(confidence), 4),
                        "variant": variant_name,
                        "ocr_source": "paddle",
                    }
                )
    return votes


def extract_text_conf_pairs_v3(result: Any) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    seen_ids: set[int] = set()

    def maybe_add(text: Any, conf: Any) -> None:
        if text is None or conf is None:
            return
        try:
            pairs.append((str(text), float(conf)))
        except Exception:
            pass

    def walk(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen_ids:
            return
        seen_ids.add(obj_id)

        for attr in ("json", "to_dict", "dict"):
            if hasattr(obj, attr):
                try:
                    value = getattr(obj, attr)
                    value = value() if callable(value) else value
                    walk(value)
                except Exception:
                    pass

        if isinstance(obj, dict):
            maybe_add(obj.get("text") or obj.get("rec_text") or obj.get("label"), obj.get("confidence") or obj.get("score") or obj.get("rec_score"))

            texts = obj.get("rec_texts") or obj.get("texts")
            scores = obj.get("rec_scores") or obj.get("scores")
            if isinstance(texts, list) and isinstance(scores, list):
                for text, score in zip(texts, scores):
                    maybe_add(text, score)

            if "res" in obj:
                walk(obj["res"])
            for value in obj.values():
                if isinstance(value, (dict, list, tuple)) or hasattr(value, "json"):
                    walk(value)
            return

        if isinstance(obj, (list, tuple)):
            if len(obj) >= 2 and isinstance(obj[0], str) and isinstance(obj[1], (float, int, np.floating)):
                maybe_add(obj[0], obj[1])
                return
            if len(obj) >= 2 and isinstance(obj[1], (list, tuple)) and len(obj[1]) >= 2 and isinstance(obj[1][0], str):
                maybe_add(obj[1][0], obj[1][1])
                return
            for item in obj:
                walk(item)
            return

        if hasattr(obj, "__dict__"):
            try:
                walk(vars(obj))
            except Exception:
                pass

    walk(result)
    return pairs


def main() -> None:
    base.build_paddle_reader = build_paddle_reader_v3
    base.read_digits_paddle = read_digits_paddle_v3
    base.extract_text_conf_pairs = extract_text_conf_pairs_v3
    base.main()


if __name__ == "__main__":
    main()
