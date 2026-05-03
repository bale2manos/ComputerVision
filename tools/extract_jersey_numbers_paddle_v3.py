from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PADDLEOCR_ROOT = ROOT / "external" / "PaddleOCR"
if PADDLEOCR_ROOT.exists() and str(PADDLEOCR_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLEOCR_ROOT))

import extract_jersey_numbers as base


class DirectPaddleRecognizer:
    def __init__(self, model_dir: Path, char_dict: Path, use_gpu: bool = False) -> None:
        try:
            from tools.infer import utility
            from tools.infer.predict_rec import TextRecognizer
        except ImportError as exc:
            raise SystemExit(
                "PaddleOCR inference modules are not importable. Run with paddle_venv and install the local clone if needed: "
                "python -m pip install -e F:\\ComputerVision\\external\\PaddleOCR"
            ) from exc

        args = utility.init_args().parse_args([])
        args.rec_model_dir = str(model_dir)
        args.rec_char_dict_path = str(char_dict)
        args.rec_algorithm = "CRNN"
        args.rec_image_shape = "3,48,160"
        args.rec_batch_num = 8
        args.max_text_length = 2
        args.use_space_char = False
        args.use_gpu = bool(use_gpu)
        args.use_xpu = False
        args.use_npu = False
        args.use_mlu = False
        args.use_metax_gpu = False
        args.use_gcu = False
        args.use_tensorrt = False
        args.use_onnx = False
        args.benchmark = False
        args.return_word_box = False
        args.show_log = False
        args.enable_mkldnn = False
        args.cpu_threads = 4
        self.recognizer = TextRecognizer(args)
        print(f"[paddle-rec] Python: {sys.executable}")
        print(f"[paddle-rec] model_dir={model_dir}")
        print(f"[paddle-rec] char_dict={char_dict}")

    def recognize(self, image: np.ndarray) -> list[tuple[str, float]]:
        try:
            output = self.recognizer([image])
        except Exception:
            return []
        rec_res = output[0] if isinstance(output, tuple) else output
        pairs: list[tuple[str, float]] = []
        if isinstance(rec_res, list):
            for item in rec_res:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        pairs.append((str(item[0]), float(item[1])))
                    except Exception:
                        pass
        return pairs


def build_paddle_reader_direct(args: argparse.Namespace) -> DirectPaddleRecognizer:
    model_dir = Path(args.paddle_rec_model_dir) if args.paddle_rec_model_dir else None
    char_dict = Path(args.paddle_rec_char_dict) if args.paddle_rec_char_dict else None
    if model_dir is None or not model_dir.exists():
        raise SystemExit("--paddle-rec-model-dir is required and must exist.")
    if char_dict is None or not char_dict.exists():
        raise SystemExit("--paddle-rec-char-dict is required and must exist.")
    return DirectPaddleRecognizer(model_dir, char_dict, bool(getattr(args, "paddle_use_gpu", False)))


def read_digits_paddle_direct(reader: DirectPaddleRecognizer, variants: list[tuple[str, np.ndarray]], min_confidence: float) -> list[dict[str, Any]]:
    votes: list[dict[str, Any]] = []
    for variant_name, image in variants:
        for text, confidence in reader.recognize(image):
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


def main() -> None:
    base.build_paddle_reader = build_paddle_reader_direct
    base.read_digits_paddle = read_digits_paddle_direct
    base.main()


if __name__ == "__main__":
    main()
