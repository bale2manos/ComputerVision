from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TEMPLATE_CONFIG = r"""
# PaddleOCR jersey recognizer fine-tuning template.
# This file is generated as a starting point. If your installed PaddleOCR version
# uses slightly different keys, copy these dataset paths into one of PaddleOCR's
# official rec configs, e.g. PP-OCRv4_mobile_rec.yml / PP-OCRv5_mobile_rec.yml.

Global:
  debug: false
  use_gpu: {use_gpu}
  epoch_num: {epochs}
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {save_model_dir}
  save_epoch_step: 5
  eval_batch_step: [0, 200]
  cal_metric_during_train: true
  pretrained_model: {pretrained_model}
  checkpoints:
  save_inference_dir: {save_inference_dir}
  use_visualdl: false
  infer_img:
  character_dict_path: {dict_path}
  max_text_length: 2
  infer_mode: false
  use_space_char: false

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: !!float {learning_rate}
  regularizer:
    name: L2
    factor: 0.00001

Architecture:
  model_type: rec
  algorithm: CRNN
  Transform:
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: small
    small_stride: [1, 2, 2, 2]
  Neck:
    name: SequenceEncoder
    encoder_type: rnn
    hidden_size: 96
  Head:
    name: CTCHead
    fc_decay: 0.00001

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: {data_dir}
    label_file_list:
      - {train_list}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
{train_aug_block}      - CTCLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 160]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: true
    batch_size_per_card: {batch_size}
    drop_last: true
    num_workers: {num_workers}

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {data_dir}
    label_file_list:
      - {val_list}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: false
      - CTCLabelEncode:
      - RecResizeImg:
          image_shape: [3, 48, 160]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: {batch_size}
    num_workers: {num_workers}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare commands/config for fine-tuning an adapted OCR recognizer for jersey digits. "
            "Run tools/prepare_jersey_ocr_dataset.py before this."
        )
    )
    parser.add_argument("--data", default="datasets/jersey_ocr_paddle")
    parser.add_argument("--output-dir", default="runs/jersey_ocr_paddle")
    parser.add_argument("--pretrained-model", default="", help="Optional PaddleOCR pretrained recognition checkpoint prefix.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-gpu", action="store_true", help="Set Global.use_gpu=true in the generated config.")
    parser.add_argument("--no-rec-aug", action="store_true", help="Remove RecAug from the training transforms. Useful for a small first experiment.")
    parser.add_argument("--config-output", default=None)
    parser.add_argument("--run", action="store_true", help="Actually call PaddleOCR train.py using --paddleocr-dir.")
    parser.add_argument("--export", action="store_true", help="After training, call PaddleOCR export_model.py.")
    parser.add_argument("--paddleocr-dir", default="", help="Path to a local PaddleOCR clone. Required for --run/--export.")
    parser.add_argument("--best-model", default=None, help="Model checkpoint prefix for export. Defaults to output_dir/best_accuracy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_list = data_dir / "train_list.txt"
    val_list = data_dir / "val_list.txt"
    dict_path = data_dir / "digit_dict.txt"
    for path in (train_list, val_list, dict_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run tools/prepare_jersey_ocr_dataset.py first.")

    config_path = Path(args.config_output).resolve() if args.config_output else output_dir / "jersey_rec_config.yml"
    save_model_dir = output_dir / "checkpoints"
    save_inference_dir = output_dir / "inference"
    pretrained = args.pretrained_model or ""
    train_aug_block = "" if args.no_rec_aug else "      - RecAug:\n"
    learning_rate = f"{float(args.learning_rate):.10f}".rstrip("0").rstrip(".")

    config = TEMPLATE_CONFIG.format(
        use_gpu=str(bool(args.use_gpu)).lower(),
        epochs=args.epochs,
        save_model_dir=as_posix(save_model_dir),
        save_inference_dir=as_posix(save_inference_dir),
        pretrained_model=pretrained,
        dict_path=as_posix(dict_path),
        learning_rate=learning_rate,
        data_dir=as_posix(data_dir),
        train_list=as_posix(train_list),
        val_list=as_posix(val_list),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_aug_block=train_aug_block,
    )
    config_path.write_text(config + "\n", encoding="utf-8")

    print("Config written:", config_path)
    print("\nRecommended flow:")
    print("1) Prepare reviewed OCR dataset:")
    print('   python tools\\prepare_jersey_ocr_dataset.py --input-root "datasets\\jersey_ocr_labeled" --output-root "datasets\\jersey_ocr_paddle" --val-ratio 0.15')
    print("2) Fine-tune PaddleOCR recognition model from a local PaddleOCR clone:")
    print(f'   cd "<PaddleOCR>"')
    print(f'   python tools/train.py -c "{config_path}"')
    print("3) Export inference model:")
    print(f'   python tools/export_model.py -c "{config_path}" -o Global.pretrained_model="{save_model_dir / "best_accuracy"}" Global.save_inference_dir="{save_inference_dir}"')
    print("4) Use it in this project after extract_jersey_numbers.py is wired for PaddleOCR inference:")
    print(f'   python tools\\extract_jersey_numbers.py --ocr-engine both --paddle-rec-model-dir "{save_inference_dir}" --paddle-rec-char-dict "{dict_path}" ...')

    if args.run or args.export:
        if not args.paddleocr_dir:
            raise SystemExit("--paddleocr-dir is required with --run or --export")
        paddle_dir = Path(args.paddleocr_dir).resolve()
        if not paddle_dir.exists():
            raise SystemExit(f"PaddleOCR dir not found: {paddle_dir}")
        if args.run:
            subprocess.run([sys.executable, "tools/train.py", "-c", str(config_path)], cwd=paddle_dir, check=True)
        if args.export:
            best_model = Path(args.best_model).resolve() if args.best_model else save_model_dir / "best_accuracy"
            subprocess.run(
                [
                    sys.executable,
                    "tools/export_model.py",
                    "-c",
                    str(config_path),
                    "-o",
                    f"Global.pretrained_model={best_model}",
                    f"Global.save_inference_dir={save_inference_dir}",
                ],
                cwd=paddle_dir,
                check=True,
            )


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


if __name__ == "__main__":
    main()
