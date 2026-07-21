"""YOLO11n AutoQuant sample using ESP-DL assets.

The calibration dataset, ONNX model, and quantized-model validator are
downloaded automatically.

Requirements::

    pip install ultralytics
"""

import importlib.util
import os
import urllib.request
import zipfile
from typing import Dict, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from esp_ppq.api import (
    AutoQuantSearchSetting,
    espdl_auto_quantize_onnx,
)
from esp_ppq.executor import TorchExecutor
from esp_ppq.IR import BaseGraph

ONNX_PATH = "./quantize_yolo11n/yolo11n.onnx"
PT_PATH = "yolo11n.pt"
DATA_CFG = "coco.yaml"

ESPDL_RAW = "https://raw.githubusercontent.com/espressif/esp-dl/master"
ONNX_URL = f"{ESPDL_RAW}/models/coco_detect/models/yolo11n.onnx"
EVAL_URL = f"{ESPDL_RAW}/examples/tutorial/how_to_quantize_model/quantize_yolo11n/yolo11n_eval.py"
EVAL_PATH = "./quantize_yolo11n/yolo11n_eval.py"

CALIB_URL = "https://dl.espressif.com/public/calib_yolo11n.zip"
ZIP_NAME = "calib_yolo11n.zip"
CALIB_ROOT = "./quantize_yolo11n"
CALIB_DIR = os.path.join(CALIB_ROOT, "calib_yolo11n")

ESPDL_PATH = "outputs/yolo11n/model.espdl"
RUN_DIR = "outputs/yolo11n"

IMG_SIZE = 640
INPUT_SHAPE = [3, IMG_SIZE, IMG_SIZE]
BATCHSIZE = 32
CALIB_STEPS = 32

TARGET = "esp32p4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_OF_CANDIDATES = 5


def _report_hook(blocknum, blocksize, total):
    if total <= 0:
        return
    downloaded = blocknum * blocksize
    percent = min(downloaded / total * 100, 100)
    print(f"\rDownloading: {percent:.2f}%", end="")


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        return
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path, reporthook=_report_hook)
    print()


def prepare_calibration_dataset() -> None:
    os.makedirs(CALIB_ROOT, exist_ok=True)
    if not os.path.exists(CALIB_DIR):
        _download(CALIB_URL, ZIP_NAME)
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_file:
            zip_file.extractall(CALIB_ROOT)


def prepare_onnx_model() -> None:
    _download(ONNX_URL, ONNX_PATH)


def prepare_eval_code() -> None:
    _download(EVAL_URL, EVAL_PATH)


class _YoloCalibDataset(Dataset):
    """Image directory dataset used for calibration."""

    def __init__(self, path: str, img_size: int = 640) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Calibration dir not found: {path}")
        self.paths = [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {path}")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_calib_dataloader() -> DataLoader:
    prepare_calibration_dataset()
    dataset = _YoloCalibDataset(CALIB_DIR, img_size=IMG_SIZE)
    return DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=False)


def _import_quant_validator_factory():
    """Import the validator implementation downloaded from ESP-DL."""
    prepare_eval_code()
    spec = importlib.util.spec_from_file_location("yolo11n_eval", EVAL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load validator from {EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_quant_validator_class


def build_evaluate_fn():
    from ultralytics import YOLO

    make_quant_validator_class = _import_quant_validator_factory()

    def evaluate_fn(quant_graph: BaseGraph) -> Tuple[float, Dict[str, float]]:
        executor = TorchExecutor(graph=quant_graph, device=DEVICE)
        QuantValidator = make_quant_validator_class(executor)

        model = YOLO(PT_PATH)
        results = model.val(
            data=DATA_CFG,
            split="val",
            imgsz=IMG_SIZE,
            device=DEVICE,
            validator=QuantValidator,
            rect=False,
            save_json=False,
        )
        map50 = float(results.box.map50)
        map5095 = float(results.box.map)
        return map5095, {"mAP50": map50, "mAP50-95": map5095}

    return evaluate_fn


def main() -> None:
    prepare_onnx_model()
    prepare_eval_code()

    setting = AutoQuantSearchSetting(
        search_mode="fast",
        num_of_candidates=NUM_OF_CANDIDATES,
        run_dir=RUN_DIR,
        resume=False,
    )
    # fast-only knobs (honored only when search_mode == "fast").
    setting.top_strategy = 10
    setting.early_stop_patience = 5
    # Search space is initialized inside __init__; adjust entries as needed.
    setting.strategy_space["mixed_precision"] = [True, False]

    topk = espdl_auto_quantize_onnx(
        onnx_import_file=ONNX_PATH,
        espdl_export_file=ESPDL_PATH,
        calib_dataloader=build_calib_dataloader(),
        calib_steps=CALIB_STEPS,
        input_shape=INPUT_SHAPE,
        evaluate_fn=build_evaluate_fn(),
        target=TARGET,
        setting=setting,
        device=DEVICE,
        verbose=0,
    )

    print("\n=== Top-K candidates ===")
    for c in topk:
        print(
            f"#{c.get('index'):04d}  score={c.get('score'):.4f}  "
            f"mAP50-95={c.get('mAP50-95'):.4f}  mAP50={c.get('mAP50'):.4f}  "
            f"folder={c.get('folder')}"
        )


if __name__ == "__main__":
    main()
