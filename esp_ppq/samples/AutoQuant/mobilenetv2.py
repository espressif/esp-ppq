"""MobileNetV2 AutoQuant sample using ESP-DL assets.

The calibration dataset and the pretrained MobileNetV2 ONNX model are
downloaded automatically. Run with::

    python -m esp_ppq.samples.AutoQuant.mobilenetv2
"""

import os
import sys
import urllib.request
import zipfile

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from esp_ppq.api import (
    AutoQuantSearchSetting,
    espdl_auto_quantize_onnx,
)
from esp_ppq.IR import BaseGraph

ONNX_PATH = "./quantize_mobilenetv2/mobilenet_v2.onnx"
ESPDL_PATH = "outputs/mobilenetv2/model.espdl"
RUN_DIR = "outputs/mobilenetv2"

ESPDL_RAW = "https://raw.githubusercontent.com/espressif/esp-dl/master"
ONNX_URL = f"{ESPDL_RAW}/examples/tutorial/how_to_quantize_model/quantize_mobilenetv2/models/torch/mobilenet_v2.onnx"

IMAGENET_URL = "https://dl.espressif.com/public/imagenet_calib.zip"
ZIP_NAME = "imagenet_calib.zip"
CALIB_ROOT = "./quantize_mobilenetv2/imagenet"
CALIB_DIR = os.path.join(CALIB_ROOT, "calib")

INPUT_SHAPE = [3, 224, 224]
BATCHSIZE = 32
CALIB_STEPS = 32
CALIB_LIMIT = 1024

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
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path, reporthook=_report_hook)
    print()


def prepare_calibration_dataset() -> None:
    os.makedirs(CALIB_ROOT, exist_ok=True)
    if not os.path.exists(CALIB_DIR):
        _download(IMAGENET_URL, ZIP_NAME)
        with zipfile.ZipFile(ZIP_NAME, "r") as zip_file:
            zip_file.extractall(CALIB_ROOT)


def prepare_onnx_model() -> None:
    _download(ONNX_URL, ONNX_PATH)


def _imagenet_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_calib_dataloader() -> DataLoader:
    """Build a calibration dataloader that yields image tensors only."""
    prepare_calibration_dataset()
    dataset = datasets.ImageFolder(CALIB_DIR, _imagenet_transform())
    if CALIB_LIMIT and len(dataset) > CALIB_LIMIT:
        dataset = Subset(dataset, indices=list(range(CALIB_LIMIT)))

    def collate(batch):
        return torch.stack([sample[0] for sample in batch], dim=0)

    return DataLoader(
        dataset=dataset,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate,
    )


def build_eval_dataloader() -> DataLoader:
    """Build a validation dataloader with labels."""
    prepare_calibration_dataset()
    dataset = datasets.ImageFolder(CALIB_DIR, _imagenet_transform())
    return DataLoader(
        dataset=dataset,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )


def build_evaluate_fn():
    """Build the AutoQuant evaluation callback."""
    here = os.path.dirname(os.path.abspath(__file__))
    util_path = os.path.normpath(os.path.join(here, "..", "Imagenet"))
    if util_path not in sys.path:
        sys.path.insert(0, util_path)
    from Utilities.Imagenet import evaluate_ppq_module_with_imagenet

    eval_loader = build_eval_dataloader()

    def evaluate_fn(quant_graph: BaseGraph):
        df = evaluate_ppq_module_with_imagenet(
            model=quant_graph,
            imagenet_validation_loader=eval_loader,
            device=DEVICE,
            verbose=False,
        )
        top1 = float(df["top1_accuracy"].mean())
        top5 = float(df["top5_accuracy"].mean())
        return top1, {"top1": top1, "top5": top5}

    return evaluate_fn


def main() -> None:
    prepare_onnx_model()

    setting = AutoQuantSearchSetting(
        search_mode="exhaustive",
        num_of_candidates=NUM_OF_CANDIDATES,
        run_dir=RUN_DIR,
        resume=False,
    )

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
            f"top1={c.get('top1'):.4f}  top5={c.get('top5'):.4f}  "
            f"folder={c.get('folder')}"
        )


if __name__ == "__main__":
    main()
