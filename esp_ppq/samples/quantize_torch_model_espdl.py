import argparse
from typing import Iterable

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_torch

BATCHSIZE = 16
INPUT_SHAPE = [3, 224, 224]
DEVICE = 'cpu'
ESPDL_MODEL_PATH = "mobilenet_v2.espdl"


class BaseInferencer:
    def __init__(self, args):
        # get quantization config.
        self.num_of_bits = args.bits
        self.target = args.target
        # Load a pretrained mobilenet v2 model
        self.model = torchvision.models.mobilenet.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model = self.model.to(DEVICE)

    def load_calibration_dataset(self) -> Iterable:
        return [torch.randn(size=INPUT_SHAPE) for _ in range(BATCHSIZE * 2)]

    def __call__(self):
        def collate_fn(batch: torch.Tensor) -> torch.Tensor:
            return batch.to(DEVICE)

        # create a setting for quantizing your network with ESPDL.
        quant_setting = QuantizationSettingFactory.espdl_setting()

        # Load training data for creating a calibration dataloader.
        calibration_dataset = self.load_calibration_dataset()
        calibration_dataloader = DataLoader(dataset=calibration_dataset, batch_size=BATCHSIZE, shuffle=False)

        # quantize your model.
        quant_ppq_graph = espdl_quantize_torch(
            model=self.model,
            espdl_export_file=ESPDL_MODEL_PATH,
            calib_dataloader=calibration_dataloader,
            calib_steps=16,
            input_shape=[1] + INPUT_SHAPE,
            target=self.target,
            num_of_bits=self.num_of_bits,
            collate_fn=collate_fn,
            setting=quant_setting,
            device=DEVICE,
            error_report=True,
            skip_export=False,
            export_test_values=True,
            verbose=1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="esp32p4",
        help="esp32p4, esp32s3 or c, (defaults: esp32p4).",
    )
    parser.add_argument(
        "-b",
        "--bits",
        type=int,
        default=8,
        help="the number of bits, support 8 or 16, (defaults: 8).",
    )
    args = parser.parse_args()
    BaseInferencer(args)()
