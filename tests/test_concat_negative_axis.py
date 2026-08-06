"""
Regression test for esp-dl issue #318:
ESP-DL exporter crashes when ONNX Concat uses negative axis (-1) in YOLOv8 export.

ResetConcatPattern used ``var_perm.index(int(axis))`` without normalizing
negative axes, so exporting a model with Concat(axis=-1) raised
``ValueError: -1 is not in list`` during reset_graph_layout.
"""

import os
import re
import tempfile

import torch
import torch.nn as nn

from esp_ppq.api.espdl_interface import espdl_quantize_torch
from esp_ppq.api.setting import QuantizationSettingFactory


class ConvConcatNegAxisModel(nn.Module):
    """YOLOv8 style: two Conv outputs concatenated with a negative axis."""

    def __init__(self, channels: int, axis: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.axis = axis

    def forward(self, input1, input2):
        return torch.cat([self.conv1(input1), self.conv2(input2)], dim=self.axis)


class PlainConcatNegAxisModel(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, input1, input2):
        return torch.cat([input1, input2], dim=self.axis)


def _quantize_and_export(model, input_shape, export_file) -> str:
    dataset = [[torch.randn(*shape) for shape in input_shape] for _ in range(4)]
    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.dispatcher = 'allin'
    espdl_quantize_torch(
        model=model.eval(),
        espdl_export_file=export_file,
        calib_dataloader=dataset,
        calib_steps=4,
        input_shape=input_shape,
        target='esp32p4',
        num_of_bits=8,
        device='cpu',
        error_report=False,
        skip_export=False,
        export_config=True,
        verbose=0,
        setting=quant_setting,
    )
    info_file = os.path.splitext(export_file)[0] + '.info'
    assert os.path.exists(export_file), 'espdl model was not exported'
    assert os.path.exists(info_file), 'espdl info file was not exported'
    return info_file


def _exported_concat_axis(info_file: str) -> int:
    with open(info_file) as f:
        content = f.read()
    match = re.search(r'Concat\[axis = (-?\d+)', content)
    assert match, f'No Concat op found in {info_file}'
    return int(match.group(1))


def test_conv_concat_negative_axis_last_dim():
    """Issue #318 repro: Concat(axis=-1) on Conv outputs must export successfully.

    Before the fix this raised ValueError: -1 is not in list.
    NCHW axis -1 (=3, W) must be remapped to NHWC index 2.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = os.path.join(tmpdir, 'conv_concat_neg_axis.espdl')
        info_file = _quantize_and_export(
            ConvConcatNegAxisModel(channels=16, axis=-1),
            [[1, 16, 10, 10], [1, 16, 10, 10]],
            export_file,
        )
        assert _exported_concat_axis(info_file) == 2


def test_conv_concat_negative_axis_channel_dim():
    """Concat(axis=-3, NCHW channel dim) on Conv outputs must remap to NHWC index 3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = os.path.join(tmpdir, 'conv_concat_neg_axis_c.espdl')
        info_file = _quantize_and_export(
            ConvConcatNegAxisModel(channels=16, axis=-3),
            [[1, 16, 10, 10], [1, 16, 10, 10]],
            export_file,
        )
        assert _exported_concat_axis(info_file) == 3


def test_plain_concat_negative_axis():
    """Concat(axis=-1) directly on graph inputs must export a normalized positive axis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = os.path.join(tmpdir, 'plain_concat_neg_axis.espdl')
        info_file = _quantize_and_export(
            PlainConcatNegAxisModel(axis=-1),
            [[1, 5, 10, 10], [1, 5, 10, 31]],
            export_file,
        )
        assert _exported_concat_axis(info_file) == 3


def test_plain_concat_positive_axis_unchanged():
    """Positive axis must keep working (regression check for the fix)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_file = os.path.join(tmpdir, 'plain_concat_pos_axis.espdl')
        info_file = _quantize_and_export(
            PlainConcatNegAxisModel(axis=1),
            [[1, 5, 10, 10], [1, 3, 10, 10]],
            export_file,
        )
        assert _exported_concat_axis(info_file) == 1


if __name__ == '__main__':
    test_conv_concat_negative_axis_last_dim()
    test_conv_concat_negative_axis_channel_dim()
    test_plain_concat_negative_axis()
    test_plain_concat_positive_axis_unchanged()
    print('\nAll Concat negative-axis tests passed.')
