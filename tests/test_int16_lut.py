"""
Tests for the ESP-DL INT16 LUT activation optimization and export pass.

Covered behavior:

  * EspdlLUTFusionPass rewrites unary y=f(x) activations (Swish/SiLU, Sigmoid,
    Tanh, ...) into ``LUT`` ops on INT16 ESP-DL platforms, and is enabled by
    default through ``QuantizationSetting.lut_optimization``.
  * The pass is a no-op on INT8 (and when disabled via the setting).
  * The ``step`` knob exposed in ``QuantizationSetting.lut_optimization_setting``
    flows down to every fused op.
  * The bit-exact HardwareEmulator matches a reference reimplementation of the
    esp-dl LUT kernel (direct / nearest-neighbour / interpolation paths), which
    is the whole point: Python simulation aligns with the device.
  * Fusing changes the simulated output (LUT discretization) vs. the plain
    quantized activation.
  * Export (AddLUTPattern) emits a LUT table per fused op, named
    ``LUT_lut_{index}``, and the table is built from ``original_op_type`` +
    ``int16_lut_step`` (i.e. it equals the true activation table, not identity).
"""

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from esp_ppq import QuantizationSettingFactory
from esp_ppq.api.espdl_interface import espdl_quantize_torch
from esp_ppq.core import RoundingPolicy
from esp_ppq.executor import TorchExecutor
from esp_ppq.executor.op.torch.espdl_lut import HardwareEmulator
from esp_ppq.parser.espdl.espdl_typedef import ExporterPatternInfo
from esp_ppq.parser.espdl.export_patterns import AddLUTPattern
from esp_ppq.utils.round import ppq_tensor_round


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class LUTNet(nn.Module):
    """Conv -> SiLU(Swish) -> Conv -> Sigmoid -> Tanh."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.act1 = nn.SiLU()
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.act2 = nn.Sigmoid()
        self.act3 = nn.Tanh()

    def forward(self, x):
        x = self.act1(self.c1(x))
        x = self.act2(self.c2(x))
        x = self.act3(x)
        return x


_INPUT_SHAPE = [1, 3, 16, 16]


def _quantize(num_of_bits=16, setting=None, skip_export=True, export_file=None):
    torch.manual_seed(0)
    model = LUTNet().eval()
    calib = [torch.randn(_INPUT_SHAPE[1:]) for _ in range(8)]
    from torch.utils.data import DataLoader

    dl = DataLoader(calib, batch_size=2)
    if setting is None:
        setting = QuantizationSettingFactory.espdl_setting()
    if export_file is None:
        export_file = os.path.join(tempfile.gettempdir(), "int16_lut_test.espdl")
    graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=export_file,
        calib_dataloader=dl,
        calib_steps=4,
        input_shape=_INPUT_SHAPE,
        target="esp32p4",
        num_of_bits=num_of_bits,
        collate_fn=lambda b: b.to("cpu"),
        setting=setting,
        device="cpu",
        error_report=False,
        skip_export=skip_export,
        verbose=0,
    )
    return graph


def _lut_ops(graph):
    return [op for op in graph.operations.values() if op.type == "LUT"]


# ---------------------------------------------------------------------------
# Reference reimplementation of dl::module::LUT::forward (QUANT_TYPE_SYMM_16BIT)
# ---------------------------------------------------------------------------
def _c_reference(q_int16: np.ndarray, table: np.ndarray, step: int) -> np.ndarray:
    tsz = table.shape[0]
    is_pow2 = step > 1 and (step & (step - 1)) == 0
    idx = q_int16.astype(np.int64) + 32768  # signed -> unsigned

    if step == 1:
        out = table[np.clip(idx, 0, tsz - 1)]
    elif is_pow2:
        # PIE8 SIMD nearest-neighbour, half-even rounding.
        nn_idx = np.clip(np.round(idx / step).astype(np.int64), 0, tsz - 1)
        out = table[nn_idx]
    else:
        base = np.clip(idx // step, 0, tsz - 2)
        rem = idx % step
        x = table[base].astype(np.int64)
        y = table[base + 1].astype(np.int64)
        num = rem * (y - x)
        interp = np.where(num >= 0, num // step, -((-num) // step))  # C trunc toward zero
        out = x + interp
    return np.clip(out, -32768, 32767)


# ===================================================================
# Test 1: default fusion on INT16
# ===================================================================
def test_lut_fusion_default_int16():
    graph = _quantize(num_of_bits=16)
    lut_ops = _lut_ops(graph)

    fused_types = sorted(op.attributes.get("original_op_type") for op in lut_ops)
    assert fused_types == ["Sigmoid", "Swish", "Tanh"], f"Unexpected fused types: {fused_types}"

    for op in lut_ops:
        assert op.attributes.get("int16_lut_step") == 32
        assert len(op.inputs) == 1 and len(op.outputs) == 1

    # Conv ops must NOT be fused (they carry parameters).
    assert any(op.type == "Conv" for op in graph.operations.values())
    print("PASS: test_lut_fusion_default_int16")


# ===================================================================
# Test 2: no fusion on INT8
# ===================================================================
def test_lut_no_fusion_int8():
    graph = _quantize(num_of_bits=8)
    assert not _lut_ops(graph), "LUT fusion must not run on INT8 platforms"
    print("PASS: test_lut_no_fusion_int8")


# ===================================================================
# Test 3: fusion can be disabled through the setting
# ===================================================================
def test_lut_fusion_disabled():
    setting = QuantizationSettingFactory.espdl_setting()
    setting.lut_optimization = False
    graph = _quantize(num_of_bits=16, setting=setting)
    assert not _lut_ops(graph), "LUT fusion should be disabled via setting.lut_optimization=False"
    print("PASS: test_lut_fusion_disabled")


# ===================================================================
# Test 4: the step setting flows down to every fused op
# ===================================================================
def test_lut_step_setting():
    setting = QuantizationSettingFactory.espdl_setting()
    setting.lut_optimization_setting.step = 64
    graph = _quantize(num_of_bits=16, setting=setting)
    lut_ops = _lut_ops(graph)
    assert lut_ops, "Expected LUT ops"
    assert all(op.attributes.get("int16_lut_step") == 64 for op in lut_ops)
    print("PASS: test_lut_step_setting")


# ===================================================================
# Test 5: bit-exact emulation vs the esp-dl C kernel (all step regimes)
# ===================================================================
def test_lut_emulator_bit_exact():
    in_scale, out_scale = 2.0**-9, 2.0**-15
    rounding = RoundingPolicy.ROUND_HALF_EVEN

    def math_fn(op, values):
        return torch.nn.functional.silu(values[0])

    def build_table(step):
        n = 65536 // step + 1
        pivots = (torch.arange(0, n, dtype=torch.float32) * step - 32768) * in_scale
        out = math_fn(None, [pivots])
        return ppq_tensor_round(out / out_scale, rounding).clamp(-32768, 32767).to(torch.int32).numpy()

    q = np.arange(-32768, 32768, dtype=np.int64)
    x = torch.tensor(q.astype(np.float64) * in_scale)

    for step in (1, 32, 64, 256, 100):
        HardwareEmulator._table_cache.clear()
        y = HardwareEmulator.apply(
            x, math_fn, object(), torch.tensor(in_scale), torch.tensor(out_scale), step, rounding
        )
        y_int = torch.round(y.double() / out_scale).to(torch.int64).numpy()
        ref = _c_reference(q, build_table(step), step)
        mism = int((y_int != ref).sum())
        assert mism == 0, f"step={step}: {mism} mismatches vs C reference"
    print("PASS: test_lut_emulator_bit_exact")


# ===================================================================
# Test 6: fusing changes the simulated output (LUT discretization active)
# ===================================================================
def test_lut_simulation_changes_output():
    setting_off = QuantizationSettingFactory.espdl_setting()
    setting_off.lut_optimization = False
    graph_ideal = _quantize(num_of_bits=16, setting=setting_off)
    graph_lut = _quantize(num_of_bits=16)  # default: fusion on

    assert not _lut_ops(graph_ideal)
    assert _lut_ops(graph_lut)

    torch.manual_seed(123)
    x = torch.randn(_INPUT_SHAPE)
    y_ideal = TorchExecutor(graph=graph_ideal, device="cpu")(x)[0]
    y_lut = TorchExecutor(graph=graph_lut, device="cpu")(x)[0]

    assert y_ideal.shape == y_lut.shape
    max_diff = (y_ideal - y_lut).abs().max().item()
    assert max_diff > 0, "LUT emulation should change the simulated output vs. plain activation"
    print(f"PASS: test_lut_simulation_changes_output (max|diff|={max_diff:.3e})")


# ===================================================================
# Test 7: export emits a correct LUT table per fused op
# ===================================================================
def test_lut_export_tables():
    export_file = os.path.join(tempfile.gettempdir(), "int16_lut_export.espdl")
    info_file = export_file.rsplit(".", 1)[0] + ".info"
    for f in (export_file, info_file):
        if os.path.exists(f):
            os.unlink(f)

    graph = _quantize(num_of_bits=16, skip_export=False, export_file=export_file)

    assert os.path.exists(export_file), "espdl model was not exported"
    assert os.path.exists(info_file), "info file was not exported"

    info_txt = open(info_file).read()
    names = sorted(set(__import__("re").findall(r"LUT_lut_\d+", info_txt)))
    assert len(names) == 3, f"Expected 3 LUT tables in export, got {names}"
    assert names == ["LUT_lut_0", "LUT_lut_1", "LUT_lut_2"], f"Naming rule broken: {names}"

    for f in (export_file, info_file, export_file.rsplit(".", 1)[0] + ".json"):
        if os.path.exists(f):
            os.unlink(f)
    print("PASS: test_lut_export_tables")


# ===================================================================
# Test 8: exported table is built from original_op_type (real math, not identity)
# ===================================================================
def test_lut_table_uses_original_op_type():
    graph = _quantize(num_of_bits=16)
    sigmoid_lut = next(op for op in _lut_ops(graph) if op.attributes.get("original_op_type") == "Sigmoid")

    info = ExporterPatternInfo()
    info.reset()
    in_exp = sigmoid_lut.input_quant_config[0].scale.log2().round().int().item()
    out_exp = sigmoid_lut.output_quant_config[0].scale.log2().round().int().item()
    info.add_var_exponents(sigmoid_lut.inputs[0].name, in_exp)
    info.add_var_exponents(sigmoid_lut.outputs[0].name, out_exp)

    step = sigmoid_lut.attributes["int16_lut_step"]
    lut = AddLUTPattern(int16_step=step).calculate_lut(sigmoid_lut, info, 2**15 - 1, -(2**15), step)

    # op.type must be restored to "LUT" after table generation.
    assert sigmoid_lut.type == "LUT"
    assert lut.numel() == 65536 // step + 1

    # Reference sigmoid table at the pivots.
    in_scale = 2.0**in_exp
    out_scale = 2.0**out_exp
    pivots = (torch.arange(0, 65536 // step + 1, dtype=torch.float32) * step - 32768) * in_scale
    ref = ppq_tensor_round(
        torch.sigmoid(pivots) / out_scale, sigmoid_lut.output_quant_config[0].rounding
    ).clamp(-32768, 32767).to(torch.int32)
    identity = ppq_tensor_round(
        pivots / out_scale, sigmoid_lut.output_quant_config[0].rounding
    ).clamp(-32768, 32767).to(torch.int32)

    assert torch.equal(lut.cpu().to(torch.int32), ref), "LUT table does not match sigmoid math"
    assert not torch.equal(lut.cpu().to(torch.int32), identity), "LUT table is identity (op.type dispatch bug)"
    print("PASS: test_lut_table_uses_original_op_type")


if __name__ == "__main__":
    test_lut_fusion_default_int16()
    test_lut_no_fusion_int8()
    test_lut_fusion_disabled()
    test_lut_step_setting()
    test_lut_emulator_bit_exact()
    test_lut_simulation_changes_output()
    test_lut_export_tables()
    test_lut_table_uses_original_op_type()
    print("\nAll INT16 LUT tests passed.")
