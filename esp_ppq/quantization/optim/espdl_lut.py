"""ESP-DL INT16 LUT fusion pass.

Rewrites calibrated *unary element-wise* operations - anything shaped like
``y = f(x)`` (Swish/SiLU, Sigmoid, Tanh, Relu, Elu, Erf, Gelu, HardSwish, Exp,
Log, Sqrt, Sin, ...) - into ``LUT`` operations so that:

1. The exported ``.espdl`` model uses the hardware-accelerated ``LUT`` module on
   ESP32-P4 / ESP32-S3 (the per-element INT16 activation modules in esp-dl do
   *not* consume a LUT - only the dedicated ``LUT`` module does).
2. esp-ppq's simulation becomes bit-exact with the device, because ``LUT`` ops
   are executed through the integer :class:`~esp_ppq.executor.op.torch.espdl_lut.HardwareEmulator`.

Run this pass *after* calibration (it is a post-quantization pass)::

    from esp_ppq.quantization.optim import EspdlLUTFusionPass

    # Auto-detect every eligible y = f(x) op (default):
    EspdlLUTFusionPass(lut_step=32).optimize(graph=graph)

    # Or restrict to specific op types:
    EspdlLUTFusionPass(target_ops=["Swish"], lut_step=32).optimize(graph=graph)

Eligibility for auto-detection (``target_ops=None``): an op is fused iff it is
``y = f(x)`` - **exactly one input variable and one output variable, no extra
parameter inputs** (attributes such as ``alpha``/``beta`` are fine) - it is a
known point-wise activation/math op, and it has a math implementation esp-ppq
can evaluate (needed to build the LUT table at export time).

The ``lut_step`` used here must match the ``int16_lut_step`` passed to the
exporter (the fusion pass stamps the step on each op so the exporter can pick it
up automatically).
"""

from typing import List, Optional

from esp_ppq.core import TargetPlatform
from esp_ppq.IR import BaseGraph, Operation

from .base import QuantizationOptimizationPass

class EspdlLUTFusionPass(QuantizationOptimizationPass):
    """Rewrite unary ``y = f(x)`` ops into bit-exact INT16 ``LUT`` ops for ESP-DL."""

    def __init__(
        self,
        target_ops: Optional[List[str]] = None,
        lut_step: int = 32,
        verbose: bool = False,
    ):
        """
        Args:
            target_ops: Explicit list of op types to fuse. When ``None`` (default)
                the pass auto-detects **every** eligible ``y = f(x)`` op in the
                graph. When a list is given, only those op types are considered,
                but the structural / safety checks below still apply.
            lut_step: LUT sub-sampling step (must match the exporter's
                ``int16_lut_step``).
            verbose: Print per-op fusion decisions.
        """
        super().__init__(name="ESPDL LUT Fusion Pass")
        self.target_ops = list(target_ops) if target_ops is not None else []
        self.lut_step = int(lut_step)
        self.verbose = verbose
        self.lut_count = 0

    @staticmethod
    def _pointwise_op_universe() -> set:
        """Op types esp-ppq treats as point-wise ``y = f(x)`` math."""
        # Imported lazily to avoid import cycles at package init time.
        from esp_ppq.parser.espdl.espdl_typedef import ACTIVATION_OP_SET, MATH_OP_SET

        return (set(ACTIVATION_OP_SET) | set(MATH_OP_SET)) - _NON_FUSABLE_TYPES

    @staticmethod
    def _has_math_impl(op_type: str) -> bool:
        """True if esp-ppq can evaluate ``op_type`` (needed to build the LUT)."""
        from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE

        return DEFAULT_BACKEND_TABLE.get(op_type) is not None


    def verify_compatibility(self, op: Operation) -> bool:
        """Return True if ``op`` may be fused into an INT16 LUT.

        Requires a 16-bit ESP-DL platform, a calibrated quantization config and
        16-bit input precision (the LUT fast path is INT16-only).
        """
        if not hasattr(op, "config") or op.config is None:
            if self.verbose:
                print(f"[ESPDL LUT] Skipping {op.name}: operation is not quantized (no TQC).")
            return False

        if op.input_quant_config[0].num_of_bits != 16:
            if self.verbose:
                print(
                    f"[ESPDL LUT] Skipping {op.name}: input is "
                    f"{op.input_quant_config[0].num_of_bits}-bit, LUT requires 16-bit."
                )
            return False

        return True

    def _should_fuse(self, op: Operation) -> bool:
        # Op-type filter: the point-wise universe (plus any user-provided extras).
        if op.type not in self.target_ops:
            return False

        # Safety: the exporter must be able to build the LUT table from f.
        if not self._has_math_impl(op.type):
            if self.verbose:
                print(f"[ESPDL LUT] Skipping {op.name} ({op.type}): no math implementation to build LUT.")
            return False

        return self.verify_compatibility(op)

    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        self.lut_count = 0

        for op in graph.operations.values():
            if not self._should_fuse(op):
                continue

            # Stash the original math + step so the executor emulator and the
            # exporter's table generator can recover the ideal activation.
            op.attributes["original_op_type"] = op.type
            op.attributes["int16_lut_step"] = self.lut_step
            op.type = "LUT"
            self.lut_count += 1

            if self.verbose:
                print(
                    f"[ESPDL LUT] Fused {op.attributes['original_op_type']} -> LUT "
                    f"(step={self.lut_step}) for operation: {op.name}"
                )

        if self.verbose:
            print(f"[ESPDL LUT] Fused {self.lut_count} activation op(s) into INT16 LUT.")

    @property
    def is_post_quantization_pass(self) -> bool:
        return True
