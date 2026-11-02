"""ESP-DL INT16 LUT fusion pass.

Rewrites unary ops listed in ``target_ops`` into ``LUT`` after calibration.
The pass stores ``int16_lut_step`` on each fused op. Simulation and export
both read that stamp. Relu, Clip, and PRelu are not in the default list.
"""

from typing import List, Optional

from esp_ppq.core import ESPDL_QUANT_PLATFORMS, TargetPlatform
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
            target_ops: Op types to fuse. ``None`` fuses nothing.
            lut_step: Sub-sampling step written to ``int16_lut_step``.
            verbose: Print per-op fusion decisions.
        """
        super().__init__(name="ESPDL LUT Fusion Pass")
        self.target_ops = list(target_ops) if target_ops is not None else []
        self.lut_step = int(lut_step)
        self.verbose = verbose
        self.lut_count = 0

    @staticmethod
    def _has_math_impl(op_type: str) -> bool:
        """True if esp-ppq can evaluate ``op_type`` (needed to build the LUT)."""
        from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE

        return DEFAULT_BACKEND_TABLE.get(op_type) is not None

    def verify_compatibility(self, op: Operation) -> bool:
        """Return True if this operator itself is a 16-bit INT16 LUT candidate.

        ``op.platform`` confirms that the operator targets an ESP-DL quantized
        platform; ``config.num_of_bits`` checks that both the activation input
        and output are 16-bit.
        """
        if getattr(op, "platform", None) not in ESPDL_QUANT_PLATFORMS:
            if self.verbose:
                print(f"[ESPDL LUT] Skipping {op.name}: platform {op.platform} is not an ESP-DL quantized platform.")
            return False
        if not hasattr(op, "config") or op.config is None:
            if self.verbose:
                print(f"[ESPDL LUT] Skipping {op.name}: operation is not quantized (no TQC).")
            return False
        if not op.input_quant_config or not op.output_quant_config:
            if self.verbose:
                print(f"[ESPDL LUT] Skipping {op.name}: missing input or output quantization config.")
            return False

        in_bits = op.input_quant_config[0].dominated_by.num_of_bits
        out_bits = op.output_quant_config[0].dominated_by.num_of_bits
        if in_bits != 16 or out_bits != 16:
            if self.verbose:
                print(
                    f"[ESPDL LUT] Skipping {op.name}: activation is "
                    f"in={in_bits}-bit out={out_bits}-bit, LUT requires 16-bit on both."
                )
            return False

        return True

    def _should_fuse(self, op: Operation) -> bool:
        if op.type not in self.target_ops:
            return False

        # LUT only supports unary element-wise operations.
        if len(op.inputs) != 1 or len(op.outputs) != 1:
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
