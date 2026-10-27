"""Fused Attention: admission limits and masked-softmax contract."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
import torch
import torch.nn.functional as F

from esp_ppq.executor.op.torch.espdl_attention import (
    ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM,
    ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS,
    ESPDL_ATTENTION_LAYOUTS,
    ESPDL_ATTENTION_MAX_HEAD_DIM,
    ESPDL_ATTENTION_MAX_KV_TOKENS,
    espdl_attention_eligible,
    espdl_attention_ineligible_reason,
    espdl_attention_logit_scale_ineligible_reason,
    espdl_attention_op_logit_scale_ineligible_reason,
    espdl_attention_shapes_eligible,
    espdl_attention_shapes_ineligible_reason,
    lower_espdl_attention_attributes,
)
from esp_ppq.IR import Operation, Variable


@contextmanager
def _silence_export_logs():
    """Hide ``torch.onnx.export`` chatter. Tests assert on the graph, not the log."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        yield


def _export_trace_and_fuse(module, args, input_names, output_names=("out",)):
    """Export, trace, then fuse. Returns ``(graph, feed)``."""
    import os
    import tempfile

    from esp_ppq.api import load_onnx_graph
    from esp_ppq.executor import TorchExecutor
    from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

    named = {name: tensor for name, tensor in zip(input_names, args)}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.onnx")
        with _silence_export_logs():
            torch.onnx.export(
                module,
                tuple(args),
                path,
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=18,
            )
            graph = load_onnx_graph(path)
        feed = [named[name] for name in graph.inputs.keys()]
        TorchExecutor(graph=graph, device="cpu").tracing_operation_meta(inputs=feed)
        fuse_espdl_attention_graph(graph)
    return graph, feed


def _attention_op(attributes=None, num_inputs: int = 3, mask_shape=None) -> Operation:
    """A bare ai.onnx::Attention op with ``num_inputs`` placeholder inputs."""
    inputs = [Variable(name=f"in{i}") for i in range(num_inputs)]
    if mask_shape is not None and num_inputs >= 4:
        inputs[3].shape = list(mask_shape)
    return Operation(
        name="attn",
        op_type="Attention",
        attributes=dict(attributes or {}),
        inputs=inputs,
    )


class TestEligibility(unittest.TestCase):
    """Admission test — lockstep with esp-dl deserialize."""

    def test_capability_defaults_match_device_headers(self):
        self.assertEqual(ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM, 64)
        self.assertEqual(ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS, 2048)
        header_root = Path(__file__).resolve().parents[2] / "esp-dl" / "esp-dl" / "dl" / "base"
        head_header = header_root / "dl_base_attention.hpp"
        kv_header = header_root / "dl_base_attention_common.hpp"
        if not head_header.is_file() or not kv_header.is_file():
            self.skipTest(f"device Attention headers are not in this checkout: {header_root}")
        import re

        head = re.search(
            r"#ifndef\s+ESP_DL_ATTENTION_MAX_HEAD_DIM\s+#define\s+ESP_DL_ATTENTION_MAX_HEAD_DIM\s+(\d+)",
            head_header.read_text(encoding="utf-8"),
        )
        kv = re.search(
            r"#ifndef\s+ESP_DL_ATTENTION_MAX_KV_TOKENS\s+#define\s+ESP_DL_ATTENTION_MAX_KV_TOKENS\s+(\d+)",
            kv_header.read_text(encoding="utf-8"),
        )
        self.assertIsNotNone(head)
        self.assertIsNotNone(kv)
        self.assertEqual(int(head.group(1)), ESPDL_ATTENTION_DEFAULT_MAX_HEAD_DIM)
        self.assertEqual(int(kv.group(1)), ESPDL_ATTENTION_DEFAULT_MAX_KV_TOKENS)

    def test_logit_scale_rejects_overflowing_qk_exponents(self):
        self.assertEqual(espdl_attention_logit_scale_ineligible_reason(None, None, None, 32), "")
        self.assertEqual(espdl_attention_logit_scale_ineligible_reason(0.125, 2**-15, 2**-15, 32), "")
        self.assertIn(
            "logit_scale",
            espdl_attention_logit_scale_ineligible_reason(1.0, float(2**80), float(2**80), 32),
        )
        op = _attention_op()
        self.assertEqual(espdl_attention_op_logit_scale_ineligible_reason(op, 32), "")
        self.assertIn(
            "logit_scale was not checked",
            espdl_attention_op_logit_scale_ineligible_reason(op, 32, require_scales=True),
        )

    def test_accepts_plain_self_attention(self):
        op = _attention_op()
        shapes = [[1, 4, 16, 32]] * 3
        self.assertTrue(espdl_attention_eligible(op, shapes))

    def test_accepts_gqa_and_optional_mask_input(self):
        op = _attention_op(num_inputs=4)
        shapes = [[1, 8, 32, 32], [1, 2, 32, 32], [1, 2, 32, 32]]
        self.assertTrue(espdl_attention_eligible(op, shapes))

    def test_mask_bound_is_checked_against_the_score_shape(self):
        # The device right-aligns the trailing four mask axes onto
        # [B, Hq, Lq, Lk]; each must be 1 or the full extent.
        qkv = [[2, 4, 16, 32]] * 3
        for mask, ok in (
            ([16, 16], True),  # rank 2, right-aligned
            ([1, 16], True),  # broadcast over Lq
            ([1, 1, 16, 16], True),
            ([2, 4, 16, 16], True),  # fully materialised
            ([2, 1, 1, 16], True),
            ([16, 1], True),  # one bias per query, shared over keys
            ([8, 16], False),  # Lq=8 != 16 and != 1
            ([1, 1, 16, 8], False),  # Lk mismatch
            ([3, 4, 16, 16], False),  # B=3 != 2
            ([1, 3, 16, 16], False),  # H=3 != 4 and != 1
            ([1, 1, 1, 1, 16, 16], False),  # rank 6: Add would give rank 6
            ([2, 1, 4, 16, 16], False),
        ):
            with self.subTest(mask=mask):
                op = _attention_op(num_inputs=4, mask_shape=mask)
                self.assertEqual(espdl_attention_eligible(op, qkv), ok)
                self.assertEqual(espdl_attention_ineligible_reason(op, qkv) == "", ok)

    def test_mask_dtype_is_bool_or_fp32(self):
        # Same contract as ONNX Attention / SDPA / TensorRT; the device
        # rejects any other attn_mask dtype at forward time.
        from esp_ppq.core import DataType

        qkv = [[1, 4, 16, 32]] * 3
        for dtype, ok in (
            (DataType.FP32, True),
            (DataType.BOOL, True),
            (DataType.INT8, False),
            (DataType.INT16, False),
            (DataType.INT64, False),
            (DataType.FP16, False),
        ):
            with self.subTest(dtype=dtype):
                op = _attention_op(num_inputs=4, mask_shape=[1, 1, 16, 16])
                op.inputs[3].dtype = dtype
                self.assertEqual(espdl_attention_eligible(op, qkv), ok)
                reason = espdl_attention_ineligible_reason(op, qkv)
                self.assertEqual(reason == "", ok)
                if not ok:
                    self.assertIn(dtype.name, reason)

    def test_bool_parameter_mask_is_eligible(self):
        op = _attention_op(num_inputs=4)
        op.inputs[3] = Variable(name="m", value=torch.ones(1, 1, 16, 16, dtype=torch.bool), is_parameter=True)
        self.assertTrue(espdl_attention_eligible(op, [[1, 4, 16, 32]] * 3))

    def test_unshaped_mask_is_not_a_rejection(self):
        # Consistent with the rest of the pass: unknown is not the same as
        # bad. The export-time gate is what bounds this.
        op = _attention_op(num_inputs=4, mask_shape=None)
        self.assertTrue(espdl_attention_eligible(op, [[1, 4, 16, 32]] * 3))

    def test_symbolic_mask_dims_are_rejected(self):
        op = _attention_op(num_inputs=4, mask_shape=[1, 1, "Lq", 16])
        self.assertFalse(espdl_attention_eligible(op, [[1, 4, 16, 32]] * 3))
        self.assertIn("non-static", espdl_attention_ineligible_reason(op, [[1, 4, 16, 32]] * 3))

    def test_mask_bound_follows_the_bnhd_axes(self):
        # Under BNHD, Lq is axis 1 and Hq axis 2, so a mask that is valid
        # read as BHND must be rejected and vice versa.
        qkv = [[1, 16, 4, 32]] * 3  # B=1, Lq=16, Hq=4
        ok_mask = [1, 4, 16, 16]  # [B, Hq, Lq, Lk]
        bad_mask = [1, 16, 4, 16]
        op = _attention_op({"layout": "BNHD"}, num_inputs=4, mask_shape=ok_mask)
        self.assertTrue(espdl_attention_eligible(op, qkv))
        op_bad = _attention_op({"layout": "BNHD"}, num_inputs=4, mask_shape=bad_mask)
        self.assertFalse(espdl_attention_eligible(op_bad, qkv))

    def test_accepts_head_dim_at_upper_bound(self):
        op = _attention_op()
        shapes = [[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM]] * 3
        self.assertTrue(espdl_attention_eligible(op, shapes))

    def test_rejects_head_dim_above_upper_bound(self):
        op = _attention_op()
        shapes = [[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3
        self.assertFalse(espdl_attention_eligible(op, shapes))

    def test_rejects_non_positive_head_dim(self):
        op = _attention_op()
        self.assertFalse(espdl_attention_eligible(op, [[1, 4, 16, 0]] * 3))

    def test_accepts_lk_at_upper_bound(self):
        op = _attention_op()
        shapes = [[1, 4, ESPDL_ATTENTION_MAX_KV_TOKENS, 32]] * 3
        self.assertTrue(espdl_attention_eligible(op, shapes))

    def test_rejects_lk_above_upper_bound(self):
        # Device-side validate_attention_shapes rejects this from inside
        # deserialize, which fails the whole model load; the fusion pass has
        # to see it first so the node stays as MatMul/Softmax primitives.
        op = _attention_op()
        lk = ESPDL_ATTENTION_MAX_KV_TOKENS + 1
        self.assertFalse(espdl_attention_eligible(op, [[1, 4, lk, 32]] * 3))

    def test_lk_bound_applies_on_the_bnhd_seq_axis(self):
        # Under BNHD the sequence axis is 1, so a Q/K/V that would be legal
        # read as BHND must still be rejected.
        op = _attention_op({"layout": "BNHD"})
        lk = ESPDL_ATTENTION_MAX_KV_TOKENS + 1
        self.assertFalse(espdl_attention_eligible(op, [[1, lk, 4, 32]] * 3))
        self.assertTrue(espdl_attention_eligible(op, [[1, ESPDL_ATTENTION_MAX_KV_TOKENS, 4, 32]] * 3))

    def test_kv_token_bound_resolver_honours_env_and_fails_safe(self):
        import os

        from esp_ppq.executor.op.torch.espdl_attention import _resolve_max_kv_tokens

        original = os.environ.get("ESPDL_ATTENTION_MAX_KV_TOKENS")
        try:
            for raw, expected in (
                (None, 2048),
                ("512", 512),
                ("8192", 8192),
                ("63", 2048),  # below CONFIG range -> fall back, do not trust
                ("8193", 2048),  # above CONFIG range
                ("lots", 2048),
            ):
                with self.subTest(raw=raw):
                    os.environ.pop("ESPDL_ATTENTION_MAX_KV_TOKENS", None)
                    if raw is not None:
                        os.environ["ESPDL_ATTENTION_MAX_KV_TOKENS"] = raw
                    self.assertEqual(_resolve_max_kv_tokens(), expected)
        finally:
            os.environ.pop("ESPDL_ATTENTION_MAX_KV_TOKENS", None)
            if original is not None:
                os.environ["ESPDL_ATTENTION_MAX_KV_TOKENS"] = original

    def test_long_lq_is_allowed_above_the_lk_bound(self):
        # Only Lk is capped on device (it sizes the K-head scratch); a long
        # query sequence is streamed in Q-blocks and must not be rejected.
        op = _attention_op()
        lq = ESPDL_ATTENTION_MAX_KV_TOKENS + 1
        shapes = [[1, 4, lq, 32], [1, 4, 16, 32], [1, 4, 16, 32]]
        self.assertTrue(espdl_attention_eligible(op, shapes))

    def test_accepts_bnhd_layout(self):
        op = _attention_op({"layout": "BNHD"})
        shapes = [[1, 16, 4, 32]] * 3
        self.assertTrue(espdl_attention_eligible(op, shapes))

    def test_layout_check_is_case_insensitive(self):
        shapes_bhnd = [[1, 4, 16, 32]] * 3
        shapes_bnhd = [[1, 16, 4, 32]] * 3
        self.assertTrue(espdl_attention_eligible(_attention_op({"layout": "bhnd"}), shapes_bhnd))
        self.assertTrue(espdl_attention_eligible(_attention_op({"layout": "bnhd"}), shapes_bnhd))

    def test_bhnd_and_bnhd_are_advertised_as_supported(self):
        self.assertEqual(set(ESPDL_ATTENTION_LAYOUTS), {"BHND", "BNHD"})

    def test_rejects_kv_cache(self):
        # A 5th input is past_key / past_value.
        op = _attention_op(num_inputs=5)
        self.assertFalse(espdl_attention_eligible(op, [[1, 4, 16, 32]] * 3))

    def test_rejects_unsupported_attributes(self):
        shapes = [[1, 4, 16, 32]] * 3
        for attr in ("softcap", "qk_matmul_output_mode", "softmax_precision"):
            with self.subTest(attr=attr):
                op = _attention_op({attr: 1})
                self.assertFalse(espdl_attention_eligible(op, shapes))

    def test_accepts_default_valued_unsupported_attributes(self):
        op = _attention_op(
            {
                "softcap": 0.0,
                "qk_matmul_output_mode": 0,
                "left_window_size": -1,
                "right_window_size": -1,
                "is_causal": 0,
                "q_num_heads": 4,
                "kv_num_heads": 4,
            }
        )
        self.assertTrue(espdl_attention_eligible(op, [[1, 4, 16, 32]] * 3))

    def test_rejects_unknown_attributes(self):
        op = _attention_op({"not_a_real_attr": 1})
        self.assertFalse(espdl_attention_eligible(op, [[1, 4, 16, 32]] * 3))
        self.assertIn("unsupported attrs", espdl_attention_ineligible_reason(op, [[1, 4, 16, 32]] * 3))

    def test_rejects_microsoft_packed_qkv(self):
        op = _attention_op({"qkv_hidden_sizes": [96, 96, 96]})
        self.assertFalse(espdl_attention_eligible(op, [[1, 4, 16, 32]] * 3))

    def test_rejects_non_4d_qkv(self):
        op = _attention_op()
        self.assertFalse(espdl_attention_eligible(op, [[1, 16, 96]] * 3))

    def test_rejects_missing_shape_metadata(self):
        op = _attention_op()
        self.assertFalse(espdl_attention_eligible(op, [None, None, None]))

    def test_rejects_symbolic_head_dim(self):
        # Dynamic axes arrive as strings; the device needs a concrete D.
        op = _attention_op()
        self.assertFalse(espdl_attention_eligible(op, [[1, 4, 16, "D"]] * 3))


class TestMaskedSoftmaxContract(unittest.TestCase):
    """A fully masked query row must produce 0, not NaN."""

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.H, self.Lq, self.Lk, self.D = 1, 2, 4, 6, 8
        self.q = torch.randn(self.B, self.H, self.Lq, self.D)
        self.k = torch.randn(self.B, self.H, self.Lk, self.D)
        self.v = torch.randn(self.B, self.H, self.Lk, self.D)
        self.scale = self.D**-0.5

    def _sdpa(self, mask):
        return F.scaled_dot_product_attention(self.q, self.k, self.v, attn_mask=mask, scale=self.scale)

    def test_row_all_masked_yields_zero(self):
        mask = torch.zeros(self.B, 1, self.Lq, self.Lk)
        masked_row = 2
        mask[..., masked_row, :] = float("-inf")

        out = self._sdpa(mask)

        self.assertFalse(torch.isnan(out).any(), "fully masked row produced NaN")
        torch.testing.assert_close(out[:, :, masked_row, :], torch.zeros(self.B, self.H, self.D))

    def test_unmasked_rows_are_unaffected_by_a_masked_neighbour(self):
        masked_row = 2
        mask = torch.zeros(self.B, 1, self.Lq, self.Lk)
        mask[..., masked_row, :] = float("-inf")

        out = self._sdpa(mask)
        reference = self._sdpa(None)

        keep = [i for i in range(self.Lq) if i != masked_row]
        torch.testing.assert_close(out[:, :, keep, :], reference[:, :, keep, :])

    def test_all_rows_masked_yields_all_zero(self):
        mask = torch.full((self.B, 1, self.Lq, self.Lk), float("-inf"))

        out = self._sdpa(mask)

        self.assertFalse(torch.isnan(out).any())
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_bool_mask_all_false_row_yields_zero(self):
        mask = torch.ones(self.B, 1, self.Lq, self.Lk, dtype=torch.bool)
        masked_row = 1
        mask[..., masked_row, :] = False
        additive = torch.zeros_like(mask, dtype=self.q.dtype).masked_fill(~mask, float("-inf"))

        out = self._sdpa(additive)

        self.assertFalse(torch.isnan(out).any())
        torch.testing.assert_close(out[:, :, masked_row, :], torch.zeros(self.B, self.H, self.D))


class TestPrimitiveFusion(unittest.TestCase):
    """MatMul -> [Mul/Div] -> [Add] -> Softmax -> MatMul  ==>  Attention."""

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.H, self.L, self.D = 1, 2, 8, 16
        self.q = torch.randn(self.B, self.H, self.L, self.D)
        self.k = torch.randn(self.B, self.H, self.L, self.D)
        self.v = torch.randn(self.B, self.H, self.L, self.D)
        self.mask = torch.zeros(self.B, 1, self.L, self.L)
        self.mask[..., self.L // 2 :] = -1e4

    def _run(self, module):
        """Export, fuse, and return (fused_graph_output, torch_output, attention ops)."""
        from esp_ppq.executor import TorchExecutor

        args = (self.q, self.k, self.v, self.mask)
        golden = module(*args)
        graph, feed = _export_trace_and_fuse(module, args, ["q", "k", "v", "mask"])
        fused_out = TorchExecutor(graph=graph, device="cpu").forward(inputs=feed)[0]
        attentions = [op for op in graph.operations.values() if op.type == "Attention"]
        return fused_out, golden, attentions

    def _assert_fuses_to_attention(self, module, expected_scale, num_inputs):
        out, golden, attentions = self._run(module)
        self.assertEqual(len(attentions), 1, "expected exactly one fused Attention node")
        self.assertEqual(attentions[0].num_of_input, num_inputs)
        self.assertAlmostEqual(attentions[0].attributes["scale"], expected_scale, places=6)
        self.assertIsNotNone(out, "fused graph produced no output")
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    @staticmethod
    def _platform_graph(platform):
        from esp_ppq.core import DataType
        from esp_ppq.IR import BaseGraph

        graph = BaseGraph(name="attention_platform_fusion")

        def variable(name, shape):
            var = graph.create_variable(name=name)
            var.shape = list(shape)
            var.dtype = DataType.FP32
            return var

        q = variable("q", [1, 2, 8, 16])
        k = variable("k", [1, 2, 8, 16])
        v = variable("v", [1, 2, 8, 16])
        mask = variable("mask", [1, 1, 8, 8])
        for var in (q, k, v, mask):
            graph.mark_variable_as_graph_input(var)

        kt = variable("kt", [1, 2, 16, 8])
        scores = variable("scores", [1, 2, 8, 8])
        scaled = variable("scaled", [1, 2, 8, 8])
        biased = variable("biased", [1, 2, 8, 8])
        probs = variable("probs", [1, 2, 8, 8])
        out = variable("out", [1, 2, 8, 16])
        scale = graph.create_variable(
            name="scale_value",
            value=torch.tensor(0.25, dtype=torch.float32),
            is_parameter=True,
        )
        scale.shape = []
        scale.dtype = DataType.FP32

        graph.create_operation(
            op_type="Transpose",
            name="kT",
            attributes={"perm": [0, 1, 3, 2]},
            platform=platform,
            inputs=[k],
            outputs=[kt],
        )
        graph.create_operation(
            op_type="MatMul",
            name="qk",
            platform=platform,
            inputs=[q, kt],
            outputs=[scores],
        )
        graph.create_operation(
            op_type="Mul",
            name="scale",
            platform=platform,
            inputs=[scores, scale],
            outputs=[scaled],
        )
        graph.create_operation(
            op_type="Add",
            name="bias",
            platform=platform,
            inputs=[scaled, mask],
            outputs=[biased],
        )
        graph.create_operation(
            op_type="Softmax",
            name="sm",
            attributes={"axis": -1},
            platform=platform,
            inputs=[biased],
            outputs=[probs],
        )
        graph.create_operation(
            op_type="MatMul",
            name="pv",
            platform=platform,
            inputs=[probs, v],
            outputs=[out],
        )
        graph.mark_variable_as_graph_output(out)
        return graph

    def test_fused_attention_inherits_uniform_platform(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

        graph = self._platform_graph(TargetPlatform.FP32)
        fuse_espdl_attention_graph(graph)
        attention = next(op for op in graph.operations.values() if op.type == "Attention")
        self.assertEqual(attention.platform, TargetPlatform.FP32)

    def test_mixed_platform_runtime_chain_is_not_fused(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

        for mismatched_name in ("kT", "qk", "scale", "bias", "sm", "pv"):
            with self.subTest(mismatched_name=mismatched_name):
                graph = self._platform_graph(TargetPlatform.FP32)
                graph.operations[mismatched_name].platform = TargetPlatform.UNSPECIFIED
                original_ops = set(graph.operations)
                fuse_espdl_attention_graph(graph)
                self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
                self.assertEqual(set(graph.operations), original_ops)

    def test_plain_matmul_softmax_matmul(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)), -1), v)

        self._assert_fuses_to_attention(M(), expected_scale=1.0, num_inputs=3)

    def test_mask_without_scale(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + mask, -1), v)

        self._assert_fuses_to_attention(M(), expected_scale=1.0, num_inputs=4)

    def test_rank5_mask_is_left_as_add(self):
        # scores[1,2,L,L] + bias[1,1,1,L,L] is rank 5 under ONNX broadcasting,
        # so the primitive subgraph and a fused rank-4 Attention compute
        # different things. Fusion must decline.
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                bias = mask.unsqueeze(0)  # [1, 1, 1, L, L]
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) + bias, -1), v)

        out, golden, attentions = self._run(M())
        self.assertEqual(attentions, [], "a rank-5 mask must not be fused")
        self.assertEqual(golden.dim(), 5, "premise: the primitive graph really is rank 5")
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_div_scale_folds_into_attribute(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / 4.0, -1), v)

        self._assert_fuses_to_attention(M(), expected_scale=0.25, num_inputs=3)

    def test_constant_divided_by_scores_is_not_fused(self):
        # c / scores is not a scale on the scores; folding it as scale=1/c would
        # change the result.
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                scores = torch.matmul(q, k.transpose(-1, -2))
                return torch.matmul(torch.softmax(torch.tensor(4.0) / scores, -1), v)

        out, golden, attentions = self._run(M())
        self.assertEqual(attentions, [], "Div(c, scores) must not be fused")
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_mul_scale_folds_into_attribute(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * 0.25, -1), v)

        self._assert_fuses_to_attention(M(), expected_scale=0.25, num_inputs=3)

    def test_negative_scale_is_not_fused(self):
        # The device kernel only admits a finite positive scale.
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * -0.25, -1), v)

        out, golden, attentions = self._run(M())
        self.assertEqual(attentions, [], "a negative scale must not be fused")
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_div_scale_with_mask(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / 4.0 + mask, -1), v)

        self._assert_fuses_to_attention(M(), expected_scale=0.25, num_inputs=4)

    def test_mul_scale_with_mask(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * 0.25 + mask, -1), v)

        self._assert_fuses_to_attention(M(), expected_scale=0.25, num_inputs=4)

    def test_k_transpose_is_absorbed(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v, mask):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / 4.0, -1), v)

        _, _, attentions = self._run(M())
        k_shape = list(attentions[0].inputs[1].shape)
        self.assertEqual(k_shape, [self.B, self.H, self.L, self.D])


class TestSdpaExportFusion(unittest.TestCase):
    """Opset 18/20 SDPA lowerings. The standard Attention op does not exist yet."""

    def _load_fused(self, module, args, names, *, dynamo, opset=18, execute=True):
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.executor import TorchExecutor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            export_kwargs = dict(
                input_names=list(names),
                output_names=["y"],
                opset_version=opset,
                dynamo=dynamo,
            )
            if dynamo:
                export_kwargs.update(report=False, verify=False)
            with _silence_export_logs():
                torch.onnx.export(module, args, path, **export_kwargs)
                graph = load_onnx_graph(path)
            from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

            feed = list(args)
            tracer = TorchExecutor(graph=graph, device="cpu")
            if execute:
                tracer.tracing_operation_meta(inputs=feed)
            fuse_espdl_attention_graph(graph, executor=tracer, inputs=feed if execute else None)
            out = TorchExecutor(graph=graph, device="cpu").forward(inputs=feed)[0] if execute else None
        attentions = [op for op in graph.operations.values() if op.type == "Attention"]
        return graph, attentions, out

    def test_legacy_and_dynamo_sdpa_fuse(self):
        class Sdpa(torch.nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        torch.manual_seed(0)
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        golden = Sdpa()(q, k, v)
        for dynamo in (False, True):
            with self.subTest(dynamo=dynamo):
                _, attentions, out = self._load_fused(Sdpa().eval(), (q, k, v), ["q", "k", "v"], dynamo=dynamo)
                self.assertEqual(len(attentions), 1)
                self.assertAlmostEqual(attentions[0].attributes["scale"], 16**-0.5, places=5)
                torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_legacy_opset20_matches_opset18(self):
        class Sdpa(torch.nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        q = torch.randn(1, 4, 8, 16)
        _, attentions, _ = self._load_fused(Sdpa().eval(), (q, q, q), ["q", "k", "v"], dynamo=False, opset=20)
        self.assertEqual(len(attentions), 1)
        self.assertAlmostEqual(attentions[0].attributes["scale"], 16**-0.5, places=5)

    def test_multihead_attention_fuses_the_inner_sdpa(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = torch.nn.MultiheadAttention(64, 4, batch_first=True, bias=False)

            def forward(self, x):
                y, _ = self.m(x, x, x, need_weights=False)
                return y

        torch.manual_seed(0)
        block = Block().eval()
        x = torch.randn(1, 8, 64)
        golden = block(x)
        for dynamo in (False, True):
            with self.subTest(dynamo=dynamo):
                graph, attentions, out = self._load_fused(block, (x,), ["x"], dynamo=dynamo, execute=True)
                self.assertEqual(len(attentions), 1)
                self.assertAlmostEqual(attentions[0].attributes["scale"], 16**-0.5, places=5)
                self.assertTrue(any(op.type in ("MatMul", "Gemm") for op in graph.operations.values()))
                torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_dynamo_multihead_cross_attention_keeps_lq_and_lk_independent(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = torch.nn.MultiheadAttention(64, 4, batch_first=True, bias=False)

            def forward(self, q, memory):
                y, _ = self.m(q, memory, memory, need_weights=False)
                return y

        torch.manual_seed(0)
        block = Block().eval()
        for lq, lk in ((5, 7), (9, 4)):
            with self.subTest(lq=lq, lk=lk):
                q = torch.randn(1, lq, 64)
                memory = torch.randn(1, lk, 64)
                golden = block(q, memory)
                _, attentions, out = self._load_fused(
                    block,
                    (q, memory),
                    ["q", "memory"],
                    dynamo=True,
                    execute=True,
                )
                self.assertEqual(len(attentions), 1)
                self.assertEqual(
                    [list(var.shape) for var in attentions[0].inputs[:3]],
                    [[1, 4, lq, 16], [1, 4, lk, 16], [1, 4, lk, 16]],
                )
                torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_multihead_attention_default_need_weights_fuses_3d_bmm(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = torch.nn.MultiheadAttention(64, 4, batch_first=True, bias=False)

            def forward(self, x):
                y, _ = self.m(x, x, x)  # default need_weights=True lowers to 3-D bmm
                return y

        torch.manual_seed(0)
        block = Block().eval()
        x = torch.randn(2, 8, 64)
        golden = block(x)
        for dynamo in (False, True):
            with self.subTest(dynamo=dynamo):
                graph, attentions, out = self._load_fused(block, (x,), ["x"], dynamo=dynamo, execute=True)
                self.assertEqual(len(attentions), 1)
                self.assertAlmostEqual(attentions[0].attributes["scale"], 16**-0.5, places=5)
                self.assertEqual([list(v.shape) for v in attentions[0].inputs[:3]], [[1, 8, 8, 16]] * 3)
                torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_rank3_sdpa_fuses_with_4d_reshape(self):
        class Sdpa(torch.nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        q = torch.randn(2, 8, 16)
        graph, attentions, out = self._load_fused(Sdpa().eval(), (q, q, q), ["q", "k", "v"], dynamo=False)
        golden = Sdpa()(q, q, q)
        self.assertEqual(len(attentions), 1)
        self.assertEqual([list(v.shape) for v in attentions[0].inputs[:3]], [[1, 2, 8, 16]] * 3)
        self.assertEqual(list(attentions[0].outputs[0].shape), [1, 2, 8, 16])
        self.assertEqual(list(out.shape), [2, 8, 16])
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)


class TestWhereMaskFusion(unittest.TestCase):
    """``MatMul → [scale] → Where(cond, scores, -inf) → Softmax → MatMul``.

    Phase 2 fusion path: constant-fold Where-mask into additive Add-mask so
    the primitive fuser can consume it. Static ``cond`` only — esp-dl has
    no Where kernel, so runtime cond must fall through to the general
    MatMul/Softmax path.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.H, self.L, self.D = 1, 2, 8, 16
        self.q = torch.randn(self.B, self.H, self.L, self.D)
        self.k = torch.randn(self.B, self.H, self.L, self.D)
        self.v = torch.randn(self.B, self.H, self.L, self.D)

    def _run(self, module, extra_inputs=()):
        """Export, fuse, and return (fused output, torch output, Attention ops, Where ops)."""
        from esp_ppq.executor import TorchExecutor

        args = (self.q, self.k, self.v) + tuple(extra_inputs)
        input_names = ["q", "k", "v"] + [f"extra_{i}" for i in range(len(extra_inputs))]
        golden = module(*args)
        graph, feed = _export_trace_and_fuse(module, args, input_names)
        fused_out = TorchExecutor(graph=graph, device="cpu").forward(inputs=feed)[0]
        attentions = [op for op in graph.operations.values() if op.type == "Attention"]
        wheres = [op for op in graph.operations.values() if op.type == "Where"]
        return fused_out, golden, attentions, wheres

    def test_where_true_is_scores_folds_to_additive_mask(self):
        """``Where(cond, scores, -inf)`` with static cond fuses to Attention(Q,K,V,mask)."""

        L = self.L
        # Upper-triangular keep mask (causal-like): cond True on the row's own
        # column and everything left of it. Static (registered as buffer so it
        # inlines as a Constant on export).
        keep = torch.tril(torch.ones(L, L, dtype=torch.bool)).view(1, 1, L, L)

        class M(torch.nn.Module):
            def __init__(self, keep):
                super().__init__()
                self.register_buffer("keep", keep)

            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
                masked = torch.where(self.keep, scores, torch.tensor(float("-inf")))
                return torch.matmul(torch.softmax(masked, -1), v)

        out, golden, attentions, wheres = self._run(M(keep))
        self.assertEqual(len(attentions), 1)
        self.assertEqual(attentions[0].num_of_input, 4, "fused Attention should carry a mask")
        self.assertEqual(len(wheres), 0, "the Where op must be detached after fusion")
        self.assertAlmostEqual(attentions[0].attributes["scale"], 0.25, places=6)
        # Materialized fp32 mask comes from a new Constant op (activation-shaped)
        # so remove_operation does not strip its value when the intermediate Add
        # is detached during fusion.
        mask_var = attentions[0].inputs[3]
        self.assertFalse(getattr(mask_var, "is_parameter", False))
        self.assertIsNotNone(mask_var.source_op, "mask should be produced by a Constant")
        self.assertEqual(mask_var.source_op.type, "Constant")
        mask_val = mask_var.source_op.attributes["value"]
        self.assertEqual(mask_val.dtype, torch.float32)
        # Every position: 0 where keep is True, -inf where keep is False.
        expected_mask = torch.where(
            keep, torch.zeros_like(keep, dtype=torch.float32), torch.full_like(keep, float("-inf"), dtype=torch.float32)
        )
        torch.testing.assert_close(mask_val, expected_mask)
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_fp32_where_override_is_not_confused_with_dispatcher_fallback(self):
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.core import TargetPlatform
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

        keep = torch.tril(torch.ones(self.L, self.L, dtype=torch.bool)).view(1, 1, self.L, self.L)

        class M(torch.nn.Module):
            def __init__(self, keep):
                super().__init__()
                self.register_buffer("keep", keep)

            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2))
                masked = torch.where(self.keep, scores, torch.tensor(float("-inf")))
                return torch.matmul(torch.softmax(masked, -1), v)

        args = (self.q, self.k, self.v)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "static_where.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    M(keep),
                    args,
                    path,
                    input_names=["q", "k", "v"],
                    output_names=["out"],
                    opset_version=18,
                )
            for explicit_override in (False, True):
                with self.subTest(explicit_override=explicit_override):
                    graph = load_onnx_graph(path)
                    feed = [dict(zip(("q", "k", "v"), args))[name] for name in graph.inputs]
                    TorchExecutor(graph=graph, device="cpu").tracing_operation_meta(inputs=feed)
                    for op in graph.operations.values():
                        op.platform = TargetPlatform.UNSPECIFIED
                    where = next(op for op in graph.operations.values() if op.type == "Where")
                    where.platform = TargetPlatform.FP32
                    overrides = {where.name} if explicit_override else set()
                    fuse_espdl_attention_graph(
                        graph,
                        dispatching_overrides=overrides,
                        allow_automatic_fp32_where=True,
                    )
                    attentions = [op for op in graph.operations.values() if op.type == "Attention"]
                    if explicit_override:
                        self.assertEqual(attentions, [])
                        add = next(op for op in graph.operations.values() if op.type == "Add")
                        self.assertEqual(add.platform, TargetPlatform.FP32)
                    else:
                        self.assertEqual(len(attentions), 1)
                        self.assertEqual(attentions[0].platform, TargetPlatform.UNSPECIFIED)

    def test_where_true_is_neg_inf_folds_to_additive_mask(self):
        """``Where(cond, -inf, scores)`` — cond True masks, False keeps."""

        L = self.L
        drop = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1).view(1, 1, L, L)

        class M(torch.nn.Module):
            def __init__(self, drop):
                super().__init__()
                self.register_buffer("drop", drop)

            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
                masked = torch.where(self.drop, torch.tensor(float("-inf")), scores)
                return torch.matmul(torch.softmax(masked, -1), v)

        out, golden, attentions, wheres = self._run(M(drop))
        self.assertEqual(len(attentions), 1)
        self.assertEqual(attentions[0].num_of_input, 4)
        self.assertEqual(len(wheres), 0)
        mask_var = attentions[0].inputs[3]
        self.assertEqual(mask_var.source_op.type, "Constant")
        mask_val = mask_var.source_op.attributes["value"]
        expected_mask = torch.where(
            drop, torch.full_like(drop, float("-inf"), dtype=torch.float32), torch.zeros_like(drop, dtype=torch.float32)
        )
        torch.testing.assert_close(mask_val, expected_mask)
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_runtime_cond_falls_back_to_matmul_softmax(self):
        """A runtime cond has no static value — no fold, no fusion, no crash."""

        L = self.L

        class M(torch.nn.Module):
            def forward(self, q, k, v, cond):
                scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
                masked = torch.where(cond, scores, torch.tensor(float("-inf")))
                return torch.matmul(torch.softmax(masked, -1), v)

        cond = torch.tril(torch.ones(L, L, dtype=torch.bool)).view(1, 1, L, L)
        out, golden, attentions, wheres = self._run(M(), extra_inputs=(cond,))
        # Runtime cond: no fusion, but graph still runs the primitive form.
        self.assertEqual(len(attentions), 0)
        self.assertEqual(len(wheres), 1)
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_runtime_cond_survives_quant_calibration(self):
        """A bool Where condition must stay bool through the w8a8 calib collate."""
        import os
        import tempfile

        from esp_ppq.api.espdl_interface import collate_fn_template, espdl_quantize_onnx
        from esp_ppq.core import DataType, QuantizationStates

        L = self.L

        class M(torch.nn.Module):
            def forward(self, q, k, v, cond):
                scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
                masked = torch.where(cond, scores, torch.tensor(float("-inf")))
                return torch.matmul(torch.softmax(masked, -1), v)

        cond = torch.tril(torch.ones(L, L, dtype=torch.bool)).view(1, 1, L, L)
        activations = torch.randn(2, 3)
        converted = collate_fn_template([activations, cond])
        self.assertEqual(converted[0].dtype, torch.float32)
        self.assertEqual(converted[1].dtype, torch.bool)

        args = (self.q, self.k, self.v, cond)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    M(),
                    args,
                    path,
                    input_names=["q", "k", "v", "cond"],
                    output_names=["y"],
                    opset_version=18,
                    dynamo=False,
                )
            from esp_ppq.api import load_onnx_graph

            with _silence_export_logs():
                loaded = load_onnx_graph(path)
            named = {"q": self.q, "k": self.k, "v": self.v, "cond": cond}
            ordered = [named[n] for n in loaded.inputs.keys()]
            with _silence_export_logs():
                graph = espdl_quantize_onnx(
                    onnx_import_file=path,
                    espdl_export_file=os.path.join(tmp, "out.espdl"),
                    calib_dataloader=[ordered, [t.clone() for t in ordered]],
                    calib_steps=2,
                    input_shape=None,
                    inputs=ordered,
                    target="esp32p4",
                    quant_type="w8a8",
                    device="cpu",
                    error_report=False,
                    skip_export=True,
                    verbose=0,
                )
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        where = next(op for op in graph.operations.values() if op.type == "Where")
        self.assertEqual(where.inputs[0].dtype, DataType.BOOL)
        softmax = next(op for op in where.outputs[0].dest_ops if op.type == "Softmax")
        self.assertEqual(softmax.config.input_quantization_config[0].state, QuantizationStates.FP32)

    def test_where_without_scale_fuses(self):
        """``MatMul → Where → Softmax → MatMul`` (no scale) — scale=1.0 default."""

        L = self.L
        keep = torch.tril(torch.ones(L, L, dtype=torch.bool)).view(1, 1, L, L)

        class M(torch.nn.Module):
            def __init__(self, keep):
                super().__init__()
                self.register_buffer("keep", keep)

            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2))
                masked = torch.where(self.keep, scores, torch.tensor(float("-inf")))
                return torch.matmul(torch.softmax(masked, -1), v)

        out, golden, attentions, _ = self._run(M(keep))
        self.assertEqual(len(attentions), 1)
        self.assertEqual(attentions[0].num_of_input, 4)
        self.assertAlmostEqual(attentions[0].attributes["scale"], 1.0, places=6)
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)


class TestFusionLeavesNativeAttributes(unittest.TestCase):
    """Fusion must not rewrite attributes on a native Attention node."""

    def test_fusion_does_not_rewrite_attention_attributes(self):
        from esp_ppq.IR import BaseGraph
        from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

        attrs = {
            "scale": 0.125,
            "is_causal": 0,
            "q_num_heads": 4,
            "kv_num_heads": 4,
            "softcap": 0.0,
        }
        graph = BaseGraph(name="native")
        inputs = [graph.create_variable(name=f"in{i}", shape=[1, 4, 16, 32]) for i in range(3)]
        output = graph.create_variable(name="out", shape=[1, 4, 16, 32])
        op = graph.create_operation(
            op_type="Attention", name="attn", attributes=dict(attrs), inputs=inputs, outputs=[output]
        )
        snapshot = dict(op.attributes)
        fuse_espdl_attention_graph(graph)
        self.assertEqual(dict(op.attributes), snapshot)
        self.assertEqual(op.type, "Attention")


class TestBnhdLayout(unittest.TestCase):
    """BNHD is the token-major layout; the fused path keeps it (no graph Transpose)."""

    def test_reference_bnhd_matches_permuted_bhnd(self):
        from esp_ppq.executor.op.torch.espdl_attention import _espdl_attention_reference

        torch.manual_seed(0)
        q = torch.randn(1, 6, 2, 8)
        out_bnhd = _espdl_attention_reference(q, q, q, None, scale=0.125, is_causal=False, layout="BNHD")
        q4 = q.permute(0, 2, 1, 3)
        out_bhnd = _espdl_attention_reference(q4, q4, q4, None, scale=0.125, is_causal=False, layout="BHND")
        torch.testing.assert_close(out_bnhd, out_bhnd.permute(0, 2, 1, 3), rtol=1e-5, atol=1e-5)

    def test_int8_emulator_bnhd_matches_bhnd_bitexact(self):
        """Native-layout fake-quant + integer permute must match BHND bit-exact."""
        from esp_ppq.executor.op.torch.espdl_attention import _simulate_espdl_int8_attention

        torch.manual_seed(1)
        scale = 0.0234375  # 3/128, typical int8 activation scale
        for lq, h, d in ((17, 4, 32), (16, 4, 64), (65, 8, 32)):
            q = torch.randn(1, lq, h, d)
            k = torch.randn(1, lq, h, d)
            v = torch.randn(1, lq, h, d)
            out_bnhd = _simulate_espdl_int8_attention(q, k, v, d**-0.5, scale, scale, scale, scale, "BNHD")
            out_bhnd = _simulate_espdl_int8_attention(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                d**-0.5,
                scale,
                scale,
                scale,
                scale,
                "BHND",
            )
            got = torch.round(out_bnhd / scale).clamp(-128, 127).to(torch.int8)
            want = torch.round(out_bhnd.transpose(1, 2).contiguous() / scale).clamp(-128, 127).to(torch.int8)
            self.assertTrue(torch.equal(got, want), f"BNHD vs BHND int8 mismatch at L={lq} H={h} D={d}")

    def test_reference_accepts_bhnd(self):
        from esp_ppq.executor.op.torch.espdl_attention import _espdl_attention_reference

        q = torch.randn(1, 2, 4, 8)
        out = _espdl_attention_reference(q, q, q, None, scale=0.125, is_causal=False, layout="BHND")
        self.assertEqual(tuple(out.shape), (1, 2, 4, 8))

    def test_onnx23_bnhd_matches_reference(self):
        from esp_ppq.executor.op.torch.default import _onnx23_attention_forward
        from esp_ppq.executor.op.torch.espdl_attention import _espdl_attention_reference

        torch.manual_seed(0)
        q = torch.randn(1, 6, 2, 128)
        k = torch.randn(1, 5, 2, 128)
        v = torch.randn(1, 5, 2, 128)
        mask = torch.zeros(1, 1, 6, 5)
        mask[:, :, :, -1] = float("-inf")
        op = _attention_op({"layout": b"bnhd", "is_causal": 1, "scale": 0.125}, num_inputs=4)
        out = _onnx23_attention_forward(op=op, values=[q, k, v, mask])
        ref = _espdl_attention_reference(q, k, v, mask, scale=0.125, is_causal=True, layout="BNHD")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        self.assertEqual(tuple(out.shape), (1, 6, 2, 128))

    def test_onnx23_3d_ignores_bnhd_for_packing(self):
        from esp_ppq.executor.op.torch.default import _onnx23_attention_forward

        torch.manual_seed(1)
        q = torch.randn(1, 6, 16)
        op_bnhd = _attention_op({"layout": "BNHD", "q_num_heads": 2, "kv_num_heads": 2})
        op_bhnd = _attention_op({"q_num_heads": 2, "kv_num_heads": 2})
        out_bnhd = _onnx23_attention_forward(op=op_bnhd, values=[q, q.clone(), q.clone()])
        out_bhnd = _onnx23_attention_forward(op=op_bhnd, values=[q, q.clone(), q.clone()])
        torch.testing.assert_close(out_bnhd, out_bhnd)


class TestInt8EmulatorMask(unittest.TestCase):
    """attn_mask on the int8 fast path joins the int32 dots (attention_mask_dots)."""

    S = 0.0234375  # 3/128

    def _emu(self, q, k, v, mask=None, layout="BHND"):
        from esp_ppq.executor.op.torch.espdl_attention import _simulate_espdl_int8_attention

        d = q.shape[-1]
        return _simulate_espdl_int8_attention(q, k, v, d**-0.5, self.S, self.S, self.S, self.S, layout, attn_mask=mask)

    def _qkv(self, seed, h=4, lq=16, lk=128, d=32):
        g = torch.Generator().manual_seed(seed)
        return (
            torch.randn(1, h, lq, d, generator=g),
            torch.randn(1, h, lk, d, generator=g),
            torch.randn(1, h, lk, d, generator=g),
        )

    def _logit_scale(self, d):
        from esp_ppq.executor.op.torch.espdl_attention import _fp32

        return _fp32(_fp32(_fp32(d**-0.5) * _fp32(self.S)) * _fp32(self.S))

    def test_zero_mask_is_bitexact_with_no_mask(self):
        q, k, v = self._qkv(0)
        mask = torch.zeros(1, 1, 16, 128)
        self.assertTrue(torch.equal(self._emu(q, k, v, mask), self._emu(q, k, v)))

    def test_neg_inf_tail_tiles_equal_truncated_keys(self):
        """-inf over whole trailing K-tiles is exactly the shorter sequence."""
        q, k, v = self._qkv(1)
        mask = torch.zeros(1, 1, 16, 128)
        mask[..., 64:] = float("-inf")
        want = self._emu(q, k[:, :, :64].contiguous(), v[:, :, :64].contiguous())
        self.assertTrue(torch.equal(self._emu(q, k, v, mask), want))

    def test_row_constant_in_dot_units_is_bitexact(self):
        """Softmax is shift-invariant; a per-row offset of n dot units must not change a bit."""
        q, k, v = self._qkv(2)
        cs = self._logit_scale(32)
        offsets = torch.tensor([37.0, -1000.0, 0.0, 5.0] * 4).view(1, 1, 16, 1) * cs
        mask = offsets.expand(1, 1, 16, 128).contiguous()
        self.assertTrue(torch.equal(self._emu(q, k, v, mask), self._emu(q, k, v)))

    def test_broadcast_mask_matches_expanded(self):
        q, k, v = self._qkv(3)
        g = torch.Generator().manual_seed(33)
        mask = torch.randn(1, 1, 16, 128, generator=g)
        mask[0, 0, 3, ::3] = float("-inf")
        full = mask.expand(1, 4, 16, 128).contiguous()
        self.assertTrue(torch.equal(self._emu(q, k, v, mask), self._emu(q, k, v, full)))

    def test_fully_masked_row_is_zero(self):
        q, k, v = self._qkv(4)
        mask = torch.zeros(1, 1, 16, 128)
        mask[0, 0, 5, :] = float("-inf")
        out = self._emu(q, k, v, mask)
        self.assertTrue(torch.all(out[:, :, 5] == 0))
        self.assertTrue(torch.any(out[:, :, 4] != 0))

    def test_large_negative_mask_matches_neg_inf(self):
        """-1e9 clamps to -2^28 dot units, which lands on the zero end of the exp LUT."""
        q, k, v = self._qkv(5)
        m_inf = torch.zeros(1, 1, 16, 128)
        m_inf[..., 1::2] = float("-inf")
        m_big = torch.zeros(1, 1, 16, 128)
        m_big[..., 1::2] = -1e9
        self.assertTrue(torch.equal(self._emu(q, k, v, m_big), self._emu(q, k, v, m_inf)))

    def test_additive_bias_tracks_fp32_reference(self):
        from esp_ppq.executor.op.torch.espdl_attention import _espdl_attention_reference

        for layout, (h, lq, lk, d) in (
            ("BHND", (4, 16, 100, 32)),
            ("BHND", (2, 20, 70, 64)),
            ("BNHD", (4, 17, 65, 32)),
        ):
            q, k, v = self._qkv(6, h=h, lq=lq, lk=lk, d=d)
            g = torch.Generator().manual_seed(66)
            mask = torch.randn(1, 1, lq, lk, generator=g) * 2.0
            mask[..., : lk // 3] = float("-inf")
            mask[0, 0, 0, :] = float("-inf")
            if layout == "BNHD":
                q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            dq = [torch.round(t / self.S).clamp(-128, 127) * self.S for t in (q, k, v)]
            ref = torch.nan_to_num(_espdl_attention_reference(*dq, mask, scale=d**-0.5, is_causal=False, layout=layout))
            got = self._emu(q, k, v, mask, layout=layout)
            # Mask rounding to dot units is ~1e-3 LSB; the rest is the int8
            # softmax (7-bit e, LUT step), same as an unmasked peaky row.
            self.assertLessEqual((got - ref).abs().max().item(), 3 * self.S, f"{layout} {h, lq, lk, d}")


class TestInt8EmulatorCausal(unittest.TestCase):
    """Causal nodes reach the int8 emulator through ``espdl_attention_forward``."""

    S = 0.0234375  # 3/128
    SCALE = 0.125

    def _op(self, attributes, num_inputs=3):
        from types import SimpleNamespace

        from esp_ppq.core.quant import QuantizationStates

        cfg = SimpleNamespace(
            dominated_by=None,
            state=QuantizationStates.ACTIVATED,
            policy=SimpleNamespace(has_property=lambda _prop: True),
            num_of_bits=8,
            scale=torch.tensor(self.S),
        )
        op = _attention_op({"scale": self.SCALE, **attributes}, num_inputs=num_inputs)
        op.input_quant_config = [cfg] * num_inputs
        op.output_quant_config = [cfg]
        return op

    def _forward(self, op, values):
        from esp_ppq.executor.op.torch.espdl_attention import (
            espdl_attention_forward,
            set_espdl_attention_simulation_enabled,
        )

        set_espdl_attention_simulation_enabled(True)
        try:
            return espdl_attention_forward(op, values)[0]
        finally:
            set_espdl_attention_simulation_enabled(False)

    def _emu(self, q, k, v, mask):
        from esp_ppq.executor.op.torch.espdl_attention import _simulate_espdl_int8_attention

        return _simulate_espdl_int8_attention(
            q, k, v, self.SCALE, self.S, self.S, self.S, self.S, "BHND", attn_mask=mask
        )

    def _qkv(self, seed, lq=80, lk=80, h=2, d=32):
        g = torch.Generator().manual_seed(seed)
        return (
            torch.randn(1, h, lq, d, generator=g),
            torch.randn(1, h, lk, d, generator=g),
            torch.randn(1, h, lk, d, generator=g),
        )

    def test_square_causal_matches_lower_triangular_mask(self):
        q, k, v = self._qkv(0)
        out = self._forward(self._op({"is_causal": 1}), [q, k, v])
        keep = torch.ones(80, 80, dtype=torch.bool).tril()
        self.assertTrue(torch.equal(out, self._emu(q, k, v, keep)))
        self.assertFalse(torch.equal(out, self._emu(q, k, v, None)))

    def test_causal_with_mask_drops_causal_slots_first(self):
        q, k, v = self._qkv(1)
        g = torch.Generator().manual_seed(11)
        mask = torch.randn(1, 1, 80, 80, generator=g)
        mask[..., ::7] = float("-inf")
        mask[..., 0] = 0.0  # keep every row non-degenerate
        out = self._forward(self._op({"is_causal": 1}, num_inputs=4), [q, k, v, mask])
        keep = torch.ones(80, 80, dtype=torch.bool).tril()
        composed = torch.where(keep, mask, torch.full_like(mask, float("-inf")))
        self.assertTrue(torch.equal(out, self._emu(q, k, v, composed)))
        # Values above the frontier never reach the dots, however large.
        loud = torch.where(keep, mask, torch.full_like(mask, 1e4))
        self.assertTrue(torch.equal(self._forward(self._op({"is_causal": 1}, num_inputs=4), [q, k, v, loud]), out))

    def test_causal_with_bool_mask(self):
        q, k, v = self._qkv(2)
        g = torch.Generator().manual_seed(22)
        keep_mask = torch.rand(1, 1, 80, 80, generator=g) > 0.3
        keep_mask[..., 0] = True
        out = self._forward(self._op({"is_causal": 1}, num_inputs=4), [q, k, v, keep_mask])
        want = self._emu(q, k, v, keep_mask & torch.ones(80, 80, dtype=torch.bool).tril())
        self.assertTrue(torch.equal(out, want))

    def test_rectangular_causal_uses_device_int8_emulator(self):
        for lq, lk in ((40, 100), (100, 40)):
            with self.subTest(lq=lq, lk=lk):
                q, k, v = self._qkv(3, lq=lq, lk=lk)
                out = self._forward(self._op({"is_causal": 1}), [q, k, v])
                keep = torch.ones(lq, lk, dtype=torch.bool).tril()
                self.assertTrue(torch.equal(out, self._emu(q, k, v, keep)))


class TestInt16Emulator(unittest.TestCase):
    """int16 device-path emulator: integer QK + lookup_precise softmax."""

    S = 0.000244140625  # 2^-12

    def _emu(self, q, k, v, mask=None, layout="BHND", causal=False):
        from esp_ppq.executor.op.torch.espdl_attention import (
            _emulator_mask,
            _simulate_espdl_int16_attention,
        )

        d = int(q.shape[-1])
        l_q = int(q.shape[1] if layout == "BNHD" else q.shape[2])
        l_k = int(k.shape[1] if layout == "BNHD" else k.shape[2])
        scale = float(np.float32(1.0) / np.float32(np.sqrt(np.float32(d))))
        return _simulate_espdl_int16_attention(
            q,
            k,
            v,
            scale,
            self.S,
            self.S,
            self.S,
            self.S,
            layout,
            attn_mask=_emulator_mask(mask, causal, l_q, l_k, q.device),
        )

    def _qkv(self, seed, h=2, lq=16, lk=32, d=16):
        g = torch.Generator().manual_seed(seed)
        return (
            torch.randn(1, h, lq, d, generator=g),
            torch.randn(1, h, lk, d, generator=g),
            torch.randn(1, h, lk, d, generator=g),
        )

    def test_bnhd_matches_bhnd(self):
        q, k, v = self._qkv(0)
        out_bhnd = self._emu(q, k, v)
        q_n, k_n, v_n = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        out_bnhd = self._emu(q_n, k_n, v_n, layout="BNHD")
        self.assertTrue(torch.equal(out_bnhd, out_bhnd.transpose(1, 2).contiguous()))

    def test_zero_mask_matches_no_mask(self):
        q, k, v = self._qkv(1)
        mask = torch.zeros(1, 1, 16, 32)
        self.assertTrue(torch.equal(self._emu(q, k, v, mask), self._emu(q, k, v)))

    def test_fully_masked_row_is_zero(self):
        q, k, v = self._qkv(2)
        mask = torch.zeros(1, 1, 16, 32)
        mask[0, 0, 3, :] = float("-inf")
        out = self._emu(q, k, v, mask)
        self.assertTrue(torch.all(out[:, :, 3] == 0))
        self.assertTrue(torch.any(out[:, :, 2] != 0))

    def test_rectangular_causal_is_upper_left(self):
        for lq, lk in ((8, 20), (20, 8)):
            with self.subTest(lq=lq, lk=lk):
                q, k, v = self._qkv(3, lq=lq, lk=lk)
                got = self._emu(q, k, v, causal=True)
                keep = torch.ones(lq, lk, dtype=torch.bool).tril()
                want = self._emu(q, k, v, mask=keep)
                self.assertTrue(torch.equal(got, want))

    def test_forward_dispatches_int16_emulator(self):
        from types import SimpleNamespace

        from esp_ppq.core.quant import QuantizationStates
        from esp_ppq.executor.op.torch.espdl_attention import (
            espdl_attention_forward,
            set_espdl_attention_simulation_enabled,
        )

        cfg = SimpleNamespace(
            dominated_by=None,
            state=QuantizationStates.ACTIVATED,
            policy=SimpleNamespace(has_property=lambda _prop: True),
            num_of_bits=16,
            scale=torch.tensor(self.S),
        )
        op = _attention_op()
        op.input_quant_config = [cfg, cfg, cfg]
        op.output_quant_config = [cfg]
        q, k, v = self._qkv(4)
        set_espdl_attention_simulation_enabled(True)
        try:
            out = espdl_attention_forward(op, [q, k, v])[0]
        finally:
            set_espdl_attention_simulation_enabled(False)
        self.assertTrue(torch.equal(out, self._emu(q, k, v)))


class TestRuntimeMaskContract(unittest.TestCase):
    def test_integer_mask_is_rejected(self):
        from esp_ppq.executor.op.torch.espdl_attention import espdl_attention_forward

        op = _attention_op(num_inputs=4)
        q = torch.randn(1, 2, 4, 8)
        mask = torch.ones(1, 1, 4, 4, dtype=torch.int8)
        with self.assertRaises(ValueError) as ctx:
            espdl_attention_forward(op, [q, q, q, mask])
        self.assertIn("attn_mask dtype", str(ctx.exception))

    def test_unbroadcastable_mask_is_rejected(self):
        from esp_ppq.executor.op.torch.espdl_attention import espdl_attention_forward

        op = _attention_op(num_inputs=4)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 4, 7)
        with self.assertRaises(ValueError) as ctx:
            espdl_attention_forward(op, [q, k, v, mask])
        self.assertIn("broadcastable", str(ctx.exception))


class TestEligibilityShapeConsistency(unittest.TestCase):
    """Shape-level invariants of the fused kernel (batch / GQA / Lk / D)."""

    def test_rejects_k_v_lk_mismatch(self):
        op = _attention_op()
        shapes = [[1, 4, 16, 32], [1, 4, 32, 32], [1, 4, 64, 32]]
        self.assertFalse(espdl_attention_eligible(op, shapes))

    def test_rejects_h_kv_zero(self):
        op = _attention_op()
        shapes = [[1, 4, 16, 32], [1, 0, 16, 32], [1, 0, 16, 32]]
        self.assertFalse(espdl_attention_eligible(op, shapes))

    def test_rejects_non_divisible_gqa(self):
        op = _attention_op()
        shapes = [[1, 7, 16, 32], [1, 3, 16, 32], [1, 3, 16, 32]]
        self.assertFalse(espdl_attention_eligible(op, shapes))

    def test_rejects_batch_mismatch(self):
        op = _attention_op()
        shapes = [[1, 4, 16, 32], [2, 4, 16, 32], [2, 4, 16, 32]]
        self.assertFalse(espdl_attention_eligible(op, shapes))

    def test_rejects_head_dim_mismatch(self):
        op = _attention_op()
        shapes = [[1, 4, 16, 32], [1, 4, 16, 32], [1, 4, 16, 64]]
        self.assertFalse(espdl_attention_eligible(op, shapes))
        shapes = [[1, 4, 16, 32], [1, 4, 16, 64], [1, 4, 16, 64]]
        self.assertFalse(espdl_attention_eligible(op, shapes))

    def test_rejects_symbolic_axis(self):
        op = _attention_op()
        shapes = [[1, 4, "Lq", 32]] * 3
        self.assertFalse(espdl_attention_eligible(op, shapes))

    def test_layout_bytes_is_accepted(self):
        op = _attention_op({"layout": b"BHND"})
        self.assertTrue(espdl_attention_eligible(op, [[1, 4, 16, 32]] * 3))

    def test_layout_bytes_bnhd_is_accepted(self):
        op = _attention_op({"layout": b"BNHD"})
        self.assertTrue(espdl_attention_eligible(op, [[1, 16, 4, 32]] * 3))

    def test_shapes_helper_agrees_with_full_check(self):
        op = _attention_op()
        cases = [
            [[1, 4, 16, 32]] * 3,
            [[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM]] * 3,
            [[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3,
            [[1, 8, 16, 32], [1, 2, 16, 32], [1, 2, 16, 32]],
            [[1, 7, 16, 32], [1, 3, 16, 32], [1, 3, 16, 32]],
            [[1, 4, 16, 32], [2, 4, 16, 32], [2, 4, 16, 32]],
            [[1, 4, 16, 32], [1, 4, 16, 32], [1, 4, 16, 64]],
            [[1, 4, "Lq", 32]] * 3,
            [[1, 16, 96]] * 3,
            [None, None, None],
            [[1, 4, ESPDL_ATTENTION_MAX_KV_TOKENS, 32]] * 3,
            [[1, 4, ESPDL_ATTENTION_MAX_KV_TOKENS + 1, 32]] * 3,
        ]
        for shapes in cases:
            with self.subTest(shapes=shapes):
                self.assertEqual(espdl_attention_shapes_eligible(shapes), espdl_attention_eligible(op, shapes))
                self.assertEqual(
                    espdl_attention_shapes_ineligible_reason(shapes),
                    espdl_attention_ineligible_reason(op, shapes),
                )
        bnhd_op = _attention_op({"layout": "BNHD"})
        bnhd_ok = [[1, 16, 4, 32]] * 3
        bnhd_gqa_bad = [[1, 16, 3, 32], [1, 16, 2, 32], [1, 16, 2, 32]]
        self.assertEqual(
            espdl_attention_shapes_eligible(bnhd_ok, layout="BNHD"),
            espdl_attention_eligible(bnhd_op, bnhd_ok),
        )
        self.assertFalse(espdl_attention_eligible(bnhd_op, bnhd_gqa_bad))
        self.assertIn("Hq", espdl_attention_ineligible_reason(bnhd_op, bnhd_gqa_bad))


class TestFusionGuards(unittest.TestCase):
    """Regressions for fusion bugs that would corrupt working graphs."""

    @staticmethod
    def _load_and_fuse(module, feed, output_names=("out",)):
        """Export ``module`` to ONNX, trace shapes, run fusion, and return the graph."""
        graph, _ = _export_trace_and_fuse(module, tuple(feed.values()), list(feed.keys()), output_names)
        return graph

    def test_3d_qkv_fuses_and_2d_qkv_is_not_fused(self):
        from esp_ppq.executor import TorchExecutor

        class M(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / 4.0, -1), v)

        q3 = torch.randn(2, 16, 32)
        feed3 = {"q": q3, "k": q3.clone(), "v": q3.clone()}
        golden3 = M()(*feed3.values())
        graph3 = self._load_and_fuse(M(), feed3)
        attentions = [op for op in graph3.operations.values() if op.type == "Attention"]
        self.assertEqual(len(attentions), 1)
        self.assertEqual([list(v.shape) for v in attentions[0].inputs[:3]], [[1, 2, 16, 32]] * 3)
        ordered3 = [feed3[n] for n in graph3.inputs.keys()]
        out3 = TorchExecutor(graph=graph3, device="cpu").forward(inputs=ordered3)[0]
        torch.testing.assert_close(out3, golden3, rtol=1e-4, atol=1e-4)

        q2 = torch.randn(16, 32)
        graph2 = self._load_and_fuse(M(), {"q": q2, "k": q2.clone(), "v": q2.clone()})
        self.assertFalse(any(op.type == "Attention" for op in graph2.operations.values()))

    def test_missing_softmax_axis_follows_opset_default(self):
        """Opset < 13 omits axis and means axis 1, which is not the last dim."""
        import os
        import tempfile

        from onnx import TensorProto, helper

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

        def write(path, opset, axis):
            softmax_kwargs = {} if axis is None else {"axis": axis}
            nodes = [
                helper.make_node("Transpose", ["k"], ["kt"], name="kT", perm=[0, 1, 3, 2]),
                helper.make_node("MatMul", ["q", "kt"], ["scores"], name="qk"),
                helper.make_node("Softmax", ["scores"], ["probs"], name="sm", **softmax_kwargs),
                helper.make_node("MatMul", ["probs", "v"], ["out"], name="pv"),
            ]
            inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2, 8, 16]) for name in ("q", "k", "v")]
            graph = helper.make_graph(
                nodes,
                "g",
                inputs,
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2, 8, 8])],
            )
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)]), path)

        cases = (
            ("opset11_default", 11, None, False),
            ("opset11_explicit_last", 11, -1, True),
            ("opset18_default", 18, None, True),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for label, opset, axis, expect_attention in cases:
                with self.subTest(label=label):
                    path = os.path.join(tmp, f"{label}.onnx")
                    write(path, opset, axis)
                    graph = load_onnx_graph(path)
                    executor = TorchExecutor(graph=graph, device="cpu")
                    feed = [torch.zeros(1, 2, 8, 16) for _ in graph.inputs]
                    fuse_espdl_attention_graph(graph, executor=executor, inputs=feed)
                    fused = any(op.type == "Attention" for op in graph.operations.values())
                    self.assertEqual(fused, expect_attention)
                    if not expect_attention:
                        softmax = next(op for op in graph.operations.values() if op.type == "Softmax")
                        self.assertNotIn("axis", softmax.attributes)

    def test_known_non4d_is_not_fused(self):
        """A known rank other than 4 stays MatMul. An unshaped Q stays MatMul too."""
        import os
        import tempfile

        from onnx import TensorProto, helper, numpy_helper

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.core import TargetPlatform
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import (
            fuse_espdl_attention_graph,
            legalize_espdl_attention_graph,
        )

        def save(path, nodes, inputs, initializers=()):
            graph = helper.make_graph(
                nodes,
                "g",
                inputs,
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
                initializer=list(initializers),
            )
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)

        def attention_chain(q, k, v, extra_nodes=(), mask=None):
            nodes = list(extra_nodes)
            nodes.append(helper.make_node("Transpose", [k], ["kt"], name="kT", perm=[0, 1, 3, 2]))
            score = "scores"
            nodes.append(helper.make_node("MatMul", [q, "kt"], [score], name="qk"))
            if mask is not None:
                nodes.append(helper.make_node("Add", [score, mask], ["biased"], name="bias"))
                score = "biased"
            nodes.append(helper.make_node("Softmax", [score], ["probs"], name="sm", axis=-1))
            nodes.append(helper.make_node("MatMul", ["probs", v], ["out"], name="pv"))
            return nodes

        with tempfile.TemporaryDirectory() as tmp:
            cases = []
            mixed = os.path.join(tmp, "mixed.onnx")
            save(
                mixed,
                attention_chain("q", "k", "v"),
                [
                    helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 8, 16]),
                    helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 8, 16]),
                    helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 8, 16]),
                ],
            )
            cases.append(("mixed_q_rank", mixed, {"q": [1, 8, 16], "k": [1, 2, 8, 16], "v": [1, 2, 8, 16]}, False))

            unknown = os.path.join(tmp, "unknown.onnx")
            save(
                unknown,
                attention_chain(
                    "q_relu", "k", "v", extra_nodes=[helper.make_node("Relu", ["x"], ["q_relu"], name="rq")]
                ),
                [
                    helper.make_tensor_value_info("x", TensorProto.FLOAT, None),
                    helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 2, 8, 16]),
                    helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 2, 8, 16]),
                ],
            )
            cases.append(("unknown_q", unknown, {"x": None, "k": [1, 2, 8, 16], "v": [1, 2, 8, 16]}, False))

            wide_mask = os.path.join(tmp, "mask.onnx")
            save(
                wide_mask,
                attention_chain("q", "k", "v", mask="mask"),
                [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2, 8, 16]) for name in ("q", "k", "v")]
                + [helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 1, 8, 8])],
            )
            cases.append(
                (
                    "rank5_mask",
                    wide_mask,
                    {"q": [1, 2, 8, 16], "k": [1, 2, 8, 16], "v": [1, 2, 8, 16], "mask": [1, 1, 1, 8, 8]},
                    False,
                )
            )

            shape = numpy_helper.from_array(np.array([1, 2, 8, 16], dtype=np.int64), "shape4")
            reshaped = os.path.join(tmp, "reshape.onnx")
            reshape_nodes = [
                helper.make_node("Reshape", [name, "shape4"], [f"{name}4"], name=f"rs_{name}")
                for name in ("q", "k", "v")
            ]
            save(
                reshaped,
                attention_chain("q4", "k4", "v4", extra_nodes=reshape_nodes),
                [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 8, 32]) for name in ("q", "k", "v")],
                initializers=[shape],
            )
            cases.append(("reshape_to_4d", reshaped, {"q": [1, 8, 32], "k": [1, 8, 32], "v": [1, 8, 32]}, True))

            mask4_on_3d = os.path.join(tmp, "mask4_on_3d.onnx")
            nodes_3d_mask4 = [
                helper.make_node("Transpose", ["k"], ["kt"], name="kT", perm=[0, 2, 1]),
                helper.make_node("MatMul", ["q", "kt"], ["scores"], name="qk"),
                helper.make_node("Add", ["scores", "mask"], ["biased"], name="bias"),
                helper.make_node("Softmax", ["biased"], ["probs"], name="sm", axis=-1),
                helper.make_node("MatMul", ["probs", "v"], ["out"], name="pv"),
            ]
            save(
                mask4_on_3d,
                nodes_3d_mask4,
                [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 8, 16]) for name in ("q", "k", "v")]
                + [helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 2, 8, 8])],
            )
            cases.append(
                (
                    "rank4_mask_on_3d",
                    mask4_on_3d,
                    {"q": [2, 8, 16], "k": [2, 8, 16], "v": [2, 8, 16], "mask": [1, 2, 8, 8]},
                    False,
                )
            )

            for label, path, input_shapes, expect_attention in cases:
                with self.subTest(label=label):
                    graph = load_onnx_graph(path)
                    self.assertTrue(all(var.shape is None for var in graph.inputs.values()))
                    self.assertTrue(all(not hasattr(var, "_onnx_shape") for var in graph.variables.values()))
                    # Without traced shapes, no candidate fuses.
                    fuse_espdl_attention_graph(graph)
                    self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
                    if all(s is not None for s in input_shapes.values()):
                        feed = [torch.zeros(input_shapes[name]) for name in graph.inputs.keys()]
                        TorchExecutor(graph=graph, device="cpu").tracing_operation_meta(inputs=feed)
                    else:
                        for name, s in input_shapes.items():
                            graph.inputs[name].shape = s
                    fuse_espdl_attention_graph(graph)
                    fused = any(op.type == "Attention" for op in graph.operations.values())
                    self.assertEqual(fused, expect_attention)
                    if expect_attention:
                        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8, quantized=False)
                        self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))

    def test_prequant_fuses_rank4_and_rank3_and_leaves_rank2(self):
        """Loading does not fuse. Prequant fuses perm length 3 and 4 and leaves 2."""
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph, quantize_onnx_model
        from esp_ppq.api.setting import QuantizationSettingFactory
        from esp_ppq.core import TargetPlatform

        class M(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / 4.0, -1), v)

        cases = {
            "rank4": (1, 2, 8, 16),
            "rank3": (1, 16, 32),
            "rank2": (16, 32),
        }
        for label, shape in cases.items():
            with self.subTest(rank=label):
                torch.manual_seed(0)
                q = torch.randn(*shape)
                feed = {"q": q, "k": q.clone(), "v": q.clone()}
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "m.onnx")
                    with _silence_export_logs():
                        torch.onnx.export(
                            M(),
                            tuple(feed.values()),
                            path,
                            input_names=list(feed.keys()),
                            output_names=["out"],
                            opset_version=18,
                            dynamo=False,
                        )
                        loaded = load_onnx_graph(path)
                    self.assertFalse(any(op.type == "Attention" for op in loaded.operations.values()))
                    sample = [feed[name] for name in loaded.inputs.keys()]
                    with _silence_export_logs():
                        quantized = quantize_onnx_model(
                            path,
                            calib_dataloader=[sample, [t.clone() for t in sample]],
                            calib_steps=2,
                            input_shape=None,
                            platform=TargetPlatform.ESPDL_INT8,
                            inputs=sample,
                            setting=QuantizationSettingFactory.espdl_setting(),
                            device="cpu",
                            verbose=0,
                        )
                kept = any(op.type == "Attention" for op in quantized.operations.values())
                self.assertEqual(kept, label in ("rank4", "rank3"))
                if label == "rank2":
                    self.assertTrue(any(op.type == "MatMul" for op in quantized.operations.values()))
                    self.assertTrue(any(op.type == "Softmax" for op in quantized.operations.values()))

    def test_attention_fusion_setting_controls_quantized_and_fp32_export(self):
        """The same setting must disable primitive fusion on both export paths."""
        import os
        import tempfile

        from esp_ppq.api import quantize_onnx_model
        from esp_ppq.api.espdl_interface import espdl_quantize_onnx
        from esp_ppq.api.setting import QuantizationSettingFactory
        from esp_ppq.core import TargetPlatform

        class M(torch.nn.Module):
            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
                return torch.matmul(torch.softmax(scores, -1), v)

        q = torch.randn(1, 2, 8, 16)
        sample = [q, q.clone(), q.clone()]
        setting = QuantizationSettingFactory.espdl_setting()
        setting.espdl_attention_fusion = False
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    M(),
                    tuple(sample),
                    path,
                    input_names=["q", "k", "v"],
                    output_names=["out"],
                    opset_version=18,
                    dynamo=False,
                )
                quantized = quantize_onnx_model(
                    path,
                    calib_dataloader=[sample, [tensor.clone() for tensor in sample]],
                    calib_steps=2,
                    input_shape=None,
                    platform=TargetPlatform.ESPDL_INT8,
                    inputs=sample,
                    setting=setting,
                    device="cpu",
                    verbose=0,
                )
                fp32 = espdl_quantize_onnx(
                    onnx_import_file=path,
                    espdl_export_file=os.path.join(tmp, "float.espdl"),
                    calib_dataloader=[sample],
                    calib_steps=1,
                    input_shape=[list(tensor.shape) for tensor in sample],
                    inputs=sample,
                    target="esp32p4",
                    quant_type="none",
                    setting=setting,
                    device="cpu",
                    error_report=False,
                    skip_export=True,
                    verbose=0,
                )

        for graph in (quantized, fp32):
            self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
            self.assertTrue(any(op.type == "MatMul" for op in graph.operations.values()))
            self.assertTrue(any(op.type == "Softmax" for op in graph.operations.values()))

    def test_oversized_head_dim_is_not_fused(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / 4.0, -1), v)

        torch.manual_seed(0)
        q = torch.randn(1, 2, 8, ESPDL_ATTENTION_MAX_HEAD_DIM * 2)
        graph = self._load_and_fuse(M(), {"q": q, "k": q.clone(), "v": q.clone()})
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))

    def test_oversized_head_dim_leaves_bnhd_transpose_untouched(self):
        # shape check must run before the in-place BNHD rewrite
        class M(torch.nn.Module):
            def forward(self, q, k_bnhd, v):
                k_kt = k_bnhd.permute(0, 2, 3, 1)  # [B,N,H,D] -> [B,H,D,N]
                return torch.matmul(torch.softmax(torch.matmul(q, k_kt) / 4.0, -1), v)

        torch.manual_seed(0)
        B, H, L, D = 1, 2, 8, ESPDL_ATTENTION_MAX_HEAD_DIM * 2
        feed = {"q": torch.randn(B, H, L, D), "k_bnhd": torch.randn(B, L, H, D), "v": torch.randn(B, H, L, D)}
        golden = M()(*feed.values())

        graph = self._load_and_fuse(M(), feed)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        transposes = [op for op in graph.operations.values() if op.type == "Transpose"]
        self.assertTrue(transposes, "the BNHD Transpose must survive an unfused match")
        for t in transposes:
            self.assertEqual(list(t.attributes["perm"]), [0, 2, 3, 1], "Transpose was rewritten for a rejected match")

        from esp_ppq.executor import TorchExecutor

        ordered = [feed[n] for n in graph.inputs.keys()]
        out = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)[0]
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_softmax_output_used_elsewhere_is_not_fused(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v):
                attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / 4.0, -1)
                return torch.matmul(attn, v), attn

        q = torch.randn(1, 2, 8, 16)
        graph = self._load_and_fuse(M(), {"q": q, "k": q.clone(), "v": q.clone()}, output_names=("out", "attn"))
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))

    def test_scores_used_elsewhere_is_not_fused(self):
        class M(torch.nn.Module):
            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
                return torch.matmul(torch.softmax(scores, -1), v), scores

        q = torch.randn(1, 2, 8, 16)
        graph = self._load_and_fuse(M(), {"q": q, "k": q.clone(), "v": q.clone()}, output_names=("out", "scores"))
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))

    def test_permute_bnhd_k_transpose_fuses_via_rewrite(self):
        class M(torch.nn.Module):
            def forward(self, q, k_bnhd, v):
                k_kt = k_bnhd.permute(0, 2, 3, 1)  # [B,N,H,D] -> [B,H,D,N]
                return torch.matmul(torch.softmax(torch.matmul(q, k_kt) / 4.0, -1), v)

        torch.manual_seed(0)
        B, H, L, D = 1, 2, 8, 16
        q = torch.randn(B, H, L, D)
        k_bnhd = torch.randn(B, L, H, D)
        v = torch.randn(B, H, L, D)
        golden = M()(q, k_bnhd, v)

        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    M(), (q, k_bnhd, v), path, input_names=["q", "k", "v"], output_names=["out"], opset_version=18
                )
                graph = load_onnx_graph(path)
                feed = [{"q": q, "k": k_bnhd, "v": v}[n] for n in graph.inputs.keys()]
                TorchExecutor(graph=graph, device="cpu").tracing_operation_meta(inputs=feed)
                fuse_espdl_attention_graph(graph)
                out = TorchExecutor(graph=graph, device="cpu").forward(inputs=feed)[0]

        types = sorted(o.type for o in graph.operations.values())
        self.assertIn("Attention", types)
        self.assertIn("Transpose", types)
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)

    def test_where_as_mask_is_logged_as_miss(self):
        # Where(cond, scores, -inf) is not fused; the export summary says so at INFO.
        class M(torch.nn.Module):
            def forward(self, q, k, v, cond):
                s = torch.matmul(q, k.transpose(-1, -2)) / 4.0
                s = torch.where(cond, s, torch.full_like(s, -1e4))
                return torch.matmul(torch.softmax(s, -1), v)

        import os
        import tempfile
        from unittest.mock import patch

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.core import TargetPlatform
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim import espdl_attention as fusion_mod
        from esp_ppq.quantization.optim.espdl_attention import (
            fuse_espdl_attention_graph,
            legalize_espdl_attention_graph,
        )

        B, H, L, D = 1, 2, 8, 16
        q = torch.randn(B, H, L, D)
        cond = torch.ones(B, 1, L, L, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    M(),
                    (q, q.clone(), q.clone(), cond),
                    path,
                    input_names=["q", "k", "v", "cond"],
                    output_names=["out"],
                    opset_version=18,
                )
                graph = load_onnx_graph(path)
                feed = [{"q": q, "k": q.clone(), "v": q.clone(), "cond": cond}[n] for n in graph.inputs.keys()]
                TorchExecutor(graph=graph, device="cpu").tracing_operation_meta(inputs=feed)
                with patch.object(fusion_mod.logger, "info") as mock_info:
                    fuse_espdl_attention_graph(graph)
                    legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8, quantized=False)
        info_messages = " | ".join(call.args[0] for call in mock_info.call_args_list if call.args)
        self.assertIn("Where(mask)", info_messages)


class TestIneligibleReason(unittest.TestCase):
    """``espdl_attention_ineligible_reason`` must explain every rejection."""

    # (label, attributes, num_inputs, shapes, expected substring)
    REJECTIONS = (
        ("microsoft", {"qkv_hidden_sizes": [96, 96, 96]}, 3, [[1, 4, 16, 32]] * 3, "com.microsoft"),
        ("softcap", {"softcap": 1}, 3, [[1, 4, 16, 32]] * 3, "unsupported attrs"),
        ("window", {"left_window_size": 4}, 3, [[1, 4, 16, 32]] * 3, "unsupported attrs"),
        ("unknown", {"not_a_real_attr": 1}, 3, [[1, 4, 16, 32]] * 3, "unsupported attrs"),
        ("scale_zero", {"scale": 0.0}, 3, [[1, 4, 16, 32]] * 3, "finite, positive, normal float32"),
        ("scale_negative", {"scale": -0.125}, 3, [[1, 4, 16, 32]] * 3, "finite, positive, normal float32"),
        ("scale_nan", {"scale": float("nan")}, 3, [[1, 4, 16, 32]] * 3, "finite, positive, normal float32"),
        ("scale_inf", {"scale": float("inf")}, 3, [[1, 4, 16, 32]] * 3, "finite, positive, normal float32"),
        ("scale_subnormal", {"scale": 1e-40}, 3, [[1, 4, 16, 32]] * 3, "finite, positive, normal float32"),
        ("scale_f32_overflow", {"scale": 1e39}, 3, [[1, 4, 16, 32]] * 3, "finite, positive, normal float32"),
        ("kv_cache", {}, 5, [[1, 4, 16, 32]] * 3, "KV cache"),
        ("layout", {"layout": "BNSD"}, 3, [[1, 16, 4, 32]] * 3, "unsupported"),
        ("missing_shapes", {}, 3, [[1, 4, 16, 32]], "shape metadata"),
        ("rank", {}, 3, [[1, 16, 96]] * 3, "not all 4-D"),
        ("no_metadata", {}, 3, [None, None, None], "not all 4-D"),
        ("symbolic", {}, 3, [[1, 4, 16, "D"]] * 3, "non-static dims"),
        ("head_dim_hi", {}, 3, [[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3, "outside [1,"),
        ("head_dim_zero", {}, 3, [[1, 4, 16, 0]] * 3, "outside [1,"),
        ("batch", {}, 3, [[2, 4, 16, 32], [1, 4, 16, 32], [1, 4, 16, 32]], "batch mismatch"),
        ("kv_disagree", {}, 3, [[1, 4, 16, 32], [1, 4, 16, 32], [1, 4, 8, 32]], "(Hkv, Lk)"),
        ("v_head_size", {}, 3, [[1, 4, 16, 32], [1, 4, 16, 32], [1, 4, 16, 16]], "v_head_size"),
        ("gqa", {}, 3, [[1, 6, 16, 32], [1, 4, 16, 32], [1, 4, 16, 32]], "multiple of Hkv"),
        ("lk_hi", {}, 3, [[1, 4, ESPDL_ATTENTION_MAX_KV_TOKENS + 1, 32]] * 3, "MAX_KV_TOKENS"),
        ("lk_zero", {}, 3, [[1, 4, 0, 32]] * 3, "MAX_KV_TOKENS"),
        ("q_heads_mismatch", {"q_num_heads": 8}, 3, [[1, 4, 16, 32]] * 3, "q_num_heads=8"),
        ("kv_heads_mismatch", {"kv_num_heads": 1}, 3, [[1, 4, 16, 32]] * 3, "kv_num_heads=1"),
        (
            "bnhd_heads_use_seq_axis",
            {"layout": "BNHD", "q_num_heads": 16, "kv_num_heads": 16},
            3,
            [[1, 16, 4, 32]] * 3,
            "q_num_heads=16",
        ),
        ("heads_not_integer", {"q_num_heads": "4"}, 3, [[1, 4, 16, 32]] * 3, "not a positive integer"),
    )

    def test_every_rejection_has_a_reason(self):
        for label, attrs, num_inputs, shapes, expected in self.REJECTIONS:
            with self.subTest(case=label):
                op = _attention_op(attrs, num_inputs=num_inputs)
                self.assertFalse(espdl_attention_eligible(op, shapes))
                reason = espdl_attention_ineligible_reason(op, shapes)
                self.assertTrue(reason, "no reason produced for a rejected node")
                self.assertIn(expected, reason)

    def test_eligible_node_has_no_reason(self):
        for label, attrs, shapes in (
            ("mha", {}, [[1, 4, 16, 32]] * 3),
            ("gqa", {}, [[1, 8, 32, 32], [1, 2, 32, 32], [1, 2, 32, 32]]),
            ("causal", {"is_causal": 1}, [[1, 4, 16, 32]] * 3),
            ("explicit_scale", {"scale": 0.125}, [[1, 4, 16, 32]] * 3),
            (
                "default_noops",
                {"softcap": 0.0, "qk_matmul_output_mode": 0, "left_window_size": -1},
                [[1, 4, 16, 32]] * 3,
            ),
            ("bnhd", {"layout": "BNHD"}, [[1, 16, 4, 32]] * 3),
            ("matching_heads", {"q_num_heads": 4, "kv_num_heads": 4}, [[1, 4, 16, 32]] * 3),
            ("zero_heads_mean_infer", {"q_num_heads": 0, "kv_num_heads": 0}, [[1, 4, 16, 32]] * 3),
            (
                "gqa_declared_heads",
                {"q_num_heads": 8, "kv_num_heads": 2},
                [[1, 8, 32, 32], [1, 2, 32, 32], [1, 2, 32, 32]],
            ),
            (
                "bnhd_declared_heads",
                {"layout": "BNHD", "q_num_heads": 4, "kv_num_heads": 4},
                [[1, 16, 4, 32]] * 3,
            ),
            ("head_dim_bound", {}, [[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM]] * 3),
            ("lk_bound", {}, [[1, 4, ESPDL_ATTENTION_MAX_KV_TOKENS, 32]] * 3),
        ):
            with self.subTest(case=label):
                op = _attention_op(attrs)
                self.assertTrue(espdl_attention_eligible(op, shapes))
                self.assertEqual(espdl_attention_ineligible_reason(op, shapes), "")


class TestDeployabilityGuard(unittest.TestCase):
    """Ineligible fused Attention is expanded to MatMul/Softmax, or refused."""

    @staticmethod
    def _graph_with(shapes, attributes=None, num_inputs: int = 3):
        from esp_ppq.IR import BaseGraph

        graph = BaseGraph(name="g")
        ins = []
        for i in range(num_inputs):
            shape = None if i >= len(shapes) else shapes[i]
            var = graph.create_variable(name=f"in{i}")
            if shape is not None:
                var.shape = shape
            ins.append(var)
            if i < 3:
                graph.mark_variable_as_graph_input(var)
        out = graph.create_variable(name="out")
        graph.create_operation(
            op_type="Attention",
            name="attn",
            attributes=dict(attributes or {}),
            inputs=ins,
            outputs=[out],
        )
        graph.mark_variable_as_graph_output(out)
        return graph

    def setUp(self):
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        self.check = legalize_espdl_attention_graph

    def test_accepts_eligible_attention(self):
        self.check(self._graph_with([[1, 4, 16, 32]] * 3))

    def test_oversized_head_dim_is_decomposed(self):
        graph = self._graph_with([[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3)
        self.check(graph)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "MatMul" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "Softmax" for op in graph.operations.values()))
        self.assertIn("out", graph.outputs)

    def test_scale_only_oversized_node_is_decomposed(self):
        graph = self._graph_with([[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3, {"scale": 0.125})
        self.check(graph)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "Mul" for op in graph.operations.values()))

    def test_non_positive_scale_is_decomposed(self):
        for scale in (0.0, -0.125):
            with self.subTest(scale=scale):
                graph = self._graph_with([[1, 4, 16, 32]] * 3, {"scale": scale})
                self.check(graph)
                self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
                self.assertTrue(any(op.type == "Softmax" for op in graph.operations.values()))

    def test_native_oversized_node_is_decomposed(self):
        graph = self._graph_with(
            [[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3,
            {"q_num_heads": 4, "kv_num_heads": 4},
        )
        self.check(graph)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))

    def test_mismatched_head_attr_is_refused_not_decomposed(self):
        # Expanding would follow the tensor shape and drop q_num_heads, which
        # is the same silent ignore the device export used to do.
        graph = self._graph_with([[1, 4, 16, 32]] * 3, {"q_num_heads": 8, "kv_num_heads": 4})
        with self.assertRaises(ValueError) as ctx:
            self.check(graph)
        self.assertIn("q_num_heads=8", str(ctx.exception))
        self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertFalse(any(op.type == "MatMul" for op in graph.operations.values()))

    def test_non_broadcastable_mask_is_refused_not_decomposed(self):
        # Not a fused-kernel limitation — such a mask breaks ai.onnx::Attention
        # itself, and expanding it would push the failure somewhere the mask is
        # no longer visible.
        graph = self._graph_with([[1, 4, 16, 32]] * 3 + [[1, 1, 16, 8]], num_inputs=4)
        with self.assertRaises(ValueError) as ctx:
            self.check(graph)
        self.assertIn("broadcastable", str(ctx.exception))
        self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))

    def test_integer_mask_is_refused_not_decomposed(self):
        from esp_ppq.core import DataType

        for dtype in (DataType.INT8, DataType.INT16):
            with self.subTest(dtype=dtype):
                graph = self._graph_with([[1, 4, 16, 32]] * 3 + [[1, 1, 16, 16]], num_inputs=4)
                graph.variables["in3"].dtype = dtype
                with self.assertRaises(ValueError) as ctx:
                    self.check(graph)
                self.assertIn("neither bool nor float", str(ctx.exception))
                self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))

    def test_bool_mask_stays_fused(self):
        from esp_ppq.core import DataType

        graph = self._graph_with([[1, 4, 16, 32]] * 3 + [[1, 1, 16, 16]], num_inputs=4)
        graph.variables["in3"].dtype = DataType.BOOL
        self.check(graph)
        self.assertEqual(sum(op.type == "Attention" for op in graph.operations.values()), 1)

    def test_unmarked_3d_attention_is_still_refused(self):
        graph = self._graph_with([[1, 8, 16]] * 3, {"scale": 1.0})
        with self.assertRaises(ValueError) as ctx:
            self.check(graph)
        self.assertIn("q_num_heads", str(ctx.exception))
        self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))

    def test_high_rank_mask_is_refused_even_with_leading_ones(self):
        # Squeezing [1,1,1,1,16,16] to [1,1,16,16] looks harmless but is not:
        # ONNX Add would produce rank 6, so the device would then disagree with
        # the fp32 reference about the output shape.
        for mask in ([1, 1, 1, 1, 16, 16], [2, 1, 4, 16, 16]):
            with self.subTest(mask=mask):
                graph = self._graph_with([[1, 4, 16, 32]] * 3 + [mask], num_inputs=4)
                with self.assertRaises(ValueError) as ctx:
                    self.check(graph)
                self.assertIn("does not lower rank", str(ctx.exception))

    def test_oversized_lk_is_decomposed(self):
        graph = self._graph_with([[1, 4, ESPDL_ATTENTION_MAX_KV_TOKENS + 1, 32]] * 3)
        self.check(graph)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "MatMul" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "Softmax" for op in graph.operations.values()))
        self.assertIn("out", graph.outputs)

    def test_unknown_shapes_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.check(self._graph_with([None, None, None]))
        self.assertIn("shapes are still", str(ctx.exception))

    def test_tracing_rescues_shapes_before_legalize(self):
        graph = self._graph_with([None, None, None])

        class _Executor:
            def tracing_operation_meta(self, inputs=None):
                for var in graph.operations["attn"].inputs[:3]:
                    var.shape = [1, 4, 16, 32]

        self.check(graph, executor=_Executor(), inputs=object())

    def test_connected_unshaped_mask_is_rejected(self):
        graph = self._graph_with([[1, 4, 16, 32]] * 3, num_inputs=4)
        with self.assertRaises(ValueError) as ctx:
            self.check(graph)
        self.assertIn("attn_mask", str(ctx.exception))

    def test_bnhd_layout_stays_fused(self):
        graph = self._graph_with([[1, 16, 4, 32]] * 3, {"layout": "BNHD"})
        self.check(graph)
        attn = [op for op in graph.operations.values() if op.type == "Attention"]
        self.assertEqual(len(attn), 1)
        self.assertEqual(str(attn[0].attributes.get("layout", "")).upper(), "BNHD")

    def test_default_softcap_stays_fused(self):
        graph = self._graph_with([[1, 4, 16, 32]] * 3, {"softcap": 0.0, "is_causal": 0})
        self.check(graph)
        attn = [op for op in graph.operations.values() if op.type == "Attention"]
        self.assertEqual(len(attn), 1)
        self.assertEqual(attn[0].attributes.get("softcap"), 0.0)
        self.assertEqual(attn[0].attributes.get("is_causal"), 0)

    def test_non_default_window_is_refused_not_decomposed(self):
        graph = self._graph_with([[1, 4, 16, 32]] * 3, {"left_window_size": 8})
        with self.assertRaises(ValueError) as ctx:
            self.check(graph)
        self.assertIn("unsupported attrs", str(ctx.exception))
        self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))

    def test_rejects_kv_cache(self):
        with self.assertRaises(ValueError) as ctx:
            self.check(self._graph_with([[1, 4, 16, 32]] * 3, num_inputs=5))
        self.assertIn("KV cache", str(ctx.exception))

    def test_ignores_other_op_types(self):
        from esp_ppq.IR import BaseGraph

        graph = BaseGraph(name="g")
        graph.create_operation(op_type="MatMul", name="mm")
        self.check(graph)

    def test_microsoft_attention_is_also_refused(self):
        graph = self._graph_with([[1, 4, 16, 32]] * 3, {"qkv_hidden_sizes": [96, 96, 96], "num_heads": 4})
        with self.assertRaises(ValueError) as ctx:
            self.check(graph)
        self.assertIn("com.microsoft", str(ctx.exception).lower())

    def test_quantize_runs_legalize_then_finalize(self):
        from unittest import mock

        from esp_ppq.api.setting import QuantizationSettingFactory
        from esp_ppq.core import TargetPlatform
        from esp_ppq.IR import BaseGraph
        from esp_ppq.quantization.optim.espdl_attention import (
            EspdlAttentionFusionPass,
            EspdlAttentionLegalizePass,
        )
        from esp_ppq.quantization.optim.rnn_arity_normalization import (
            RNNArityNormalizationPass,
        )
        from esp_ppq.quantization.quantizer.base import BaseQuantizer
        from esp_ppq.quantization.quantizer.EspdlQuantizer import BaseEspdlQuantizer

        graph = self._graph_with([[1, 4, 16, 32]] * 3)
        quantizer = BaseEspdlQuantizer(graph)
        executor = mock.Mock(name="executor")
        inputs = mock.Mock(name="inputs")
        setting = QuantizationSettingFactory.espdl_setting()
        passes = list(quantizer.build_prequant_pipeline(setting, executor))
        self.assertIsInstance(passes[0], RNNArityNormalizationPass)
        self.assertIsInstance(passes[1], EspdlAttentionFusionPass)
        self.assertIsInstance(passes[2], EspdlAttentionLegalizePass)
        self.assertEqual(passes[2].platform, TargetPlatform.ESPDL_INT8)
        self.assertTrue(passes[2].quantized)

        order = []
        with (
            mock.patch(
                "esp_ppq.quantization.optim.espdl_attention.fuse_espdl_attention_graph",
                side_effect=lambda *args, **kwargs: order.append("fuse"),
            ),
            mock.patch(
                "esp_ppq.quantization.optim.espdl_attention.legalize_espdl_attention_graph",
                side_effect=lambda *args, **kwargs: order.append("legalize"),
            ) as legalize,
        ):
            passes[1].optimize(graph, executor=executor, inputs=inputs)
            passes[2].optimize(graph, executor=executor, inputs=inputs)
        self.assertEqual(order, ["fuse", "legalize"])
        self.assertIs(legalize.call_args.args[0], graph)
        self.assertEqual(legalize.call_args.kwargs["platform"], TargetPlatform.ESPDL_INT8)
        self.assertTrue(legalize.call_args.kwargs["quantized"])
        self.assertIs(legalize.call_args.kwargs["executor"], executor)
        self.assertIs(legalize.call_args.kwargs["inputs"], inputs)

        def run(target):
            seen = []
            runner = BaseEspdlQuantizer(target)
            with (
                mock.patch(
                    "esp_ppq.quantization.quantizer.EspdlQuantizer.finalize_espdl_attention_for_export",
                    side_effect=lambda *args, **kwargs: seen.append("finalize"),
                ) as finalize,
                mock.patch.object(BaseQuantizer, "quantize", side_effect=lambda *args, **kwargs: seen.append("super")),
            ):
                runner.quantize(inputs, calib_dataloader=None, executor=executor, setting=setting)
            return seen, finalize

        seen, finalize = run(graph)
        self.assertEqual(seen, ["super", "finalize"])
        self.assertEqual(finalize.call_args.args, (graph,))
        self.assertEqual(finalize.call_args.kwargs, {})

        plain = BaseGraph(name="plain")
        plain.create_operation(op_type="Relu", name="relu")
        seen, finalize = run(plain)
        self.assertEqual(seen, ["super"])
        finalize.assert_not_called()

    def test_quantize_traces_the_quantize_inputs_before_prequant(self):
        from unittest import mock

        from esp_ppq.api.setting import QuantizationSettingFactory
        from esp_ppq.core import TargetPlatform
        from esp_ppq.IR import BaseGraph
        from esp_ppq.quantization.optim.espdl_attention import _ensure_traced_shapes
        from esp_ppq.quantization.quantizer.base import BaseQuantizer
        from esp_ppq.quantization.quantizer.EspdlQuantizer import BaseEspdlQuantizer

        graph = BaseGraph(name="unshaped_attention_candidate")
        q_in = graph.create_variable(name="q")
        k_in = graph.create_variable(name="k")
        v_in = graph.create_variable(name="v")
        for var in (q_in, k_in, v_in):
            graph.mark_variable_as_graph_input(var)
        kt_var = graph.create_variable(name="kt")
        scores_var = graph.create_variable(name="scores")
        probs_var = graph.create_variable(name="probs")
        out = graph.create_variable(name="out")
        graph.create_operation(
            op_type="Transpose", name="kT", attributes={"perm": [0, 1, 3, 2]}, inputs=[k_in], outputs=[kt_var]
        )
        graph.create_operation(op_type="MatMul", name="qk", inputs=[q_in, kt_var], outputs=[scores_var])
        graph.create_operation(
            op_type="Softmax", name="sm", attributes={"axis": -1}, inputs=[scores_var], outputs=[probs_var]
        )
        graph.create_operation(op_type="MatMul", name="pv", inputs=[probs_var, v_in], outputs=[out])
        graph.mark_variable_as_graph_output(out)

        order = []
        inputs = object()
        executor = mock.Mock()
        executor.tracing_operation_meta.side_effect = lambda **kwargs: order.append(("trace", kwargs["inputs"]))
        setting = QuantizationSettingFactory.espdl_setting()
        runner = BaseEspdlQuantizer(graph)
        with mock.patch.object(
            BaseQuantizer,
            "quantize",
            side_effect=lambda *args, **kwargs: order.append(("super", kwargs["inputs"])),
        ):
            runner.quantize(inputs, calib_dataloader=[object()], executor=executor, setting=setting)
        self.assertEqual(order, [("trace", inputs), ("super", inputs)])

        classifier = BaseGraph(name="classifier_softmax")
        cls_in = classifier.create_variable(name="in")
        classifier.mark_variable_as_graph_input(cls_in)
        cls_out = classifier.create_variable(name="out")
        classifier.create_operation(op_type="Softmax", name="sm", inputs=[cls_in], outputs=[cls_out])
        classifier.mark_variable_as_graph_output(cls_out)
        cls_executor = mock.Mock()
        with mock.patch.object(BaseQuantizer, "quantize"):
            BaseEspdlQuantizer(classifier).quantize(
                inputs, calib_dataloader=[object()], executor=cls_executor, setting=setting
            )
        cls_executor.tracing_operation_meta.assert_not_called()

        shaped = self._graph_with([[1, 4, 16, 32]] * 3)
        shaped_executor = mock.Mock()
        with (
            mock.patch.object(BaseQuantizer, "quantize"),
            mock.patch("esp_ppq.quantization.quantizer.EspdlQuantizer.finalize_espdl_attention_for_export"),
        ):
            BaseEspdlQuantizer(shaped).quantize(
                inputs, calib_dataloader=None, executor=shaped_executor, setting=setting
            )
        shaped_executor.tracing_operation_meta.assert_not_called()

        helper_executor = mock.Mock()
        _ensure_traced_shapes(graph, executor=helper_executor, dataloader=[object()])
        helper_executor.tracing_operation_meta.assert_not_called()

        with mock.patch.object(
            BaseEspdlQuantizer,
            "target_platform",
            new_callable=mock.PropertyMock,
            return_value=TargetPlatform.PPL_CUDA_INT8,
        ):
            softmax_only = mock.Mock()
            with mock.patch.object(BaseQuantizer, "quantize"):
                BaseEspdlQuantizer(graph).quantize(
                    inputs, calib_dataloader=[object()], executor=softmax_only, setting=setting
                )
            softmax_only.tracing_operation_meta.assert_not_called()

            unshaped_attention = self._graph_with([None, None, None])
            attention_executor = mock.Mock()
            with (
                mock.patch.object(BaseQuantizer, "quantize"),
                mock.patch("esp_ppq.quantization.quantizer.EspdlQuantizer.finalize_espdl_attention_for_export"),
            ):
                BaseEspdlQuantizer(unshaped_attention).quantize(
                    inputs, calib_dataloader=None, executor=attention_executor, setting=setting
                )
            attention_executor.tracing_operation_meta.assert_called_once_with(inputs=inputs)

    def test_prequant_trace_decomposes_oversized_native_attention_without_espdl_kernel(self):
        from unittest import mock

        from esp_ppq.api.setting import QuantizationSettingFactory
        from esp_ppq.core import TargetPlatform
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.quantizer.EspdlQuantizer import BaseEspdlQuantizer

        dim = ESPDL_ATTENTION_MAX_HEAD_DIM + 1
        sample = torch.randn(1, 2, 4, dim)
        graph = self._graph_with([None, None, None])
        graph.operations["attn"].platform = TargetPlatform.ESPDL_INT8
        quantizer = BaseEspdlQuantizer(graph)
        executor = TorchExecutor(graph=graph, device="cpu")

        batches = [
            [sample, sample.clone(), sample.clone()],
            [sample.clone(), sample.clone(), sample.clone()],
        ]
        setting = QuantizationSettingFactory.espdl_setting()
        setting.espdl_attention_fusion = False
        with mock.patch(
            "esp_ppq.executor.op.torch.default.espdl_attention_forward",
            side_effect=AssertionError("oversized native Attention must not call espdl_attention_forward"),
        ):
            quantizer.quantize(
                inputs=batches[0],
                calib_dataloader=batches,
                calib_steps=2,
                executor=executor,
                setting=setting,
            )

        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "MatMul" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "Softmax" for op in graph.operations.values()))
        self.assertEqual(graph.outputs["out"].shape, [1, 2, 4, dim])

    def test_prequant_trace_handles_downstream_shape_reshape_and_gather_after_oversized_attention(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.IR import BaseGraph
        from esp_ppq.parser.espdl_exporter import EspdlExporter

        dim = ESPDL_ATTENTION_MAX_HEAD_DIM + 1
        graph = BaseGraph(name="oversized_attention_tail")
        q, k, v = (graph.create_variable(name=name) for name in ("q", "k", "v"))
        for var in (q, k, v):
            graph.mark_variable_as_graph_input(var)
        attention_out = graph.create_variable(name="attention_out")
        shape = graph.create_variable(name="shape")
        reshaped = graph.create_variable(name="reshaped")
        cls_idx = graph.create_variable(
            name="cls_idx",
            value=torch.tensor(0, dtype=torch.int64),
            is_parameter=True,
        )
        out = graph.create_variable(name="out")
        graph.create_operation(
            op_type="Attention",
            name="attn",
            inputs=[q, k, v],
            outputs=[attention_out],
            platform=TargetPlatform.ESPDL_INT8,
        )
        graph.create_operation(
            op_type="Shape",
            name="shape",
            inputs=[attention_out],
            outputs=[shape],
            platform=TargetPlatform.ESPDL_INT8,
        )
        graph.create_operation(
            op_type="Reshape",
            name="reshape",
            inputs=[attention_out, shape],
            outputs=[reshaped],
            platform=TargetPlatform.ESPDL_INT8,
        )
        graph.create_operation(
            op_type="Gather",
            name="gather_cls",
            inputs=[reshaped, cls_idx],
            outputs=[out],
            attributes={"axis": 2},
            platform=TargetPlatform.ESPDL_INT8,
        )
        graph.mark_variable_as_graph_output(out)

        sample = torch.randn(1, 2, 4, dim)
        EspdlExporter.prepare_graph_for_float_export(
            graph,
            platform=TargetPlatform.ESPDL_INT8,
            inputs=[sample, sample.clone(), sample.clone()],
            device="cpu",
        )
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertEqual(attention_out.shape, [1, 2, 4, dim])
        self.assertEqual(shape.shape, [4])
        self.assertEqual(reshaped.shape, [1, 2, 4, dim])
        self.assertEqual(out.shape, [1, 2, dim])

    def test_float_prepare_skips_legalize_without_attention(self):
        from unittest import mock

        from esp_ppq.core import TargetPlatform
        from esp_ppq.IR import BaseGraph
        from esp_ppq.parser.espdl_exporter import EspdlExporter
        from esp_ppq.quantization.optim import espdl_attention

        graph = BaseGraph(name="mobilenet_like")
        graph.create_operation(op_type="Relu", name="relu")
        with (
            mock.patch.object(espdl_attention, "legalize_espdl_attention_graph") as legalize,
            mock.patch("esp_ppq.parser.espdl_exporter.TorchExecutor") as executor_cls,
        ):
            executor_cls.return_value.tracing_operation_meta.return_value = None
            EspdlExporter.prepare_graph_for_float_export(
                graph,
                platform=TargetPlatform.ESPDL_INT8,
                inputs=torch.zeros(1, 3, 224, 224),
                device="cpu",
            )
            legalize.assert_not_called()

    @staticmethod
    def _capture_decompose_logs(check_fn, graph):
        # NaiveLogger prints directly to the module logger's methods; patching
        # ``info`` / ``warning`` on the module logger is the least-fragile
        # way to observe the routing (assertLogs sees nothing because the
        # logger never reaches Python's logging framework).
        from unittest import mock

        from esp_ppq.quantization.optim import espdl_attention

        captured = {"info": [], "warning": []}
        with (
            mock.patch.object(espdl_attention.logger, "info", side_effect=captured["info"].append),
            mock.patch.object(espdl_attention.logger, "warning", side_effect=captured["warning"].append),
        ):
            check_fn(graph)
        return captured

    def test_oversized_head_dim_decompose_warns_with_both_knobs(self):
        # Silent decompose is a UX gotcha: if the device raised
        # ESP_DL_ATTENTION_MAX_HEAD_DIM to 128 but the export
        # environment stayed on the default 64, fusion silently drops out.
        # WARNING + both knob names in the reason lets the user notice and fix.
        graph = self._graph_with([[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3)
        logs = self._capture_decompose_logs(self.check, graph)
        self.assertTrue(logs["warning"], "expected a WARNING-level decompose notice")
        joined = "\n".join(logs["warning"])
        self.assertIn("expanded to MatMul/Softmax", joined)
        self.assertIn("ESP_DL_ATTENTION_MAX_HEAD_DIM", joined)
        self.assertIn("ESPDL_ATTENTION_MAX_HEAD_DIM", joined)

    def test_oversized_lk_decompose_warns_with_both_knobs(self):
        graph = self._graph_with([[1, 4, ESPDL_ATTENTION_MAX_KV_TOKENS + 1, 32]] * 3)
        logs = self._capture_decompose_logs(self.check, graph)
        self.assertTrue(logs["warning"], "expected a WARNING-level decompose notice")
        joined = "\n".join(logs["warning"])
        self.assertIn("ESP_DL_ATTENTION_MAX_KV_TOKENS", joined)
        self.assertIn("ESPDL_ATTENTION_MAX_KV_TOKENS", joined)


def _activate_qk_scales(graph, scale=2**-8):
    """Attach activated symmetric per-tensor Q/K scales, as calibration would."""
    from esp_ppq.core import (
        OperationQuantizationConfig,
        QuantizationPolicy,
        QuantizationProperty,
        QuantizationStates,
        RoundingPolicy,
        TargetPlatform,
        TensorQuantizationConfig,
    )
    from esp_ppq.IR import QuantableOperation

    op = graph.operations["attn"]
    policy = QuantizationPolicy(
        QuantizationProperty.SYMMETRICAL
        + QuantizationProperty.LINEAR
        + QuantizationProperty.PER_TENSOR
        + QuantizationProperty.POWER_OF_2
    )

    def config():
        return TensorQuantizationConfig(
            policy=policy,
            rounding=RoundingPolicy.ROUND_HALF_EVEN,
            num_of_bits=8,
            quant_min=-128,
            quant_max=127,
            scale=scale,
            offset=0,
            state=QuantizationStates.ACTIVATED,
        )

    quantized = QuantableOperation(
        convert_from=op,
        quantize_config=OperationQuantizationConfig(
            [config() for _ in op.inputs],
            [config() for _ in op.outputs],
        ),
        platform=TargetPlatform.ESPDL_INT8,
    )
    graph.operations[op.name] = quantized
    return graph


class TestAttentionFinalizeGate(unittest.TestCase):
    """Post-calibration ``logit_scale`` check. Shape and platform stay in legalize."""

    _graph_with = staticmethod(TestDeployabilityGuard._graph_with)

    def setUp(self):
        from esp_ppq.quantization.optim import finalize_espdl_attention_for_export

        self.gate = finalize_espdl_attention_for_export

    def test_accepts_eligible_attention(self):
        self.gate(_activate_qk_scales(self._graph_with([[1, 4, 16, 32]] * 3)))

    def test_missing_qk_scale_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.gate(self._graph_with([[1, 4, 16, 32]] * 3))
        self.assertIn("logit_scale was not checked", str(ctx.exception))

    def test_overflowing_qk_scale_is_rejected(self):
        graph = _activate_qk_scales(self._graph_with([[1, 4, 16, 32]] * 3), scale=float(2**80))
        with self.assertRaises(ValueError) as ctx:
            self.gate(graph)
        message = str(ctx.exception)
        self.assertIn("logit_scale", message)
        self.assertNotIn("was not checked", message)

    def test_unusable_qk_scale_is_rejected(self):
        from esp_ppq.core import QuantizationPolicy, QuantizationProperty, QuantizationStates

        per_channel = QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL
            + QuantizationProperty.LINEAR
            + QuantizationProperty.PER_CHANNEL
            + QuantizationProperty.POWER_OF_2
        )
        cases = (
            ("per-channel", lambda op: setattr(op.input_quant_config[0], "policy", per_channel)),
            ("initial", lambda op: setattr(op.input_quant_config[0], "state", QuantizationStates.INITIAL)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                graph = _activate_qk_scales(self._graph_with([[1, 4, 16, 32]] * 3))
                mutate(graph.operations["attn"])
                with self.assertRaises(ValueError) as ctx:
                    self.gate(graph)
                self.assertIn("logit_scale was not checked", str(ctx.exception))

    def test_ignores_other_op_types(self):
        from esp_ppq.IR import BaseGraph

        graph = BaseGraph(name="g")
        graph.create_operation(op_type="MatMul", name="mm")
        self.gate(graph)

    def test_does_not_repeat_legalize_checks(self):
        graph = _activate_qk_scales(self._graph_with([[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3))
        self.gate(graph)
        self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertFalse(any(op.type == "MatMul" for op in graph.operations.values()))

    def test_accepts_explicit_fp32_attention_and_marks_f32_quant_type(self):
        from esp_ppq.core import QuantizationStates, TargetPlatform
        from esp_ppq.parser.espdl.espdl_typedef import EspQuantType
        from esp_ppq.parser.espdl.export_patterns import InsertQuantTypePattern

        fp32_graph = self._graph_with([[1, 4, 16, 32]] * 3)
        fp32_graph.operations["attn"].platform = TargetPlatform.FP32
        self.gate(fp32_graph)

        bad_fp32_graph = self._graph_with([[1, 4, 16, 32]] * 3, {"scale": 0.0})
        bad_fp32_graph.operations["attn"].platform = TargetPlatform.FP32
        with self.assertRaises(ValueError) as ctx:
            self.gate(bad_fp32_graph)
        self.assertIn("finite, positive, normal float32", str(ctx.exception))

        dequant_graph = _activate_qk_scales(self._graph_with([[1, 4, 16, 32]] * 3))
        attn_op = dequant_graph.operations["attn"]
        for cfg in attn_op.input_quant_config + attn_op.output_quant_config:
            cfg.state = QuantizationStates.FP32
        self.gate(dequant_graph)
        InsertQuantTypePattern().export(attn_op, dequant_graph)
        self.assertEqual(attn_op.attributes["quant_type"], EspQuantType.F32)

        import os
        import tempfile

        from esp_ppq.api import quantize_onnx_model
        from esp_ppq.api.setting import QuantizationSettingFactory
        from esp_ppq.parser.espdl_exporter import EspdlExporter

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = os.path.join(tmp, "mixed_fp32_attention.onnx")
            espdl_path = os.path.join(tmp, "mixed_fp32_attention.espdl")
            model = _native_attention_block_model(head_dim=8)
            for node in model.graph.node:
                if node.op_type == "Attention":
                    node.name = "attn"
            onnx.save(model, onnx_path)
            setting = QuantizationSettingFactory.espdl_setting()
            setting.dispatching_table.append("attn", TargetPlatform.FP32)
            sample = [torch.randn(1, 2, 4, 8)]
            with _silence_export_logs():
                quantized_graph = quantize_onnx_model(
                    onnx_import_file=onnx_path,
                    calib_dataloader=[sample, sample],
                    calib_steps=2,
                    input_shape=None,
                    platform=TargetPlatform.ESPDL_INT8,
                    inputs=sample,
                    setting=setting,
                    device="cpu",
                    verbose=0,
                )
            self.assertEqual(quantized_graph.operations["attn"].platform, TargetPlatform.FP32)
            EspdlExporter().export(espdl_path, quantized_graph, export_config=False)
            self.assertTrue(os.path.isfile(espdl_path))
            self.assertGreater(os.path.getsize(espdl_path), 0)

    def test_rejects_mixed_storage_quantization(self):
        from esp_ppq.core import QuantizationStates

        cases = (
            ("mixed_v_bits", lambda op: setattr(op.input_quant_config[2], "num_of_bits", 16), "bit-widths must match"),
            ("mixed_o_bits", lambda op: setattr(op.output_quant_config[0], "num_of_bits", 16), "bit-widths must match"),
            (
                "fp32_v_only",
                lambda op: setattr(op.input_quant_config[2], "state", QuantizationStates.FP32),
                "logit_scale was not checked",
            ),
        )
        for name, mutate, expected in cases:
            with self.subTest(name=name):
                graph = _activate_qk_scales(self._graph_with([[1, 4, 16, 32]] * 3))
                mutate(graph.operations["attn"])
                with self.assertRaises(ValueError) as ctx:
                    self.gate(graph)
                self.assertIn(expected, str(ctx.exception))


class TestAttentionPlatformGate(unittest.TestCase):
    """Every esp-dl chip keeps an eligible fused Attention. Other platforms expand it."""

    _graph_with = staticmethod(TestDeployabilityGuard._graph_with)
    _ESPDL = (
        "ESPDL_INT8",
        "ESPDL_INT16",
        "ESPDL_H_PRE_INT16",
        "ESPDL_W8A16",
        "ESPDL_S3_INT8",
        "ESPDL_S3_INT16",
        "ESPDL_S3_H_PRE_INT16",
        "ESPDL_S3_W8A16",
        "ESPDL_C_INT8",
        "ESPDL_C_INT16",
        "ESPDL_C_H_PRE_INT16",
        "ESPDL_C_W8A16",
    )

    @staticmethod
    def _types(graph):
        return {op.type for op in graph.operations.values()}

    def test_platform_table(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim import espdl_attention_platform_unsupported_reason

        self.assertIsNone(espdl_attention_platform_unsupported_reason(None))
        for name in self._ESPDL:
            self.assertIsNone(espdl_attention_platform_unsupported_reason(TargetPlatform[name]), name)
        reason = espdl_attention_platform_unsupported_reason(TargetPlatform.FP32)
        self.assertIn("FP32", reason)
        self.assertIn("no esp-dl Attention kernel", reason)
        self.assertIn("ESP-DL target platform", reason)
        self.assertNotIn("ESP32-P4 only", reason)

    def test_target_strings_map_as_expected(self):
        from esp_ppq.api.espdl_interface import get_target_platform
        from esp_ppq.quantization.optim import espdl_attention_platform_unsupported_reason

        for quant_type in ("w8a8", "w8a16", "w16a16"):
            for target in ("esp32p4", "esp32s3", "esp32c3", "c"):
                platform = get_target_platform(target, quant_type)
                self.assertIsNone(espdl_attention_platform_unsupported_reason(platform), (target, quant_type))

    def test_w8a16_exports_attention_as_s16(self):
        # Attention has no weights, so under W8A16 every operand is a 16-bit activation.
        from esp_ppq.core import TargetPlatform
        from esp_ppq.parser.espdl.espdl_typedef import EspQuantType
        from esp_ppq.parser.espdl.export_patterns import InsertQuantTypePattern
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        graph = self._graph_with([[1, 4, 16, 32]] * 3)
        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_W8A16, quantized=True)
        (op,) = [o for o in graph.operations.values() if o.type == "Attention"]
        op.platform = TargetPlatform.ESPDL_W8A16
        InsertQuantTypePattern().export(op, graph)
        self.assertEqual(op.attributes["quant_type"], EspQuantType.S16)

    def test_eligible_node_is_kept_on_s3_and_c(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        for platform in (TargetPlatform.ESPDL_S3_INT8, TargetPlatform.ESPDL_C_INT16):
            graph = self._graph_with([[1, 4, 16, 32]] * 3)
            legalize_espdl_attention_graph(graph, platform=platform)
            self.assertIn("Attention", self._types(graph), platform.name)

    def test_eligible_node_is_kept_on_p4(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        graph = self._graph_with([[1, 4, 16, 32]] * 3)
        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT16)
        self.assertIn("Attention", self._types(graph))

    def test_expansion_without_the_module_matches_attention_forward(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        torch.manual_seed(5)
        q = torch.randn(1, 4, 16, 32)
        k = torch.randn(1, 2, 20, 32)
        v = torch.randn(1, 2, 20, 32)
        mask = torch.zeros(1, 1, 16, 20)
        mask[..., -3:] = -1e4
        graph, feed = TestAttentionDecompose._graph(q, k, v, {"scale": 0.2}, mask=mask)
        ordered = [feed[n] for n in graph.inputs.keys()]
        golden = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)[0]
        legalize_espdl_attention_graph(graph, platform=TargetPlatform.FP32)
        self.assertNotIn("Attention", self._types(graph))
        got = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)[0]
        torch.testing.assert_close(got, golden, rtol=1e-4, atol=1e-4)

    def test_unexpandable_node_raises_with_platform_reason(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        graph = self._graph_with([[1, 4, 16, 32]] * 3, num_inputs=5)
        with self.assertRaises(ValueError) as ctx:
            legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_C_INT8)
        message = str(ctx.exception)
        self.assertIn("KV cache", message)
        self.assertNotIn("ESPDL_C_INT8", message)

    def test_expanded_compute_stays_quantizable(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        def with_const_mask(value):
            graph = self._graph_with([[1, 4, 16, 32]] * 3)
            op = graph.operations["attn"]
            graph.create_link_with_op(None, op, graph.create_variable(value=value, is_parameter=True))
            op.inputs[3].shape = list(value.shape)
            return graph

        def assert_compute_quantizable(graph):
            self.assertNotIn("Attention", self._types(graph))
            compute = [op for op in graph.operations.values() if op.type in {"MatMul", "Softmax", "Mul"}]
            self.assertTrue(compute)
            for op in compute:
                self.assertNotEqual(op.platform, TargetPlatform.FP32, op.name)
            for op in graph.operations.values():
                if op.type == "Add":
                    self.assertEqual(op.platform, TargetPlatform.FP32, op.name)

        bool_mask = torch.ones(1, 1, 16, 16, dtype=torch.bool).tril()
        inf_mask = torch.zeros(1, 1, 16, 16).masked_fill(~bool_mask, float("-inf"))
        cases = (
            lambda: self._graph_with([[1, 4, 16, 32]] * 3, {"is_causal": 1}),
            lambda: with_const_mask(bool_mask),
            lambda: with_const_mask(inf_mask),
        )
        for make in cases:
            with self.subTest(make=make):
                graph = make()
                legalize_espdl_attention_graph(graph, platform=TargetPlatform.FP32, quantized=True)
                assert_compute_quantizable(graph)

                graph = make()
                legalize_espdl_attention_graph(graph, platform=TargetPlatform.FP32)
                self.assertNotIn("Attention", self._types(graph))

        finite = with_const_mask(torch.zeros(1, 1, 16, 16).masked_fill(~bool_mask, -1e4))
        legalize_espdl_attention_graph(finite, platform=TargetPlatform.FP32, quantized=True)
        self.assertNotIn("Attention", self._types(finite))
        for op in finite.operations.values():
            if op.type in {"MatMul", "Softmax", "Add", "Mul"}:
                self.assertNotEqual(op.platform, TargetPlatform.FP32, op.name)

        for platform in (TargetPlatform.ESPDL_S3_INT8, TargetPlatform.ESPDL_C_INT8, TargetPlatform.ESPDL_INT8):
            graph = self._graph_with([[1, 4, 16, 32]] * 3, {"is_causal": 1})
            legalize_espdl_attention_graph(graph, platform=platform, quantized=True)
            self.assertIn("Attention", self._types(graph), platform.name)
        graph = self._graph_with([[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3, {"is_causal": 1})
        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_S3_INT8, quantized=True)
        assert_compute_quantizable(graph)

    def test_where_folded_inf_mask_stays_fp32_when_expanded(self):
        from esp_ppq.core import DataType, TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        graph = self._graph_with([[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3)
        op = graph.operations["attn"]
        op.platform = TargetPlatform.ESPDL_INT8
        keep = torch.tril(torch.ones(16, 16, dtype=torch.bool))
        mask = torch.where(
            keep,
            torch.zeros(16, 16, dtype=torch.float32),
            torch.full((16, 16), float("-inf")),
        )
        mask_out = graph.create_variable(name="folded_mask", value=mask, is_parameter=False)
        mask_out.shape = list(mask.shape)
        mask_out.dtype = DataType.FP32
        const = graph.create_operation(
            op_type="Constant",
            name="folded_mask_const",
            attributes={"value": mask},
            platform=TargetPlatform.FP32,
            inputs=[],
            outputs=[mask_out],
        )
        graph.create_link_with_op(const, op, mask_out)

        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8, quantized=True)
        self.assertNotIn("Attention", self._types(graph))
        adds = [item for item in graph.operations.values() if item.type == "Add"]
        self.assertTrue(adds)
        for item in adds:
            self.assertEqual(item.platform, TargetPlatform.FP32, item.name)
        for item in graph.operations.values():
            if item.type in {"MatMul", "Softmax"}:
                self.assertEqual(item.platform, TargetPlatform.ESPDL_INT8, item.name)

    def test_legalize_keeps_quantized_causal(self):
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        graph = self._graph_with([[1, 4, 16, 32]] * 3, {"is_causal": 1})
        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_S3_INT16, quantized=True)
        self.assertIn("Attention", self._types(graph))

    def test_constant_bool_mask_expansion_cleans_orphan_and_keeps_fp32_add(self):
        from esp_ppq.core import DataType, TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        for via_constant_op in (False, True):
            with self.subTest(via_constant_op=via_constant_op):
                graph = self._graph_with([[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3)
                op = graph.operations["attn"]
                op.platform = TargetPlatform.ESPDL_INT8
                # All-True bool mask converts to all-zero float32 additive mask, so
                # _nonfinite_mask must inspect the bool mask before _additive_mask detaches Constant.
                keep = torch.ones(16, 16, dtype=torch.bool)
                if via_constant_op:
                    mask_var = graph.create_variable(name="bool_mask", value=None, is_parameter=False)
                    mask_var.shape = list(keep.shape)
                    mask_var.dtype = DataType.BOOL
                    const = graph.create_operation(
                        op_type="Constant",
                        name="bool_mask_const",
                        attributes={"value": keep},
                        platform=TargetPlatform.FP32,
                        inputs=[],
                        outputs=[mask_var],
                    )
                    graph.create_link_with_op(const, op, mask_var)
                else:
                    mask_var = graph.create_variable(name="bool_mask", value=keep, is_parameter=True)
                    mask_var.shape = list(keep.shape)
                    mask_var.dtype = DataType.BOOL
                    op.inputs.append(mask_var)
                    mask_var.dest_ops.append(op)

                legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8, quantized=True)
                self.assertNotIn("Attention", self._types(graph))
                self.assertNotIn("bool_mask", graph.variables)
                adds = [item for item in graph.operations.values() if item.type == "Add"]
                self.assertEqual(len(adds), 1)
                self.assertEqual(adds[0].platform, TargetPlatform.FP32)


class TestAttentionApiConstraints(unittest.TestCase):
    """Architecture lint. These tests read source text on purpose.

    A rename or a moved call fails them. That is the signal: Attention must
    stay off the shared quantizer API. Call order is covered separately by
    ``TestDeployabilityGuard.test_quantize_runs_legalize_then_finalize``.
    ``inspect.getsource`` on a function wrapped with ``empty_ppq_cache``
    returns the wrapper, so those checks read the module source instead.
    """

    def test_attention_passes_live_on_espdl_quantizer_prequant_pipeline(self):
        import inspect

        from esp_ppq.quantization.quantizer.base import BaseQuantizer
        from esp_ppq.quantization.quantizer.EspdlQuantizer import BaseEspdlQuantizer

        # Attention passes are esp-dl-specific and live on BaseEspdlQuantizer,
        # not on BaseQuantizer (which is shared with ORT / TRT / OpenVINO / ...).
        base_source = inspect.getsource(BaseQuantizer.build_prequant_pipeline)
        self.assertNotIn("legalize_espdl_attention_graph", base_source)
        self.assertNotIn("EspdlAttentionFusionPass", base_source)
        base_quantize = inspect.getsource(BaseQuantizer.quantize)
        self.assertNotIn("legalize_espdl_attention_graph", base_quantize)

        espdl_pre = inspect.getsource(BaseEspdlQuantizer.build_prequant_pipeline)
        self.assertIn("EspdlAttentionLegalizePass", espdl_pre)
        self.assertIn("EspdlAttentionFusionPass", espdl_pre)
        self.assertLess(espdl_pre.index("EspdlAttentionLegalizePass"), espdl_pre.index("EspdlAttentionFusionPass"))
        quant_source = inspect.getsource(BaseEspdlQuantizer.build_quant_pipeline)
        self.assertNotIn("finalize_espdl_attention_for_export", quant_source)
        self.assertNotIn("EspdlAttentionFusionPass", quant_source)

    def test_fp32_path_checks_too(self):
        import inspect

        from esp_ppq.api import espdl_interface

        # getsource(espdl_quantize_onnx) returns the empty_ppq_cache wrapper
        source = inspect.getsource(espdl_interface)
        before_quantize = source.split("ppq_graph = quantize_onnx_model(")[0]
        self.assertNotIn("EspdlAttentionFusionPass", before_quantize)
        self.assertNotIn("legalize_espdl_attention_graph", source)
        self.assertNotIn("espdl_attention_test_value", source)
        self.assertNotIn("espdl_embedded_test_value_simulation", source)
        gen_src = inspect.getsource(espdl_interface.generate_test_value)
        self.assertIn("executor.forward", gen_src)
        from esp_ppq.api.interface import format_graph

        format_source = inspect.getsource(format_graph)
        self.assertNotIn("GraphCommandType.FUSE_ATTENTION", format_source)
        self.assertNotIn("fuse_espdl_attention_graph", format_source)
        from esp_ppq.parser.espdl_exporter import EspdlExporter

        fp32_branch = source.split("# support TargetPlatform.FP32")[1].split("if not skip_export")[0]
        self.assertIn("prepare_graph_for_float_export", fp32_branch)
        self.assertIn("load_onnx_graph", fp32_branch)
        self.assertNotIn("quantize_onnx_model", fp32_branch)
        float_prep = inspect.getsource(EspdlExporter.prepare_graph_for_float_export)
        self.assertIn("EspdlAttentionFusionPass", float_prep)
        self.assertIn("tracing_operation_meta", float_prep)
        fusion_call = float_prep.index("EspdlAttentionFusionPass()")
        self.assertLess(float_prep.index("tracing_operation_meta"), fusion_call)
        self.assertLess(fusion_call, float_prep.index("legalize_espdl_attention_graph("))
        self.assertIn('op.type == "Attention"', float_prep)
        self.assertIn("legalize_espdl_attention_graph(", float_prep)
        self.assertIn("quantized=False", float_prep)
        self.assertNotIn("finalize_espdl_attention_for_export", float_prep)
        self.assertNotIn("fuse_primitives", float_prep)

    def test_espdl_quantize_legalizes_before_pipeline(self):
        import inspect

        from esp_ppq.quantization.quantizer.EspdlQuantizer import (
            BaseEspdlQuantizer,
            EspdlCQuantizer,
            EspdlQuantizer,
            EspdlS3Quantizer,
        )

        quantize_source = inspect.getsource(BaseEspdlQuantizer.quantize)
        self.assertNotIn("EspdlAttentionFusionPass", quantize_source)
        self.assertNotIn("legalize_espdl_attention_graph", quantize_source)
        self.assertIn("tracing_operation_meta", quantize_source)
        self.assertLess(
            quantize_source.index("tracing_operation_meta"),
            quantize_source.index("super().quantize"),
        )
        self.assertIn("finalize_espdl_attention_for_export", quantize_source)
        prequant_source = inspect.getsource(BaseEspdlQuantizer.build_prequant_pipeline)
        self.assertIn("EspdlAttentionFusionPass", prequant_source)
        self.assertIn("EspdlAttentionLegalizePass", prequant_source)
        for cls in (EspdlQuantizer, EspdlS3Quantizer, EspdlCQuantizer):
            self.assertIs(cls.quantize, BaseEspdlQuantizer.quantize, cls.__name__)
            self.assertIs(cls.build_prequant_pipeline, BaseEspdlQuantizer.build_prequant_pipeline, cls.__name__)


class TestAttentionDecompose(unittest.TestCase):
    """Expanded MatMul/Softmax must match the pre-decompose Attention forward."""

    @staticmethod
    def _graph(q, k, v, attrs=None, mask=None):
        from esp_ppq.IR import BaseGraph

        graph = BaseGraph(name="g")
        feed = {"q": q, "k": k, "v": v}
        qv = graph.create_variable(name="q")
        kv = graph.create_variable(name="k")
        vv = graph.create_variable(name="v")
        qv.shape = list(q.shape)
        kv.shape = list(k.shape)
        vv.shape = list(v.shape)
        graph.mark_variable_as_graph_input(qv)
        graph.mark_variable_as_graph_input(kv)
        graph.mark_variable_as_graph_input(vv)
        ins = [qv, kv, vv]
        if mask is not None:
            mv = graph.create_variable(name="mask")
            mv.shape = list(mask.shape)
            graph.mark_variable_as_graph_input(mv)
            ins.append(mv)
            feed["mask"] = mask
        out = graph.create_variable(name="out")
        graph.create_operation(
            op_type="Attention",
            name="attn",
            attributes=dict(attrs or {}),
            inputs=ins,
            outputs=[out],
        )
        graph.mark_variable_as_graph_output(out)
        return graph, feed

    def _parity(self, q, k, v, attrs=None, mask=None):
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        graph, feed = self._graph(q, k, v, attrs, mask)
        ordered = [feed[n] for n in graph.inputs.keys()]
        golden = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)[0]
        legalize_espdl_attention_graph(graph)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        got = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)[0]
        torch.testing.assert_close(got, golden, rtol=1e-4, atol=1e-4)

    def test_oversized_mha_matches_attention_forward(self):
        torch.manual_seed(0)
        d = ESPDL_ATTENTION_MAX_HEAD_DIM + 1
        q = torch.randn(1, 2, 8, d)
        self._parity(q, q.clone(), q.clone())

    def test_causal_and_additive_mask(self):
        torch.manual_seed(1)
        d = ESPDL_ATTENTION_MAX_HEAD_DIM * 2
        q = torch.randn(2, 4, 6, d)
        mask = torch.zeros(1, 1, 6, 6)
        mask[..., -1] = -1e4
        self._parity(q, q.clone(), q.clone(), {"is_causal": 1, "scale": 0.1}, mask=mask)

    def test_gqa_repeat_matches(self):
        torch.manual_seed(2)
        d = ESPDL_ATTENTION_MAX_HEAD_DIM + 8
        q = torch.randn(1, 8, 4, d)
        k = torch.randn(1, 2, 4, d)
        v = torch.randn(1, 2, 4, d)
        self._parity(q, k, v)

    def test_bnhd_matches_permuted_sdpa(self):
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        torch.manual_seed(3)
        b, n, h, d = 1, 6, 3, ESPDL_ATTENTION_MAX_HEAD_DIM + 1
        q = torch.randn(b, n, h, d)
        k = q.clone()
        v = q.clone()
        graph, feed = self._graph(q, k, v, {"layout": "BNHD"})
        legalize_espdl_attention_graph(graph)
        ordered = [feed[name] for name in graph.inputs.keys()]
        got = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)[0]
        q4 = q.permute(0, 2, 1, 3)
        k4 = k.permute(0, 2, 1, 3)
        v4 = v.permute(0, 2, 1, 3)
        scale = d**-0.5
        golden = F.scaled_dot_product_attention(q4, k4, v4, scale=scale).permute(0, 2, 1, 3)
        torch.testing.assert_close(got, golden, rtol=1e-4, atol=1e-4)

    def test_fused_bnhd_matches_permuted_sdpa(self):
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        torch.manual_seed(4)
        b, n, h, d = 1, 8, 4, 32
        q = torch.randn(b, n, h, d)
        graph, feed = self._graph(q, q.clone(), q.clone(), {"layout": "BNHD"})
        legalize_espdl_attention_graph(graph)
        self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))
        ordered = [feed[name] for name in graph.inputs.keys()]
        got = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)[0]
        q4 = q.permute(0, 2, 1, 3)
        golden = F.scaled_dot_product_attention(q4, q4, q4, scale=d**-0.5).permute(0, 2, 1, 3)
        torch.testing.assert_close(got, golden, rtol=1e-4, atol=1e-4)


class TestBnhdPackTraffic(unittest.TestCase):
    """Host cache-line model used while choosing BNHD gather.

    The device no longer does a full-token SRAM pack. int8 QACC BNHD fills
    ``h_chunk`` adjacent heads (``hc*D`` bytes, 64 B when D=32) into the
    existing K/V/Q scratch. This test still records the older full-token
    vs per-head-stride transaction counts for regression of that model.
    """

    def test_sequential_pack_touches_fewer_psram_transactions(self):
        h, d, lk = 8, 32, 1601
        heads = 8
        token_bytes = h * d
        row_bytes = d
        stride_txns = heads * lk
        stride_bytes = stride_txns * row_bytes
        pack_bytes = heads * lk * token_bytes
        self.assertGreater(pack_bytes, stride_bytes)
        win = 16 * 1024
        chunk = win // token_bytes
        self.assertGreaterEqual(chunk, 64)
        bursts_per_head = (lk + chunk - 1) // chunk
        pack_bursts = heads * bursts_per_head
        self.assertLess(pack_bursts, stride_txns)
        self.assertLess(pack_bursts * 8, stride_txns)

    def test_pack_scatter_matches_numpy_permute(self):
        import numpy as np

        rng = np.random.default_rng(0)
        b, n, h, d = 1, 20, 8, 32
        src = rng.integers(-128, 127, size=(b, n, h, d), dtype=np.int8)
        packed = np.empty((b, h, n, d), dtype=np.int8)
        for t in range(n):
            token = src[0, t]
            for hh in range(h):
                packed[0, hh, t] = token[hh]
        np.testing.assert_array_equal(packed, src.transpose(0, 2, 1, 3))


class TestOnnx23FallbackChecks(unittest.TestCase):
    """The generic ai.onnx::Attention path must reject shapes it cannot run."""

    def _op_with_attrs(self, attrs):
        op = _attention_op(attrs)
        return op

    def test_4d_head_attr_mismatch_is_rejected(self):
        from esp_ppq.executor.op.torch.default import _onnx23_attention_forward

        q = torch.randn(1, 4, 16, 32)
        op = self._op_with_attrs({"q_num_heads": 8, "kv_num_heads": 4})
        with self.assertRaises(ValueError) as ctx:
            _onnx23_attention_forward(op=op, values=[q, q.clone(), q.clone()])
        self.assertIn("q_num_heads=8", str(ctx.exception))

    def test_v_head_size_mismatch_is_rejected(self):
        from esp_ppq.executor.op.torch.default import _onnx23_attention_forward

        q = torch.randn(1, 4, 16, 32)
        k = torch.randn(1, 4, 16, 32)
        v = torch.randn(1, 4, 16, 16)
        op = self._op_with_attrs({"q_num_heads": 4, "kv_num_heads": 4})
        with self.assertRaises(NotImplementedError) as ctx:
            _onnx23_attention_forward(op=op, values=[q, k, v])
        self.assertIn("v_head_size", str(ctx.exception))

    def test_k_head_size_mismatch_is_rejected(self):
        from esp_ppq.executor.op.torch.default import _onnx23_attention_forward

        q = torch.randn(1, 4, 16, 32)
        k = torch.randn(1, 4, 16, 16)  # wrong D
        v = torch.randn(1, 4, 16, 32)
        op = self._op_with_attrs({"q_num_heads": 4, "kv_num_heads": 4})
        with self.assertRaises(ValueError) as ctx:
            _onnx23_attention_forward(op=op, values=[q, k, v])
        self.assertIn("head_dim", str(ctx.exception))

    def test_unknown_layout_is_rejected(self):
        from esp_ppq.executor.op.torch.default import _onnx23_attention_forward

        q = torch.randn(1, 4, 16, 32)
        op = self._op_with_attrs({"layout": "NHBD"})
        with self.assertRaises(ValueError) as ctx:
            _onnx23_attention_forward(op=op, values=[q, q.clone(), q.clone()])
        self.assertIn("layout", str(ctx.exception))


def _sdpa_reference(q, k, v, mask, is_causal, scale):
    """Spec-literal SDPA in fp64: additive mask first, causal on top."""
    scores = (q.double() @ k.double().transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask.double()
    if is_causal:
        keep = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v.double()


class TestMicrosoftHiddenSizes(unittest.TestCase):
    """``qkv_hidden_sizes`` is an INTS attribute, not a scalar."""

    HIDDEN = 96
    HEADS = 4
    BATCH = 2
    SEQ = 8

    def setUp(self):
        torch.manual_seed(0)
        self.inp = torch.randn(self.BATCH, self.SEQ, self.HIDDEN, dtype=torch.float64)
        self.weight = torch.randn(self.HIDDEN, 3 * self.HIDDEN, dtype=torch.float64)
        self.bias = torch.randn(3 * self.HIDDEN, dtype=torch.float64)

    def _reference(self):
        hidden, heads = self.HIDDEN, self.HEADS
        head = hidden // heads
        qkv = self.inp @ self.weight + self.bias
        parts = [qkv[:, :, hidden * n : hidden * (n + 1)] for n in range(3)]
        q, k, v = (t.reshape(self.BATCH, -1, heads, head).permute(0, 2, 1, 3) for t in parts)
        scores = torch.softmax(q @ k.transpose(-2, -1) * head**-0.5, dim=-1)
        return (scores @ v).transpose(1, 2).reshape(self.BATCH, -1, hidden)

    def test_ints_attribute_is_reduced_to_a_scalar(self):
        from esp_ppq.executor.op.torch.default import Attention_forward

        op = _attention_op({"num_heads": self.HEADS, "qkv_hidden_sizes": [self.HIDDEN] * 3})
        out = Attention_forward(op=op, values=[self.inp, self.weight, self.bias])
        self.assertTrue(torch.allclose(out, self._reference()))

    def test_scalar_attribute_still_accepted(self):
        from esp_ppq.executor.op.torch.default import _microsoft_attention_forward

        op = _attention_op({"num_heads": self.HEADS, "qkv_hidden_sizes": self.HIDDEN})
        out = _microsoft_attention_forward(op=op, values=[self.inp, self.weight, self.bias])
        self.assertTrue(torch.allclose(out, self._reference()))

    def test_absent_attribute_derives_from_packed_weight(self):
        from esp_ppq.executor.op.torch.default import _microsoft_attention_forward

        op = _attention_op({"num_heads": self.HEADS})
        out = _microsoft_attention_forward(op=op, values=[self.inp, self.weight, self.bias])
        self.assertTrue(torch.allclose(out, self._reference()))

    def test_asymmetric_sizes_are_refused(self):
        from esp_ppq.executor.op.torch.default import _microsoft_attention_forward

        op = _attention_op({"num_heads": self.HEADS, "qkv_hidden_sizes": [self.HIDDEN, self.HIDDEN, 48]})
        with self.assertRaises(NotImplementedError) as ctx:
            _microsoft_attention_forward(op=op, values=[self.inp, self.weight, self.bias])
        self.assertIn("asymmetric", str(ctx.exception))


class TestCausalWithMask(unittest.TestCase):
    """ONNX composes ``is_causal`` and ``attn_mask``; torch < 2.3 cannot take both."""

    SHAPES = ((8, 8), (4, 10), (10, 4))

    def setUp(self):
        torch.manual_seed(0)

    def _qkv(self, l_q, l_k, heads=2, dim=32):
        q = torch.randn(1, heads, l_q, dim, dtype=torch.float64)
        k = torch.randn(1, heads, l_k, dim, dtype=torch.float64)
        v = torch.randn(1, heads, l_k, dim, dtype=torch.float64)
        mask = torch.randn(1, 1, l_q, l_k, dtype=torch.float64)
        return q, k, v, mask, dim**-0.5

    def test_fold_helper_matches_spec(self):
        from esp_ppq.executor.op.torch.espdl_attention import compose_causal_into_mask

        for l_q, l_k in self.SHAPES:
            with self.subTest(l_q=l_q, l_k=l_k):
                q, k, v, mask, scale = self._qkv(l_q, l_k)
                folded, causal = compose_causal_into_mask(mask, True, q, k)
                self.assertFalse(causal, "causal must be folded away, not forwarded")
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=folded, dropout_p=0.0, is_causal=causal, scale=scale
                )
                self.assertTrue(torch.allclose(out, _sdpa_reference(q, k, v, mask, True, scale)))

    def test_fold_helper_is_a_noop_without_both(self):
        from esp_ppq.executor.op.torch.espdl_attention import compose_causal_into_mask

        q, k, _, mask, _ = self._qkv(8, 8)
        self.assertEqual(compose_causal_into_mask(mask, False, q, k), (mask, False))
        self.assertEqual(compose_causal_into_mask(None, True, q, k), (None, True))

    def test_generic_path_composes_mask_and_causal(self):
        from esp_ppq.executor.op.torch.default import _onnx23_attention_forward

        for l_q, l_k in self.SHAPES:
            with self.subTest(l_q=l_q, l_k=l_k):
                q, k, v, mask, scale = self._qkv(l_q, l_k)
                op = _attention_op({"is_causal": 1, "q_num_heads": 2, "kv_num_heads": 2}, num_inputs=4)
                out = _onnx23_attention_forward(op=op, values=[q, k, v, mask])
                self.assertTrue(torch.allclose(out, _sdpa_reference(q, k, v, mask, True, scale)))

    def test_generic_path_composes_bool_mask_and_causal(self):
        from esp_ppq.executor.op.torch.default import _onnx23_attention_forward

        q, k, v, _, scale = self._qkv(6, 6)
        keep = torch.rand(1, 1, 6, 6) > 0.3
        keep[..., 0] = True  # keep every row non-degenerate
        additive = torch.zeros(1, 1, 6, 6, dtype=torch.float64).masked_fill(~keep, float("-inf"))
        op = _attention_op({"is_causal": 1, "q_num_heads": 2, "kv_num_heads": 2}, num_inputs=4)
        out = _onnx23_attention_forward(op=op, values=[q, k, v, keep])
        self.assertTrue(torch.allclose(out, _sdpa_reference(q, k, v, additive, True, scale)))

    def test_fused_path_composes_mask_and_causal(self):
        from esp_ppq.executor.op.torch.default import Attention_forward

        for l_q, l_k in self.SHAPES:
            with self.subTest(l_q=l_q, l_k=l_k):
                q, k, v, mask, scale = self._qkv(l_q, l_k)
                shapes = [list(t.shape) for t in (q, k, v)]
                op = _attention_op({"is_causal": 1, "scale": scale}, num_inputs=4)
                self.assertTrue(espdl_attention_eligible(op, shapes), "expected the fused path")
                out = Attention_forward(op=op, values=[q, k, v, mask])[0]
                # espdl_attention_forward casts Q/K/V to fp32.
                self.assertTrue(torch.allclose(out, _sdpa_reference(q, k, v, mask, True, scale), atol=1e-6, rtol=1e-5))


class TestDeviceAttrLowering(unittest.TestCase):
    """``.espdl`` export keeps only the fields the device deserializer reads."""

    def test_export_drops_defaults_and_executor_only_fields(self):
        lowered = lower_espdl_attention_attributes(
            {
                "scale": 0.125,
                "is_causal": 0,
                "q_num_heads": 4,
                "kv_num_heads": 4,
                "softcap": 0.0,
                "layout": "BHND",
                "quant_type": 1,
            }
        )
        self.assertEqual(set(lowered), {"scale", "quant_type"})
        self.assertEqual(lowered["scale"], 0.125)

    def test_export_keeps_causal_and_bnhd(self):
        self.assertEqual(
            lower_espdl_attention_attributes({"is_causal": 1, "layout": "bnhd"}),
            {"is_causal": 1, "layout": "BNHD"},
        )


class TestConstantMaskFusion(unittest.TestCase):
    """A constant attn_mask (graph initializer) fused the way the quant pipeline runs it."""

    def test_constant_mask_is_kept_and_graph_is_clean(self):
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

        torch.manual_seed(0)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 6:] = float("-inf")

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mask", mask)

            def forward(self, q, k, v):
                s = torch.matmul(q, k.transpose(-1, -2)) * 0.25 + self.mask
                return torch.matmul(torch.softmax(s, -1), v)

        q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
        golden = M()(q, k, v)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    M(),
                    (q, k, v),
                    path,
                    input_names=["q", "k", "v"],
                    output_names=["out"],
                    opset_version=18,
                    dynamo=False,
                )
            graph = load_onnx_graph(path)
            feed = [{"q": q, "k": k, "v": v}[n] for n in graph.inputs.keys()]
            executor = TorchExecutor(graph=graph, device="cpu")
            # No explicit tracing: the pass must trace on its own.
            fuse_espdl_attention_graph(graph, executor=executor, inputs=feed)
            out = TorchExecutor(graph=graph, device="cpu").forward(inputs=feed)[0]

        attentions = [op for op in graph.operations.values() if op.type == "Attention"]
        self.assertEqual(len(attentions), 1)
        fused_mask = attentions[0].inputs[3]
        self.assertTrue(fused_mask.is_parameter)
        self.assertIn(fused_mask.name, graph.variables)
        self.assertIn(attentions[0], fused_mask.dest_ops)
        torch.testing.assert_close(fused_mask.value, mask)
        orphans = [
            var.name
            for var in graph.variables.values()
            if var.source_op is None
            and not var.dest_ops
            and var.name not in graph.inputs
            and var.name not in graph.outputs
        ]
        self.assertEqual(orphans, [])
        torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)


def _primitive_attention_onnx(path, mask):
    """Write ``softmax(q·kᵀ·0.25 + mask)·v`` as primitive ONNX ops."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mask", mask)

        def forward(self, q, k, v):
            s = torch.matmul(q, k.transpose(-1, -2)) * 0.25 + self.mask
            return torch.matmul(torch.softmax(s, -1), v)

    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    with _silence_export_logs():
        torch.onnx.export(
            M(),
            (q, k, v),
            path,
            input_names=["q", "k", "v"],
            output_names=["out"],
            opset_version=18,
            dynamo=False,
        )
    return {"q": q, "k": k, "v": v}, M()(q, k, v)


class TestFp32PathFusion(unittest.TestCase):
    """quant_type='none' must yield the same Attention nodes as a quantized export."""

    def _run(self, mask, platform, fuse=True):
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import (
            fuse_espdl_attention_graph,
            legalize_espdl_attention_graph,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            feed, golden = _primitive_attention_onnx(path, mask)
            graph = load_onnx_graph(path)
            ordered = [feed[n] for n in graph.inputs.keys()]
            executor = TorchExecutor(graph=graph, device="cpu")
            executor.tracing_operation_meta(inputs=ordered)
            if fuse:
                fuse_espdl_attention_graph(graph, executor=executor, inputs=ordered)
            legalize_espdl_attention_graph(graph, platform=platform, quantized=False, executor=executor, inputs=ordered)
            out = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)[0]
            torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)
            return [op for op in graph.operations.values() if op.type == "Attention"]

    def test_p4_fuses_and_compacts_inf_mask(self):
        from esp_ppq.core import DataType, TargetPlatform

        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 6:] = float("-inf")
        attentions = self._run(mask, TargetPlatform.ESPDL_INT8)
        self.assertEqual(len(attentions), 1)
        fused_mask = attentions[0].inputs[3]
        self.assertEqual(fused_mask.dtype, DataType.BOOL)
        torch.testing.assert_close(fused_mask.value, mask == 0)

    def test_finite_mask_stays_fp32(self):
        from esp_ppq.core import DataType, TargetPlatform

        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 6:] = -1e4
        attentions = self._run(mask, TargetPlatform.ESPDL_INT8)
        self.assertEqual(len(attentions), 1)
        self.assertEqual(attentions[0].inputs[3].dtype, DataType.FP32)

    def test_graph_output_mask_and_scale_parameter_remain_intact(self):
        from esp_ppq.core import DataType, TargetPlatform
        from esp_ppq.quantization.optim.espdl_attention import (
            fuse_espdl_attention_graph,
            legalize_espdl_attention_graph,
        )

        graph = TestDeployabilityGuard._graph_with([[1, 4, 8, 16]] * 3 + [[1, 1, 8, 8]], num_inputs=4)
        mask_var = graph.operations["attn"].inputs[3]
        mask_var.is_parameter = True
        mask_var.value = torch.tensor([0.0, float("-inf")], dtype=torch.float32).repeat(1, 1, 8, 4)
        mask_var.dtype = DataType.FP32
        graph.mark_variable_as_graph_output(mask_var)

        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8, quantized=False)
        self.assertEqual(mask_var.dtype, DataType.FP32)
        self.assertTrue(mask_var.value.is_floating_point())
        self.assertIs(graph.outputs[mask_var.name], mask_var)

        from esp_ppq.IR import BaseGraph

        prim = BaseGraph(name="prim_scale_graph_out")
        shape = [1, 2, 4, 8]
        q_in, k_in, v_in = (prim.create_variable(name=n) for n in ("q", "k", "v"))
        for var in (q_in, k_in, v_in):
            var.shape = list(shape)
            prim.mark_variable_as_graph_input(var)
        kt = prim.create_variable(name="kt")
        kt.shape = [1, 2, 8, 4]
        scores = prim.create_variable(name="scores")
        scores.shape = [1, 2, 4, 4]
        scaled = prim.create_variable(name="scaled")
        scaled.shape = [1, 2, 4, 4]
        probs = prim.create_variable(name="probs")
        probs.shape = [1, 2, 4, 4]
        out = prim.create_variable(name="out")
        out.shape = list(shape)
        scale_param = prim.create_variable(name="scale_param", value=torch.tensor(0.25), is_parameter=True)
        scale_param.shape = []
        prim.create_operation(
            op_type="Transpose", name="kT", attributes={"perm": [0, 1, 3, 2]}, inputs=[k_in], outputs=[kt]
        )
        prim.create_operation(op_type="MatMul", name="qk", inputs=[q_in, kt], outputs=[scores])
        prim.create_operation(op_type="Mul", name="scale_mul", inputs=[scores, scale_param], outputs=[scaled])
        prim.create_operation(op_type="Softmax", name="sm", attributes={"axis": -1}, inputs=[scaled], outputs=[probs])
        prim.create_operation(op_type="MatMul", name="pv", inputs=[probs, v_in], outputs=[out])
        prim.mark_variable_as_graph_output(out)
        prim.mark_variable_as_graph_output(scale_param)

        fuse_espdl_attention_graph(prim)
        self.assertTrue(any(op.type == "Attention" for op in prim.operations.values()))
        self.assertIs(prim.variables["scale_param"], scale_param)
        self.assertIs(prim.outputs["scale_param"], scale_param)
        self.assertIsNotNone(scale_param.value)

    def test_s3_and_c_fuse_unless_fusion_is_disabled(self):
        from esp_ppq.core import TargetPlatform

        mask = torch.zeros(1, 1, 8, 8)
        self.assertEqual(len(self._run(mask, TargetPlatform.ESPDL_S3_INT8)), 1)
        self.assertEqual(len(self._run(mask, TargetPlatform.ESPDL_C_INT8)), 1)
        self.assertEqual(self._run(mask, TargetPlatform.ESPDL_S3_INT8, fuse=False), [])
        self.assertEqual(self._run(mask, TargetPlatform.ESPDL_INT8, fuse=False), [])

    def test_shared_mask_is_not_compacted(self):
        from esp_ppq.core import DataType, TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph

        graph = TestDeployabilityGuard._graph_with([[1, 4, 16, 32]] * 3)
        op = graph.operations["attn"]
        mask = torch.zeros(1, 1, 16, 16).masked_fill(torch.ones(16, 16, dtype=torch.bool).triu(1), float("-inf"))
        var = graph.create_variable(value=mask, is_parameter=True)
        graph.create_link_with_op(None, op, var)
        var.shape = list(mask.shape)
        var.dest_ops.append(graph.create_operation(op_type="Identity", name="other"))
        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8)
        self.assertEqual(op.inputs[3].dtype, DataType.FP32)


class TestFusionDecision(unittest.TestCase):
    """Fusion judges a match with legalize's rule before touching the graph, and
    the export summary reports every outcome once."""

    @staticmethod
    def _export_and_legalize(module, feed, platform, output_names=("out",)):
        import os
        import tempfile
        from unittest import mock

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim import espdl_attention
        from esp_ppq.quantization.optim.espdl_attention import (
            fuse_espdl_attention_graph,
            legalize_espdl_attention_graph,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    module,
                    tuple(feed.values()),
                    path,
                    input_names=list(feed.keys()),
                    output_names=list(output_names),
                    opset_version=18,
                    dynamo=False,
                )
            graph = load_onnx_graph(path)
        ordered = [feed[n] for n in graph.inputs.keys()]
        TorchExecutor(graph=graph, device="cpu").tracing_operation_meta(inputs=ordered)
        logs = {"info": [], "warning": []}
        with (
            mock.patch.object(espdl_attention.logger, "info", side_effect=logs["info"].append),
            mock.patch.object(espdl_attention.logger, "warning", side_effect=logs["warning"].append),
        ):
            fuse_espdl_attention_graph(graph)
            legalize_espdl_attention_graph(graph, platform=platform, quantized=False)
        outputs = TorchExecutor(graph=graph, device="cpu").forward(inputs=ordered)
        return graph, logs, outputs

    @staticmethod
    def _build_attention_model(scale=4.0):
        class M(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / scale, -1), v)

        return M()

    def test_fused_node_is_kept_and_summarised(self):
        from esp_ppq.core import TargetPlatform

        torch.manual_seed(0)
        feed = {n: torch.randn(1, 2, 8, 16) for n in "qkv"}
        graph, logs, _ = self._export_and_legalize(self._build_attention_model(), feed, TargetPlatform.ESPDL_INT8)
        self.assertEqual(sum(op.type == "Attention" for op in graph.operations.values()), 1)
        self.assertEqual(logs["warning"], [])
        self.assertEqual(len(logs["info"]), 1)
        self.assertIn("1 Attention kept for the device kernel (1 fused from MatMul/Softmax)", logs["info"][0])
        self.assertNotIn("expanded", logs["info"][0])

    def test_rejected_bnhd_match_leaves_graph_untouched(self):
        from esp_ppq.core import TargetPlatform

        class M(torch.nn.Module):
            def forward(self, q, k_bnhd, v):
                scores = torch.matmul(q, k_bnhd.permute(0, 2, 3, 1)) / 4.0
                return torch.matmul(torch.softmax(scores, -1), v), scores

        torch.manual_seed(0)
        B, H, L, D = 1, 2, 8, 16
        feed = {"q": torch.randn(B, H, L, D), "k_bnhd": torch.randn(B, L, H, D), "v": torch.randn(B, H, L, D)}
        golden = M()(*feed.values())
        graph, _, outputs = self._export_and_legalize(
            M(), feed, TargetPlatform.ESPDL_INT8, output_names=("out", "scores")
        )
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        perms = [list(op.attributes["perm"]) for op in graph.operations.values() if op.type == "Transpose"]
        self.assertEqual(perms, [[0, 2, 3, 1]])
        for out, ref in zip(outputs, golden):
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_oversized_head_dim_is_left_with_actionable_warning(self):
        from esp_ppq.core import TargetPlatform

        torch.manual_seed(0)
        feed = {n: torch.randn(1, 2, 8, ESPDL_ATTENTION_MAX_HEAD_DIM * 2) for n in "qkv"}
        graph, logs, _ = self._export_and_legalize(self._build_attention_model(), feed, TargetPlatform.ESPDL_INT8)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertEqual(len(logs["warning"]), 1)
        self.assertIn("left as MatMul/Softmax", logs["warning"][0])
        self.assertIn("ESP_DL_ATTENTION_MAX_HEAD_DIM", logs["warning"][0])
        self.assertIn("ESPDL_ATTENTION_MAX_HEAD_DIM", logs["warning"][0])

    def test_ceiling_hidden_behind_another_reason_is_info(self):
        # head_dim is over the ceiling, but the reported reason is the fp16 mask;
        # raising the knob would not fuse it, so no WARNING.
        from esp_ppq.core import TargetPlatform

        class M(torch.nn.Module):
            def forward(self, q, k, v):
                s = torch.matmul(q, k.transpose(-1, -2)) / 4.0 + torch.zeros(1, 1, 8, 8, dtype=torch.float16)
                return torch.matmul(torch.softmax(s, -1), v)

        torch.manual_seed(0)
        feed = {n: torch.randn(1, 2, 8, ESPDL_ATTENTION_MAX_HEAD_DIM * 2, dtype=torch.float16) for n in "qkv"}
        graph, logs, _ = self._export_and_legalize(M(), feed, TargetPlatform.ESPDL_INT8)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertEqual(logs["warning"], [])
        self.assertIn("attn_mask dtype FP16", "\n".join(logs["info"]))

    def test_untraced_match_raises_without_shape_trace(self):
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim.espdl_attention import (
            fuse_espdl_attention_graph,
            legalize_espdl_attention_graph,
        )

        feed = {n: torch.randn(1, 2, 8, 16) for n in "qkv"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    self._build_attention_model(),
                    tuple(feed.values()),
                    path,
                    input_names=list("qkv"),
                    output_names=["out"],
                    opset_version=18,
                    dynamo=False,
                )
                graph = load_onnx_graph(path)
            # Without traced shapes, MatMul/Softmax stays unfused (never half-fuses
            # from static ONNX input value_info).
            fuse_espdl_attention_graph(graph)
            legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8, quantized=False)
            self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))

            # A native Attention without traced shapes raises in legalize.
            attn_graph = TestDeployabilityGuard._graph_with([None, None, None])
        with self.assertRaises(ValueError) as ctx:
            legalize_espdl_attention_graph(attn_graph, platform=TargetPlatform.ESPDL_INT8, quantized=False)
        self.assertIn("shapes are still", str(ctx.exception))

    def test_untraced_match_is_fused_after_trace(self):
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.core import TargetPlatform
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import (
            fuse_espdl_attention_graph,
            legalize_espdl_attention_graph,
        )

        feed = {n: torch.randn(1, 2, 8, 16) for n in "qkv"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    self._build_attention_model(),
                    tuple(feed.values()),
                    path,
                    input_names=list("qkv"),
                    output_names=["out"],
                    opset_version=18,
                    dynamo=False,
                )
                graph = load_onnx_graph(path)
        ordered = [feed[n] for n in graph.inputs.keys()]
        executor = TorchExecutor(graph=graph, device="cpu")
        fuse_espdl_attention_graph(graph, executor=executor, inputs=ordered)
        legalize_espdl_attention_graph(
            graph, platform=TargetPlatform.ESPDL_INT8, quantized=False, executor=executor, inputs=ordered
        )
        self.assertEqual(sum(op.type == "Attention" for op in graph.operations.values()), 1)

    def test_non_ceiling_rejection_is_info_only(self):
        from esp_ppq.core import TargetPlatform

        torch.manual_seed(0)
        feed = {n: torch.randn(8, 16) for n in "qkv"}
        graph, logs, _ = self._export_and_legalize(self._build_attention_model(), feed, TargetPlatform.ESPDL_INT8)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertEqual(logs["warning"], [])
        joined = "\n".join(logs["info"])
        self.assertIn("1 MatMul/Softmax pattern(s) left unfused", joined)
        self.assertIn("matched attention rank 2 is not 3-D or 4-D", joined)

    def test_platform_without_kernel_expands_at_info(self):
        from unittest import mock

        from esp_ppq.core import TargetPlatform
        from esp_ppq.quantization.optim import espdl_attention, legalize_espdl_attention_graph

        graph = TestDeployabilityGuard._graph_with([[1, 4, 16, ESPDL_ATTENTION_MAX_HEAD_DIM + 1]] * 3)
        logs = {"info": [], "warning": []}
        with (
            mock.patch.object(espdl_attention.logger, "info", side_effect=logs["info"].append),
            mock.patch.object(espdl_attention.logger, "warning", side_effect=logs["warning"].append),
        ):
            legalize_espdl_attention_graph(graph, platform=TargetPlatform.FP32)
        self.assertEqual(logs["warning"], [])
        joined = "\n".join(logs["info"])
        self.assertIn("0 Attention kept for the device kernel, 1 expanded to MatMul/Softmax", joined)
        self.assertIn("expanded to MatMul/Softmax (1: attn)", joined)

    def test_graph_without_attention_logs_nothing(self):
        from unittest import mock

        from esp_ppq.core import TargetPlatform
        from esp_ppq.IR import BaseGraph
        from esp_ppq.quantization.optim import espdl_attention, legalize_espdl_attention_graph

        graph = BaseGraph(name="g")
        graph.create_operation(op_type="MatMul", name="mm")
        with mock.patch.object(espdl_attention.logger, "info") as info:
            legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8)
        info.assert_not_called()


class TestMaskStaysFp32(unittest.TestCase):
    """attn_mask is never quantized: it is bool or fp32, and int8/int16 cannot hold -inf."""

    @staticmethod
    def _attention_graph(num_inputs: int):
        from esp_ppq.IR import BaseGraph

        graph = BaseGraph(name="g")
        ins = []
        for i in range(num_inputs):
            var = graph.create_variable(name=f"in{i}")
            var.shape = [1, 4, 16, 32] if i < 3 else [1, 1, 16, 16]
            graph.mark_variable_as_graph_input(var)
            ins.append(var)
        out = graph.create_variable(name="out")
        graph.create_operation(op_type="Attention", name="attn", inputs=ins, outputs=[out])
        graph.mark_variable_as_graph_output(out)
        return graph

    def _init_config(self, quantizer_cls, platform, num_inputs: int):
        from esp_ppq.core import TargetPlatform

        graph = self._attention_graph(num_inputs)
        op = graph.operations["attn"]
        op.platform = getattr(TargetPlatform, platform)
        return quantizer_cls(graph).init_quantize_config(op)

    def test_mask_is_fp32_on_int8_and_int16(self):
        from esp_ppq.core import QuantizationStates
        from esp_ppq.quantization.quantizer.EspdlQuantizer import EspdlInt16Quantizer, EspdlQuantizer

        for cls, platform in ((EspdlQuantizer, "ESPDL_INT8"), (EspdlInt16Quantizer, "ESPDL_INT16")):
            with self.subTest(platform=platform):
                config = self._init_config(cls, platform, num_inputs=4)
                self.assertEqual(config.input_quantization_config[3].state, QuantizationStates.FP32)
                for i in range(3):
                    self.assertNotEqual(config.input_quantization_config[i].state, QuantizationStates.FP32)

    def test_custom_bit_width_does_not_requantize_mask(self):
        from esp_ppq.core import QuantizationStates, TargetPlatform
        from esp_ppq.quantization.quantizer.EspdlQuantizer import EspdlQuantizer

        graph = self._attention_graph(4)
        op = graph.operations["attn"]
        op.platform = TargetPlatform.ESPDL_INT8
        quantizer = EspdlQuantizer(graph)
        quantizer._custom_tqc = {"attn": {"input_3": {"bit_width": 16}}}
        config = quantizer.init_quantize_config(op)
        self.assertEqual(config.input_quantization_config[3].state, QuantizationStates.FP32)

    def test_three_input_attention_is_unaffected(self):
        from esp_ppq.core import QuantizationStates
        from esp_ppq.quantization.quantizer.EspdlQuantizer import EspdlQuantizer

        config = self._init_config(EspdlQuantizer, "ESPDL_INT8", num_inputs=3)
        self.assertEqual(len(config.input_quantization_config), 3)
        for tqc in config.input_quantization_config:
            self.assertNotEqual(tqc.state, QuantizationStates.FP32)

    def test_exporter_rejects_quantized_mask(self):
        from esp_ppq.core import QuantizationStates
        from esp_ppq.parser.espdl_exporter import EspdlExporter

        graph = _activate_qk_scales(
            TestDeployabilityGuard._graph_with([[1, 4, 16, 32]] * 3 + [[1, 1, 16, 16]], num_inputs=4)
        )
        op = graph.operations["attn"]
        exporter = EspdlExporter()
        with self.assertRaises(ValueError) as ctx:
            exporter.build_operator_proto(op)
        self.assertIn("must stay FP32", str(ctx.exception))

        op.input_quant_config[3].state = QuantizationStates.FP32
        self.assertIsInstance(exporter.build_operator_proto(op), int)

    def test_direct_exporter_preflights_attention_contract(self):
        import os
        import tempfile

        from esp_ppq.core import DataType, TargetPlatform
        from esp_ppq.IR import BaseGraph
        from esp_ppq.parser.espdl_exporter import EspdlExporter

        def build_float_graph(attributes=None):
            graph = BaseGraph(name="direct_attention_export")
            shape = [1, 4, 8, 32]
            inputs = []
            for name in ("q", "k", "v"):
                var = graph.create_variable(name=name)
                var.shape, var.dtype = list(shape), DataType.FP32
                graph.mark_variable_as_graph_input(var)
                inputs.append(var)
            out = graph.create_variable(name="out")
            out.shape, out.dtype = list(shape), DataType.FP32
            graph.create_operation(
                op_type="Attention",
                name="attn",
                attributes=dict(attributes or {}),
                platform=TargetPlatform.FP32,
                inputs=inputs,
                outputs=[out],
            )
            graph.mark_variable_as_graph_output(out)
            return graph

        with tempfile.TemporaryDirectory() as tmp:
            softcap_graph = build_float_graph({"softcap": 1.0})
            with self.assertRaises(ValueError) as ctx:
                EspdlExporter().export(
                    os.path.join(tmp, "softcap.espdl"),
                    softcap_graph,
                    export_config=False,
                )
            self.assertIn("unsupported attrs ['softcap']", str(ctx.exception))
            self.assertEqual(softcap_graph.operations["attn"].attributes.get("softcap"), 1.0)

            bad_scale_graph = _activate_qk_scales(
                TestDeployabilityGuard._graph_with([[1, 4, 16, 32]] * 3),
                scale=float(2**80),
            )
            with self.assertRaises(ValueError) as ctx:
                EspdlExporter().export(
                    os.path.join(tmp, "bad_scale.espdl"),
                    bad_scale_graph,
                    export_config=False,
                )
            self.assertIn("logit_scale", str(ctx.exception))

            valid_float_graph = build_float_graph({"is_causal": 1})
            float_path = os.path.join(tmp, "valid_float.espdl")
            EspdlExporter().export(float_path, valid_float_graph, export_config=False)
            self.assertTrue(os.path.isfile(float_path))
            self.assertGreater(os.path.getsize(float_path), 0)
            self.assertEqual(valid_float_graph.operations["attn"].attributes, {"is_causal": 1})


class TestOnnxAttentionDomain(unittest.TestCase):
    """Only com.microsoft.Attention belongs in the com.microsoft domain."""

    def setUp(self):
        from esp_ppq.IR import BaseGraph
        from esp_ppq.parser.onnx_exporter import AttentionExporter

        self.exporter = AttentionExporter()
        self.graph = BaseGraph(name="g")

    def _export(self, attributes):
        op = _attention_op(attributes)
        return self.exporter.export(op=op, graph=self.graph)

    def test_standard_attention_keeps_default_domain(self):
        op = self._export({"scale": 0.125, "is_causal": 1})
        self.assertNotIn("domain", op.attributes)

    def test_microsoft_attention_is_stamped(self):
        for attr in ("qkv_hidden_sizes", "num_heads"):
            with self.subTest(attr=attr):
                op = self._export({attr: 4})
                self.assertEqual(op.attributes["domain"], "com.microsoft")


def _native_attention_block_model(head_dim: int = 8):
    """Relu -> Identity -> ai.onnx::Attention (opset 23) -> Identity -> Add.

    Identity elimination runs without onnxruntime, so the test does not depend
    on the installed onnxruntime supporting opset 23 for constant folding.
    """
    shape = [1, 2, 4, head_dim]
    q = onnx.helper.make_tensor_value_info("q", onnx.TensorProto.FLOAT, shape)
    out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, shape)
    bias = onnx.numpy_helper.from_array(np.ones(shape, dtype=np.float32), "bias")
    nodes = [
        onnx.helper.make_node("Relu", ["q"], ["r"]),
        onnx.helper.make_node("Identity", ["r"], ["r2"]),
        onnx.helper.make_node("Attention", ["r2", "r2", "r2"], ["a"], is_causal=1),
        onnx.helper.make_node("Identity", ["a"], ["a2"]),
        onnx.helper.make_node("Add", ["a2", "bias"], ["out"]),
    ]
    graph = onnx.helper.make_graph(nodes, "attn_block", [q], [out], initializer=[bias])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 23)])
    model.ir_version = 8
    return model


class TestNativeAttentionMetadataTrace(unittest.TestCase):
    def test_oversized_opset23_attention_is_legalized_before_real_trace(self):
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.core import TargetPlatform
        from esp_ppq.parser.espdl_exporter import EspdlExporter

        head_dim = ESPDL_ATTENTION_MAX_HEAD_DIM + 1
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "native_attention.onnx")
            onnx.save(_native_attention_block_model(head_dim), path)
            graph = load_onnx_graph(path)
            EspdlExporter.prepare_graph_for_float_export(
                graph,
                platform=TargetPlatform.ESPDL_INT8,
                inputs=[torch.randn(1, 2, 4, head_dim)],
                device="cpu",
            )

        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "MatMul" for op in graph.operations.values()))
        self.assertTrue(any(op.type == "Softmax" for op in graph.operations.values()))
        self.assertEqual(graph.outputs["out"].shape, [1, 2, 4, head_dim])


class TestOnnxsimNativeOpset23(unittest.TestCase):
    """onnxsim itself simplifies around a native opset-23 Attention node."""

    def test_rest_of_graph_is_simplified(self):
        from onnxsim import simplify

        model = _native_attention_block_model()
        before = onnx.ModelProto()
        before.CopyFrom(model)
        sim, check = simplify(model)

        self.assertTrue(check)
        self.assertEqual(model, before, "input model must not be mutated")
        ops = [(n.domain, n.op_type) for n in sim.graph.node]
        self.assertIn(("", "Attention"), ops)
        attn = next(n for n in sim.graph.node if n.op_type == "Attention")
        self.assertEqual([(a.name, a.i) for a in attn.attribute], [("is_causal", 1)])
        self.assertNotIn("Identity", [t for _, t in ops])
        self.assertLess(len(sim.graph.node), len(model.graph.node))
        self.assertTrue(all(v.type.tensor_type.HasField("shape") for v in sim.graph.output))


class TestReviewBugRegressions(unittest.TestCase):
    """Regressions for bugs uncovered during the esp-ppq review."""

    def test_fused_where_mask_exports_espdl_and_drops_k_transpose(self):
        import os
        import tempfile

        from esp_ppq.api.espdl_interface import espdl_quantize_onnx

        class FusedWhereAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cond", torch.tril(torch.ones(8, 8, dtype=torch.bool)).view(1, 1, 8, 8))

            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2)) / 4.0
                masked = torch.where(self.cond, scores, torch.tensor(float("-inf")))
                return torch.matmul(torch.softmax(masked, -1), v)

        q = torch.randn(1, 2, 8, 16)
        feed = [q, q.clone(), q.clone()]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            espdl_path = os.path.join(tmp, "m.espdl")
            with _silence_export_logs():
                torch.onnx.export(
                    FusedWhereAttn().eval(),
                    tuple(feed),
                    path,
                    input_names=["q", "k", "v"],
                    output_names=["out"],
                    opset_version=18,
                    dynamo=False,
                )
                graph = espdl_quantize_onnx(
                    onnx_import_file=path,
                    espdl_export_file=espdl_path,
                    calib_dataloader=[feed, [t.clone() for t in feed]],
                    calib_steps=2,
                    input_shape=[list(t.shape) for t in feed],
                    inputs=feed,
                    target="esp32p4",
                    quant_type="w8a8",
                    device="cpu",
                    error_report=False,
                    skip_export=False,
                    verbose=0,
                )
            self.assertTrue(os.path.exists(espdl_path))
            self.assertNotIn("Transpose", [op.type for op in graph.operations.values()])
            self.assertIn("Attention", [op.type for op in graph.operations.values()])

    def test_unfused_or_expanded_neg_inf_mask_calibrates_and_exports_f32_softmax(self):
        import os
        import tempfile

        from esp_ppq.api.espdl_interface import espdl_quantize_onnx

        class OversizedWhereAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cond", torch.tril(torch.ones(8, 8, dtype=torch.bool)).view(1, 1, 8, 8))

            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2)) / 8.0
                masked = torch.where(self.cond, scores, torch.tensor(float("-inf")))
                return torch.matmul(torch.softmax(masked, -1), v)

        class CausalSdpa(torch.nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        head_dim = ESPDL_ATTENTION_MAX_HEAD_DIM + 1
        q = torch.randn(1, 2, 8, head_dim)
        feed = [q, q.clone(), q.clone()]
        with tempfile.TemporaryDirectory() as tmp:
            for idx, mod in enumerate((OversizedWhereAttn().eval(), CausalSdpa().eval())):
                path = os.path.join(tmp, f"m_{idx}.onnx")
                espdl_path = os.path.join(tmp, f"m_{idx}.espdl")
                info_path = os.path.join(tmp, f"m_{idx}.info")
                with _silence_export_logs():
                    torch.onnx.export(
                        mod,
                        tuple(feed),
                        path,
                        input_names=["q", "k", "v"],
                        output_names=["out"],
                        opset_version=18,
                        dynamo=False,
                    )
                    espdl_quantize_onnx(
                        onnx_import_file=path,
                        espdl_export_file=espdl_path,
                        calib_dataloader=[feed, [t.clone() for t in feed]],
                        calib_steps=2,
                        input_shape=[list(t.shape) for t in feed],
                        inputs=feed,
                        target="esp32p4",
                        quant_type="w8a8",
                        device="cpu",
                        error_report=False,
                        skip_export=False,
                        verbose=0,
                    )
                with open(info_path) as f:
                    info = f.read()
                self.assertIn("Softmax[axis = -1, quant_type = 'F32']", info)

            native_path = os.path.join(tmp, "native_causal.onnx")
            native_espdl = os.path.join(tmp, "native_causal.espdl")
            native_info = os.path.join(tmp, "native_causal.info")
            onnx.save(_native_attention_block_model(head_dim), native_path)
            x = torch.randn(1, 2, 4, head_dim)
            with _silence_export_logs():
                espdl_quantize_onnx(
                    onnx_import_file=native_path,
                    espdl_export_file=native_espdl,
                    calib_dataloader=[[x], [x.clone()]],
                    calib_steps=2,
                    input_shape=[[1, 2, 4, head_dim]],
                    inputs=[x],
                    target="esp32p4",
                    quant_type="w8a8",
                    device="cpu",
                    error_report=False,
                    skip_export=False,
                    verbose=0,
                )
            with open(native_info) as f:
                info = f.read()
            self.assertIn("Softmax[axis = -1, quant_type = 'F32']", info)

    def test_graph_output_on_k_transpose_or_prescale_is_preserved(self):
        from esp_ppq.executor import TorchExecutor

        class KtOut(torch.nn.Module):
            def forward(self, q, k, v):
                kt = k.transpose(-1, -2)
                scores = torch.matmul(q * 0.5, kt * 0.5)
                out = torch.matmul(torch.softmax(scores, -1), v)
                return out, kt

        class WhereOut(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cond", torch.tril(torch.ones(6, 6, dtype=torch.bool)).view(1, 1, 6, 6))

            def forward(self, q, k, v):
                scores = torch.matmul(q, k.transpose(-1, -2)) * 0.25
                masked = torch.where(self.cond, scores, torch.tensor(float("-inf")))
                out = torch.matmul(torch.softmax(masked, -1), v)
                return out, masked

        q = torch.randn(1, 2, 6, 16)
        k = torch.randn(1, 2, 6, 16)
        v = torch.randn(1, 2, 6, 16)
        graph, feed = _export_trace_and_fuse(KtOut().eval(), (q, k, v), ["q", "k", "v"])
        self.assertTrue(any(op.type == "Attention" for op in graph.operations.values()))
        for out_var in graph.outputs.values():
            self.assertIsNotNone(out_var.source_op)
        got = TorchExecutor(graph=graph, device="cpu").forward(inputs=feed)
        ref = KtOut().eval()(q, k, v)
        torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-4)

        w_mod = WhereOut().eval()
        w_graph, w_feed = _export_trace_and_fuse(w_mod, (q, k, v), ["q", "k", "v"])
        self.assertEqual(len(w_graph.outputs), 2)
        self.assertFalse(any(op.type == "Where" for op in w_graph.operations.values()))
        self.assertTrue(any(op.type == "Add" for op in w_graph.operations.values()))
        self.assertFalse(any(op.type == "Attention" for op in w_graph.operations.values()))
        for out_var in w_graph.outputs.values():
            self.assertIsNotNone(out_var.source_op)
        w_got = TorchExecutor(graph=w_graph, device="cpu").forward(inputs=w_feed)
        w_ref = w_mod(q, k, v)
        torch.testing.assert_close(w_got[0], w_ref[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(w_got[1], w_ref[1], rtol=1e-4, atol=1e-4)

    def test_decompose_blocker_checks_bnhd_and_3d_and_constant_bool_mask(self):
        from esp_ppq.core import DataType, TargetPlatform
        from esp_ppq.quantization.optim import legalize_espdl_attention_graph
        from esp_ppq.quantization.optim.espdl_attention import _decompose_blocker

        # BNHD with batch mismatch must be blocked before decomposition
        op_bnhd = _attention_op({"layout": "BNHD"}, num_inputs=3)
        self.assertIn(
            "batch mismatch",
            _decompose_blocker(op_bnhd, [[1, 8, 4, 32], [2, 8, 4, 32], [2, 8, 4, 32]]),
        )
        # 3D with non-divisible hidden size must be blocked
        op_3d = _attention_op({"q_num_heads": 4, "kv_num_heads": 4}, num_inputs=3)
        self.assertIn(
            "not divisible",
            _decompose_blocker(op_3d, [[1, 8, 30], [1, 8, 32], [1, 8, 32]]),
        )
        # Constant-op bool mask on an expanded Attention must decompose cleanly
        d = ESPDL_ATTENTION_MAX_HEAD_DIM + 1
        q = torch.randn(1, 2, 4, d)
        graph, _ = TestAttentionDecompose._graph(q, q.clone(), q.clone())
        attn = graph.operations["attn"]
        keep = torch.tril(torch.ones(1, 1, 4, 4, dtype=torch.bool))
        mask_var = graph.create_variable(name="const_bool_mask", value=keep, is_parameter=False)
        mask_var.shape = [1, 1, 4, 4]
        mask_var.dtype = DataType.BOOL
        const_op = graph.create_operation(
            op_type="Constant",
            name="bool_mask_const",
            attributes={"value": keep},
            platform=TargetPlatform.FP32,
            inputs=[],
            outputs=[mask_var],
        )
        graph.create_link_with_op(const_op, attn, mask_var)
        legalize_espdl_attention_graph(graph, platform=TargetPlatform.ESPDL_INT8, quantized=True)
        self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))

    def test_float_export_logs_warning_when_all_candidates_rejected(self):
        import os
        import tempfile
        from unittest import mock

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.core import TargetPlatform
        from esp_ppq.parser.espdl_exporter import EspdlExporter
        from esp_ppq.quantization.optim import espdl_attention
        from esp_ppq.quantization.optim.espdl_attention import _REPORT_KEY

        class OversizedSdpa(torch.nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        d = ESPDL_ATTENTION_MAX_HEAD_DIM + 1
        q = torch.randn(1, 2, 4, d)
        feed = [q, q.clone(), q.clone()]
        warnings = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    OversizedSdpa().eval(),
                    tuple(feed),
                    path,
                    input_names=["q", "k", "v"],
                    output_names=["out"],
                    opset_version=18,
                    dynamo=False,
                )
                graph = load_onnx_graph(path)
            with (
                mock.patch.object(espdl_attention.logger, "info"),
                mock.patch.object(espdl_attention.logger, "warning", side_effect=warnings.append),
            ):
                EspdlExporter.prepare_graph_for_float_export(
                    graph,
                    platform=TargetPlatform.ESPDL_INT8,
                    inputs=feed,
                    device="cpu",
                )
        self.assertNotIn(_REPORT_KEY, graph.extension_attrib)
        self.assertTrue(any("head_dim=" in m for m in warnings), warnings)

    def test_axis_semantics_guards_reject_bnhd_and_non_head_merge(self):
        import os
        import tempfile

        from onnx import TensorProto, helper, numpy_helper

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import _report, fuse_espdl_attention_graph

        # 1. Reshape -> Transpose([0, 2, 1]) -> Reshape that merges [B, H*L, D]
        # instead of [B*H, L, D] must NOT be matched as head_merge.
        class BadMerge(torch.nn.Module):
            def forward(self, q, k, v):
                b, h, seq_len, d = k.shape
                kt = k.reshape(b, h * seq_len, d).transpose(1, 2).reshape(b, h, d, seq_len)
                scores = torch.matmul(q, kt)
                return torch.matmul(torch.softmax(scores, dim=-1), v)

        torch.manual_seed(0)
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        golden = BadMerge()(q, k, v)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_merge.onnx")
            with _silence_export_logs():
                torch.onnx.export(
                    BadMerge().eval(),
                    (q, k, v),
                    path,
                    input_names=["q", "k", "v"],
                    output_names=["out"],
                    opset_version=18,
                    dynamo=False,
                )
                graph = load_onnx_graph(path)
            executor = TorchExecutor(graph=graph, device="cpu")
            feed = [q, k, v]
            executor.tracing_operation_meta(inputs=feed)
            fuse_espdl_attention_graph(graph, executor=executor, inputs=feed)
            self.assertFalse(any(op.type == "Attention" for op in graph.operations.values()))
            out = executor.forward(inputs=feed)[0]
            torch.testing.assert_close(out, golden, rtol=1e-5, atol=1e-5)

            # 2. Untraced / traced subgraph with unequal axis-1 dims (e.g. BNHD with
            # Lq=8, Lk=4 or Lk=1 and H=4 on axis 2) must not fuse as BHND GQA.
            bnhd_gqa_like = os.path.join(tmp, "bnhd_gqa_like.onnx")
            nodes = [
                helper.make_node("Transpose", ["k"], ["kt"], name="kT", perm=[0, 1, 3, 2]),
                helper.make_node("MatMul", ["q", "kt"], ["scores"], name="qk"),
                helper.make_node("Softmax", ["scores"], ["probs"], name="sm", axis=-1),
                helper.make_node("MatMul", ["probs", "v"], ["out"], name="pv"),
            ]
            g_pb = helper.make_graph(
                nodes,
                "g",
                [
                    helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, 8, 4, 16]),
                    helper.make_tensor_value_info("k", TensorProto.FLOAT, [1, 1, 4, 16]),
                    helper.make_tensor_value_info("v", TensorProto.FLOAT, [1, 1, 4, 16]),
                ],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8, 4, 16])],
            )
            onnx.save(helper.make_model(g_pb, opset_imports=[helper.make_opsetid("", 18)]), bnhd_gqa_like)
            g_bnhd = load_onnx_graph(bnhd_gqa_like)
            bnhd_exec = TorchExecutor(graph=g_bnhd, device="cpu")
            bnhd_feed = [torch.zeros(1, 8, 4, 16), torch.zeros(1, 1, 4, 16), torch.zeros(1, 1, 4, 16)]
            fuse_espdl_attention_graph(g_bnhd, executor=bnhd_exec, inputs=bnhd_feed)
            self.assertFalse(any(op.type == "Attention" for op in g_bnhd.operations.values()))
            self.assertTrue(any("equal head counts" in r for r in _report(g_bnhd).left))

            # 3. 3D [B, N, H*D] -> 4D [B, N, H, D] BNHD Reshape feeding MatMul directly
            # without Transpose([0, 2, 1, 3]) to BHND must not be fused as BHND.
            bnhd_reshape_path = os.path.join(tmp, "bnhd_reshape.onnx")
            shape_init = numpy_helper.from_array(np.array([1, 8, 2, 16], dtype=np.int64), "bnhd_shape")
            rs_nodes = [
                helper.make_node("Reshape", [name, "bnhd_shape"], [f"{name}4"], name=f"rs_{name}")
                for name in ("q", "k", "v")
            ] + [
                helper.make_node("Transpose", ["k4"], ["kt"], name="kT", perm=[0, 1, 3, 2]),
                helper.make_node("MatMul", ["q4", "kt"], ["scores"], name="qk"),
                helper.make_node("Softmax", ["scores"], ["probs"], name="sm", axis=-1),
                helper.make_node("MatMul", ["probs", "v4"], ["out"], name="pv"),
            ]
            g_rs_pb = helper.make_graph(
                rs_nodes,
                "g_rs",
                [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 8, 32]) for name in ("q", "k", "v")],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
                initializer=[shape_init],
            )
            onnx.save(helper.make_model(g_rs_pb, opset_imports=[helper.make_opsetid("", 18)]), bnhd_reshape_path)
            g_rs = load_onnx_graph(bnhd_reshape_path)
            rs_exec = TorchExecutor(graph=g_rs, device="cpu")
            rs_feed = [torch.zeros(1, 8, 32) for _ in g_rs.inputs]
            fuse_espdl_attention_graph(g_rs, executor=rs_exec, inputs=rs_feed)
            self.assertFalse(any(op.type == "Attention" for op in g_rs.operations.values()))
            self.assertTrue(any("BNHD Reshape without transposing to BHND" in r for r in _report(g_rs).left))

    def test_scale_factor_chains_fuse_and_prune_dead_ops_and_tight_where_mask(self):
        import os
        import tempfile

        from onnx import TensorProto, helper

        from esp_ppq.api import load_onnx_graph, quantize_onnx_model
        from esp_ppq.api.setting import QuantizationSettingFactory
        from esp_ppq.core import QuantizationStates, TargetPlatform
        from esp_ppq.IR import BaseGraph
        from esp_ppq.parser.espdl_exporter import EspdlExporter
        from esp_ppq.quantization.optim.espdl_attention import _attention_prequant_needs_trace
        from esp_ppq.quantization.quantizer.EspdlQuantizer import _var_carries_nonfinite_mask

        B, L, H, HD = 1, 8, 2, 16
        D = H * HD
        rng = np.random.default_rng(0)

        def build_chain_onnx(path: str, *, prescale: bool):
            x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, L, D])
            out = helper.make_tensor_value_info("o", TensorProto.FLOAT, [B, H, L, HD])

            def w(name):
                return helper.make_tensor(
                    name, TensorProto.FLOAT, [D, D], rng.standard_normal((D, D), dtype=np.float32).reshape(-1)
                )

            def proj(name, wname):
                return [
                    helper.make_node("MatMul", ["x", wname], [f"{name}_2d"]),
                    helper.make_node("Reshape", [f"{name}_2d", f"{name}_shape"], [f"{name}_4d"]),
                    helper.make_node("Transpose", [f"{name}_4d"], [name], perm=[0, 2, 1, 3]),
                ]

            q_name = "q0" if prescale else "q"
            nodes = []
            for name in (q_name, "k", "v"):
                nodes += proj(name, f"W{name}")
            nodes += [
                helper.make_node("Shape", [q_name], ["q_dims"]),
                helper.make_node("Slice", ["q_dims", "starts", "ends"], ["d_slice"]),
                helper.make_node("Cast", ["d_slice"], ["d_float"], to=TensorProto.FLOAT),
                helper.make_node("Sqrt", ["d_float"], ["sqrt_d"]),
            ]
            if prescale:
                nodes += [
                    helper.make_node("Div", ["one", "sqrt_d"], ["inv_sqrt_d"]),
                    helper.make_node("Mul", ["q0", "inv_sqrt_d"], ["q"]),
                    helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
                    helper.make_node("MatMul", ["q", "kt"], ["scores"]),
                ]
            else:
                nodes += [
                    helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
                    helper.make_node("MatMul", ["q", "kt"], ["qk"]),
                    helper.make_node("Div", ["qk", "sqrt_d"], ["scores"]),
                ]
            nodes += [
                helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
                helper.make_node("MatMul", ["probs", "v"], ["o"]),
            ]
            inits = [w(f"W{q_name}"), w("Wk"), w("Wv")]
            for name in (q_name, "k", "v"):
                inits.append(helper.make_tensor(f"{name}_shape", TensorProto.INT64, [4], [B, L, H, HD]))
            inits += [
                helper.make_tensor("starts", TensorProto.INT64, [1], [3]),
                helper.make_tensor("ends", TensorProto.INT64, [1], [4]),
            ]
            if prescale:
                inits.append(helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]))
            g_pb = helper.make_graph(nodes, "m", [x], [out], initializer=inits)
            m = helper.make_model(g_pb, opset_imports=[helper.make_opsetid("", 18)])
            m.ir_version = 8
            onnx.save(m, path)

        torch.manual_seed(0)
        sample = [torch.randn(B, L, D)]
        dead_types = {"Shape", "Slice", "Cast", "Sqrt", "Div"}
        with tempfile.TemporaryDirectory() as tmp:
            for prescale in (True, False):
                with self.subTest(prescale=prescale):
                    path = os.path.join(tmp, f"chain_{int(prescale)}.onnx")
                    build_chain_onnx(path, prescale=prescale)
                    loaded = load_onnx_graph(path)
                    self.assertTrue(_attention_prequant_needs_trace(loaded, TargetPlatform.ESPDL_INT8))
                    with _silence_export_logs():
                        quantized = quantize_onnx_model(
                            path,
                            calib_dataloader=[sample, [t.clone() for t in sample]],
                            calib_steps=2,
                            input_shape=None,
                            platform=TargetPlatform.ESPDL_INT8,
                            inputs=sample,
                            setting=QuantizationSettingFactory.espdl_setting(),
                            device="cpu",
                            verbose=0,
                        )
                    attns = [op for op in quantized.operations.values() if op.type == "Attention"]
                    self.assertEqual(len(attns), 1)
                    self.assertAlmostEqual(attns[0].attributes["scale"], HD**-0.5, places=6)
                    self.assertEqual([op.type for op in quantized.operations.values() if op.type in dead_types], [])

                    float_graph = load_onnx_graph(path)
                    with _silence_export_logs():
                        EspdlExporter.prepare_graph_for_float_export(
                            graph=float_graph, platform=TargetPlatform.ESPDL_INT8, inputs=sample, device="cpu"
                        )
                    self.assertEqual(sum(1 for op in float_graph.operations.values() if op.type == "Attention"), 1)
                    self.assertEqual([op.type for op in float_graph.operations.values() if op.type in dead_types], [])

        # Verify _var_carries_nonfinite_mask only flags Where when a value branch carries -inf
        g_where = BaseGraph(name="where_check")
        cond_v = g_where.create_variable(name="cond", value=torch.tensor([True, False]), is_parameter=True)
        x_v = g_where.create_variable(name="x")
        finite_y = g_where.create_variable(name="finite_y", value=torch.tensor(0.0), is_parameter=True)
        inf_y = g_where.create_variable(name="inf_y", value=torch.tensor(float("-inf")), is_parameter=True)
        out_finite = g_where.create_variable(name="out_finite")
        out_inf = g_where.create_variable(name="out_inf")
        g_where.create_operation(op_type="Where", name="w_finite", inputs=[cond_v, x_v, finite_y], outputs=[out_finite])
        g_where.create_operation(op_type="Where", name="w_inf", inputs=[cond_v, x_v, inf_y], outputs=[out_inf])
        self.assertFalse(_var_carries_nonfinite_mask(out_finite))
        self.assertTrue(_var_carries_nonfinite_mask(out_inf))
        _ = QuantizationStates.FP32

    def test_overlapping_add_matmul_candidates_do_not_crash(self):
        """When two MatMul branches feed the same Add -> Softmax -> MatMul, fusing the
        first candidate removes Add/Softmax/pv_matmul; the second candidate returned by
        pattern_matching in the same scale_type batch must be skipped cleanly instead of
        raising IndexError."""
        import os
        import tempfile

        from esp_ppq.api import load_onnx_graph
        from esp_ppq.executor import TorchExecutor
        from esp_ppq.quantization.optim.espdl_attention import fuse_espdl_attention_graph

        class DualMatMulAdd(torch.nn.Module):
            def __init__(self, scaled: bool):
                super().__init__()
                self.scaled = scaled

            def forward(self, qa, ka, qb, kb, v):
                sa = torch.matmul(qa, ka.transpose(-1, -2))
                sb = torch.matmul(qb, kb)
                if self.scaled:
                    sa = sa * 0.25
                    sb = sb * 0.5
                return torch.matmul(torch.softmax(sa + sb, dim=-1), v)

        torch.manual_seed(0)
        qa = torch.randn(1, 2, 8, 16)
        ka = torch.randn(1, 2, 8, 16)
        qb = torch.randn(1, 2, 8, 16)
        kb = torch.randn(1, 2, 16, 8)
        v = torch.randn(1, 2, 8, 16)

        with tempfile.TemporaryDirectory() as tmp:
            for scaled in (False, True):
                with self.subTest(scaled=scaled):
                    mod = DualMatMulAdd(scaled=scaled)
                    golden = mod(qa, ka, qb, kb, v)
                    path = os.path.join(tmp, f"dual_{int(scaled)}.onnx")
                    with _silence_export_logs():
                        torch.onnx.export(
                            mod,
                            (qa, ka, qb, kb, v),
                            path,
                            input_names=["qa", "ka", "qb", "kb", "v"],
                            output_names=["out"],
                            opset_version=18,
                            dynamo=False,
                        )
                    graph = load_onnx_graph(path)
                    feed_map = {"qa": qa, "ka": ka, "qb": qb, "kb": kb, "v": v}
                    feed = [feed_map[n] for n in graph.inputs.keys()]
                    executor = TorchExecutor(graph=graph, device="cpu")
                    fuse_espdl_attention_graph(graph, executor=executor, inputs=feed)
                    out = TorchExecutor(graph=graph, device="cpu").forward(inputs=feed)[0]

                    self.assertEqual(sum(1 for op in graph.operations.values() if op.type == "Attention"), 1)
                    torch.testing.assert_close(out, golden, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
