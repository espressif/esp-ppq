"""Tiny residual-attention net for BHND vs BNHD layout comparison.

Two ONNX graphs share weights and math. They only differ around Attention:

* BHND — reshape to ``[B, L, H, D]``, ``Transpose(0,2,1,3)`` into
  ``[B, H, L, D]``, fused Attention (default layout), transpose back.
* BNHD — keep ``[B, L, H, D]`` and set ``Attention.layout='BNHD'``.
  The device gathers adjacent heads into the existing K/V/Q scratch
  (``head-chunk``, no extra QKV copy); inner QACC kernels stay BHND.

No RoPE / MSDA / LayerNorm. Default presets::

    tiny     B=1 L=16   H=4 D=32  blocks=1   correctness
    mid      B=1 L=256  H=8 D=32  blocks=2   layout vs graph-transpose
    encoder  B=1 L=1601 H=8 D=32  blocks=1   ECViT-T 640² (40×40+CLS)

Usage::

    PYTHONPATH=/home/zhuwenjing/0910/esp-ppq:$PYTHONPATH \\
      python tests/attention_layout_net.py --preset tiny --check

    python tests/attention_layout_net.py --preset mid --out /tmp/attn_layout --quantize
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import OperatorSetIdProto, TensorProto, helper, numpy_helper


@dataclass(frozen=True)
class Spec:
    batch: int
    seq: int
    heads: int
    head_dim: int
    blocks: int

    @property
    def channels(self) -> int:
        return self.heads * self.head_dim

    @property
    def input_shape(self) -> List[int]:
        return [self.batch, self.seq, self.channels]


PRESETS: Dict[str, Spec] = {
    "tiny": Spec(batch=1, seq=16, heads=4, head_dim=32, blocks=1),
    "mid": Spec(batch=1, seq=256, heads=8, head_dim=32, blocks=2),
    "encoder": Spec(batch=1, seq=1601, heads=8, head_dim=32, blocks=1),
}


def make_weights(spec: Spec, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    c = spec.channels
    scale = float(c) ** -0.5
    weights: Dict[str, np.ndarray] = {}
    for i in range(spec.blocks):
        weights[f"Wqkv_{i}"] = rng.randn(c, 3 * c).astype(np.float32) * scale
        weights[f"Wo_{i}"] = rng.randn(c, c).astype(np.float32) * scale
    return weights


def reference_forward(x: torch.Tensor, spec: Spec, weights: Dict[str, np.ndarray]) -> torch.Tensor:
    """BHND SDPA reference. Output layout is always ``[B, L, C]``."""
    b, n, c = x.shape
    h, d = spec.heads, spec.head_dim
    y = x
    for i in range(spec.blocks):
        qkv = y @ torch.from_numpy(weights[f"Wqkv_{i}"]).to(dtype=y.dtype, device=y.device)
        q, k, v = qkv.split(c, dim=-1)
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(b, n, c)
        y = y + attn @ torch.from_numpy(weights[f"Wo_{i}"]).to(dtype=y.dtype, device=y.device)
    return y


class _Onnx:
    def __init__(self) -> None:
        self.nodes: List[onnx.NodeProto] = []
        self.inits: List[onnx.TensorProto] = []
        self._i = 0

    def uniq(self, prefix: str) -> str:
        self._i += 1
        return f"{prefix}_{self._i}"

    def const(self, name: str, array: np.ndarray) -> str:
        self.inits.append(numpy_helper.from_array(array, name=name))
        return name

    def op(self, op_type: str, inputs: List[str], outputs: List[str], **attrs) -> str:
        self.nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))
        return outputs[0]


def build_onnx(layout: str, spec: Spec, weights: Dict[str, np.ndarray]) -> onnx.ModelProto:
    """Build one residual-attention ONNX graph. ``layout`` is ``BHND`` or ``BNHD``."""
    layout = layout.upper()
    if layout not in ("BHND", "BNHD"):
        raise ValueError(f"layout must be BHND or BNHD, got {layout!r}")
    b, n, h, d = spec.batch, spec.seq, spec.heads, spec.head_dim
    c = spec.channels
    g = _Onnx()
    x = "x"
    cur = x
    shape_bnhd = g.const("shape_bnhd", np.array([b, n, h, d], dtype=np.int64))
    shape_blc = g.const("shape_blc", np.array([b, n, c], dtype=np.int64))
    perm_to_bhnd = [0, 2, 1, 3]

    for i in range(spec.blocks):
        wqkv = g.const(f"Wqkv_{i}", weights[f"Wqkv_{i}"])
        wo = g.const(f"Wo_{i}", weights[f"Wo_{i}"])
        residual = cur
        qkv = g.op("MatMul", [cur, wqkv], [f"qkv_{i}"])
        # Slice last axis into Q / K / V, each [B, L, C].
        q = _slice_last(g, qkv, 0, c, f"q_{i}")
        k = _slice_last(g, qkv, c, 2 * c, f"k_{i}")
        v = _slice_last(g, qkv, 2 * c, 3 * c, f"v_{i}")
        q = g.op("Reshape", [q, shape_bnhd], [f"q_bnhd_{i}"])
        k = g.op("Reshape", [k, shape_bnhd], [f"k_bnhd_{i}"])
        v = g.op("Reshape", [v, shape_bnhd], [f"v_bnhd_{i}"])
        attn_inputs = [q, k, v]
        attn_attrs = {}
        if layout == "BHND":
            q = g.op("Transpose", [q], [f"q_bhnd_{i}"], perm=perm_to_bhnd)
            k = g.op("Transpose", [k], [f"k_bhnd_{i}"], perm=perm_to_bhnd)
            v = g.op("Transpose", [v], [f"v_bhnd_{i}"], perm=perm_to_bhnd)
            attn_inputs = [q, k, v]
        else:
            attn_attrs["layout"] = "BNHD"
        y = g.op("Attention", attn_inputs, [f"attn_{i}"], **attn_attrs)
        if layout == "BHND":
            y = g.op("Transpose", [y], [f"attn_bnhd_{i}"], perm=perm_to_bhnd)
        y = g.op("Reshape", [y, shape_blc], [f"attn_blc_{i}"])
        y = g.op("MatMul", [y, wo], [f"proj_{i}"])
        cur = g.op("Add", [residual, y], [f"blk_{i}"])

    x_info = helper.make_tensor_value_info(x, TensorProto.FLOAT, spec.input_shape)
    y_info = helper.make_tensor_value_info(cur, TensorProto.FLOAT, spec.input_shape)
    graph = helper.make_graph(
        g.nodes,
        f"attn_layout_{layout.lower()}",
        [x_info],
        [y_info],
        g.inits,
    )
    opset = OperatorSetIdProto()
    opset.domain = ""
    opset.version = 23
    model = helper.make_model(
        graph,
        opset_imports=[opset],
        producer_name=f"attention_layout_net_{layout}",
    )
    model.ir_version = 8
    return model


def _slice_last(g: _Onnx, src: str, start: int, end: int, out: str) -> str:
    starts = g.const(g.uniq("starts"), np.array([start], dtype=np.int64))
    ends = g.const(g.uniq("ends"), np.array([end], dtype=np.int64))
    axes = g.const(g.uniq("axes"), np.array([2], dtype=np.int64))
    return g.op("Slice", [src, starts, ends, axes], [out])


def op_histogram(model: onnx.ModelProto) -> Counter:
    return Counter(n.op_type for n in model.graph.node)


def attention_layout_attr(model: onnx.ModelProto) -> List[str]:
    out = []
    for n in model.graph.node:
        if n.op_type != "Attention":
            continue
        raw = ""
        for a in n.attribute:
            if a.name == "layout":
                raw = a.s.decode("ascii") if a.s else ""
        out.append(raw.upper() or "BHND")
    return out


def _save(model: onnx.ModelProto, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    onnx.save(model, path)
    return path


def run_ppq(onnx_path: str, x: torch.Tensor) -> torch.Tensor:
    from esp_ppq.api.interface import load_onnx_graph
    from esp_ppq.executor import TorchExecutor

    graph = load_onnx_graph(onnx_import_file=onnx_path)
    ex = TorchExecutor(graph, device=str(x.device))
    out = ex.forward(inputs=x.detach().cpu())[0]
    return out.detach()


def compare_pair(
    spec: Spec,
    weights: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 0,
    work_dir: Optional[str] = None,
) -> dict:
    """Build both graphs, run PPQ fp32, compare against the SDPA reference."""
    weights = weights or make_weights(spec, seed=seed)
    own_tmp = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="attn_layout_")
    os.makedirs(work_dir, exist_ok=True)
    bhnd_path = os.path.join(work_dir, "bhnd.onnx")
    bnhd_path = os.path.join(work_dir, "bnhd.onnx")
    bhnd = build_onnx("BHND", spec, weights)
    bnhd = build_onnx("BNHD", spec, weights)
    _save(bhnd, bhnd_path)
    _save(bnhd, bnhd_path)

    rng = np.random.RandomState(seed + 1)
    x_np = rng.randn(*spec.input_shape).astype(np.float32)
    x = torch.from_numpy(x_np)
    golden = reference_forward(x, spec, weights)
    y_bhnd = run_ppq(bhnd_path, x)
    y_bnhd = run_ppq(bnhd_path, x)

    def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().max().item())

    result = {
        "spec": spec,
        "work_dir": work_dir,
        "owned_tmp": own_tmp,
        "bhnd_onnx": bhnd_path,
        "bnhd_onnx": bnhd_path,
        "bhnd_hist": op_histogram(bhnd),
        "bnhd_hist": op_histogram(bnhd),
        "bhnd_layouts": attention_layout_attr(bhnd),
        "bnhd_layouts": attention_layout_attr(bnhd),
        "maxdiff_bhnd_golden": _maxdiff(y_bhnd, golden),
        "maxdiff_bnhd_golden": _maxdiff(y_bnhd, golden),
        "maxdiff_bhnd_bnhd": _maxdiff(y_bhnd, y_bnhd),
        "bhnd": bhnd,
        "bnhd": bnhd,
    }
    return result


def pack_traffic(spec: Spec) -> dict:
    """Same PSRAM-transaction model as ``TestBnhdPackTraffic``."""
    h, d, lk = spec.heads, spec.head_dim, spec.seq
    token_bytes = h * d  # int8 [H, D] per token
    stride_txns = h * lk  # one D=32 row gather per (head, token)
    win = 16 * 1024
    chunk = max(1, win // token_bytes)
    pack_bursts = h * ((lk + chunk - 1) // chunk)
    return {
        "stride_txns": stride_txns,
        "pack_bursts": pack_bursts,
        "ratio": (stride_txns / pack_bursts) if pack_bursts else float("inf"),
        "chunk_tokens": chunk,
    }


def quantize_onnx(
    onnx_path: str,
    espdl_path: str,
    spec: Spec,
    target: str = "esp32p4",
    quant_type: str = "w8a8",
    calib_steps: int = 8,
) -> str:
    from torch.utils.data import DataLoader

    from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx

    b, n, c = spec.input_shape
    dataset = [torch.randn(n, c) for _ in range(max(calib_steps, 2))]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    setting = QuantizationSettingFactory.espdl_setting()
    setting.dispatcher = "allin"
    os.makedirs(os.path.dirname(os.path.abspath(espdl_path)) or ".", exist_ok=True)
    work_onnx = os.path.splitext(espdl_path)[0] + "_ppq_work.onnx"
    shutil.copy2(onnx_path, work_onnx)
    espdl_quantize_onnx(
        onnx_import_file=work_onnx,
        espdl_export_file=espdl_path,
        calib_dataloader=loader,
        calib_steps=calib_steps,
        input_shape=spec.input_shape,
        target=target,
        collate_fn=lambda t: t.type(torch.float32),
        setting=setting,
        device="cpu",
        error_report=False,
        skip_export=False,
        export_config=True,
        export_test_values=True,
        verbose=1,
        quant_type=quant_type,
        metadata_props={"target": target},
    )
    return espdl_path


def _fmt_hist(hist: Counter) -> str:
    keys = ("Attention", "Transpose", "MatMul", "Add", "Reshape", "Slice")
    parts = [f"{k}={hist.get(k, 0)}" for k in keys]
    extra = sum(v for k, v in hist.items() if k not in keys)
    if extra:
        parts.append(f"other={extra}")
    return " ".join(parts)


def print_report(result: dict) -> None:
    spec: Spec = result["spec"]
    print(f"spec  B={spec.batch} L={spec.seq} H={spec.heads} D={spec.head_dim} C={spec.channels} blocks={spec.blocks}")
    print(f"BHND  {_fmt_hist(result['bhnd_hist'])}  layout={result['bhnd_layouts']}")
    print(f"BNHD  {_fmt_hist(result['bnhd_hist'])}  layout={result['bnhd_layouts']}")
    traffic = pack_traffic(spec)
    print(
        f"PSRAM model (int8 K/V scan)  "
        f"BHND-stride txns={traffic['stride_txns']}  "
        f"BNHD-pack bursts={traffic['pack_bursts']}  "
        f"ratio={traffic['ratio']:.1f}x  chunk={traffic['chunk_tokens']} tokens/16KB"
    )
    print(
        f"fp32 max|diff|  BHND-golden={result['maxdiff_bhnd_golden']:.3e}  "
        f"BNHD-golden={result['maxdiff_bnhd_golden']:.3e}  "
        f"BHND-BNHD={result['maxdiff_bhnd_bnhd']:.3e}"
    )
    print(f"onnx  {result['bhnd_onnx']}")
    print(f"      {result['bnhd_onnx']}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    p.add_argument("--seq", type=int, default=None, help="Override sequence length.")
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Directory for ONNX / .espdl.")
    p.add_argument("--check", action="store_true", help="Compare fp32 and exit non-zero on mismatch.")
    p.add_argument("--quantize", action="store_true", help="Also emit int8 .espdl for both layouts.")
    p.add_argument("--target", default="esp32p4")
    p.add_argument("--quant-type", default="w8a8", choices=("w8a8", "w8a16", "w16a16", "none"))
    p.add_argument("--atol", type=float, default=2e-5)
    return p.parse_args(list(argv) if argv is not None else None)


def resolve_spec(args: argparse.Namespace) -> Spec:
    base = PRESETS[args.preset]
    return Spec(
        batch=base.batch,
        seq=base.seq if args.seq is None else args.seq,
        heads=base.heads if args.heads is None else args.heads,
        head_dim=base.head_dim if args.head_dim is None else args.head_dim,
        blocks=base.blocks if args.blocks is None else args.blocks,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    spec = resolve_spec(args)
    if spec.head_dim < 1 or spec.head_dim > 64:
        print(f"head_dim={spec.head_dim} outside fused kernel bound [1, 64]", file=sys.stderr)
        return 2
    work_dir = args.out or os.path.join("out", f"attn_layout_{args.preset}")
    result = compare_pair(spec, seed=args.seed, work_dir=work_dir)
    print_report(result)
    failed = (
        result["maxdiff_bhnd_golden"] > args.atol
        or result["maxdiff_bnhd_golden"] > args.atol
        or result["maxdiff_bhnd_bnhd"] > args.atol
    )
    if result["bhnd_hist"].get("Transpose", 0) == 0:
        print("BHND graph unexpectedly has no Transpose", file=sys.stderr)
        failed = True
    if result["bnhd_hist"].get("Transpose", 0) != 0:
        print("BNHD graph should not contain Transpose", file=sys.stderr)
        failed = True
    if result["bnhd_layouts"] != ["BNHD"] * spec.blocks:
        print(f"BNHD Attention layout attrs {result['bnhd_layouts']}", file=sys.stderr)
        failed = True
    if args.quantize:
        for layout, onnx_path in (("bhnd", result["bhnd_onnx"]), ("bnhd", result["bnhd_onnx"])):
            espdl = os.path.join(work_dir, f"attn_layout_{layout}_s8.espdl")
            print(f"quantize {layout} -> {espdl}")
            quantize_onnx(
                onnx_path,
                espdl,
                spec,
                target=args.target,
                quant_type=args.quant_type,
            )
            print(f"wrote {espdl}")
    if args.check and failed:
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
