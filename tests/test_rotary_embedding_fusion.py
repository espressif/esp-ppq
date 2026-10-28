"""Tests for RotaryEmbeddingFusionPass, EspdlRotaryEmbeddingLegalizePass, and check_rotary_embedding."""

import inspect
import os
import tempfile

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto as TP
from onnx import helper as OH

from esp_ppq import TargetPlatform, TorchExecutor
from esp_ppq.api import dispatch_graph, load_onnx_graph
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.IR import Variable
from esp_ppq.quantization.optim import (
    EspdlRotaryEmbeddingLegalizePass,
    RotaryEmbeddingFusionPass,
    check_rotary_embedding,
    halve_rope_cache,
)
from esp_ppq.quantization.optim.rotary_embedding_fusion import (
    _rope_fusion_needs_trace,
    _rope_prequant_needs_trace,
)

ROPE = 'RotaryEmbedding'
PRIMITIVE_TYPES = {'Slice', 'Neg', 'Concat', 'Mul'}


# ---------------------------------------------------------------------------
# Graph construction helpers
#
# The subgraphs are built with onnx.helper rather than torch.onnx.export:
# the variants differ only in Concat operand order, which is not reachable
# by writing different nn.Modules, and an explicit topology keeps the test
# readable as a spec of what the matcher is supposed to accept.
# ---------------------------------------------------------------------------
def _freqs(seq, half):
    """Raw [seq, half] angles — the half-width content of an ONNX cache."""
    inv = 1.0 / (10000.0 ** (np.arange(0, 2 * half, 2, dtype=np.float64) / (2 * half)))
    return np.outer(np.arange(seq, dtype=np.float64), inv)


def _rope_tables(seq, half, *, dtype=np.float32):
    """Full-width cos/sin tables for rotate_half, shape [1, 1, seq, 2*half].

    HF builds them as ``cat((freqs, freqs), -1)``, so the two halves are
    bit-identical — exactly the invariant ``halve_rope_cache`` checks.
    """
    f = _freqs(seq, half)
    cos = np.concatenate([np.cos(f), np.cos(f)], axis=-1)
    sin = np.concatenate([np.sin(f), np.sin(f)], axis=-1)
    return cos[None, None].astype(dtype), sin[None, None].astype(dtype)


def _rope_tables_interleaved(seq, half, *, dtype=np.float32):
    """Full-width tables for the GPT-J layout, built with repeat_interleave.

    Adjacent lanes carry the same frequency ([f0 f0 f1 f1 ...]) because the
    rotation pairs adjacent elements rather than the two halves.
    """
    f = _freqs(seq, half)
    cos = np.repeat(np.cos(f), 2, axis=-1)
    sin = np.repeat(np.sin(f), 2, axis=-1)
    return cos[None, None].astype(dtype), sin[None, None].astype(dtype)


def _slice_consts(name_prefix, start, end, axis=-1):
    """Constant initializers for one opset>=10 Slice."""
    return [
        OH.make_tensor(f'{name_prefix}_starts', TP.INT64, [1], [start]),
        OH.make_tensor(f'{name_prefix}_ends', TP.INT64, [1], [end]),
        OH.make_tensor(f'{name_prefix}_axes', TP.INT64, [1], [axis]),
        OH.make_tensor(f'{name_prefix}_steps', TP.INT64, [1], [1]),
    ]


def _slice_node(name_prefix, src, out):
    return OH.make_node(
        'Slice',
        [src, f'{name_prefix}_starts', f'{name_prefix}_ends', f'{name_prefix}_axes', f'{name_prefix}_steps'],
        [out],
        name=f'slice_{name_prefix}',
    )


def _save(model, tag):
    onnx.checker.check_model(model)
    path = os.path.join(tempfile.gettempdir(), f'test_rope_fusion_{tag}.onnx')
    onnx.save(model, path)
    return path


def _finish(nodes, inits, xshape, yname, tag, out_shape=None):
    graph = OH.make_graph(
        nodes,
        f'rope_{tag}',
        [OH.make_tensor_value_info('X', TP.FLOAT, xshape)],
        [OH.make_tensor_value_info(yname, TP.FLOAT, out_shape or xshape)],
        inits,
    )
    model = OH.make_model(graph, opset_imports=[OH.make_opsetid('', 17)])
    model.ir_version = 7
    return _save(model, tag)


def build_rotate_half(
    batch_size,
    head_count,
    seq_len,
    head_dim,
    *,
    order,
    tag,
    cos_sin=None,
    slice_axis=-1,
    concat_axis=-1,
):
    """7-op subgraph whose Concat operand order is given by ``order``.

    ``order`` names what sits in concat slot 0 / slot 1:
      'neg_hi_lo'  -> Concat([-x2,  x1])   canonical, RoPE(+sin)
      'hi_neg_lo'  -> Concat([ x2, -x1])   opposite convention, RoPE(-sin)
      'lo_neg_hi'  -> Concat([ x1, -x2])   NOT a rotation, must be rejected
    """
    half = head_dim // 2
    cos, sin = cos_sin if cos_sin is not None else _rope_tables(seq_len, half)

    inits = (
        _slice_consts('lo', 0, half, axis=slice_axis)
        + _slice_consts('hi', half, head_dim, axis=slice_axis)
        + [
            OH.make_tensor('cos', TP.FLOAT, cos.shape, cos.flatten().tolist()),
            OH.make_tensor('sin', TP.FLOAT, sin.shape, sin.flatten().tolist()),
        ]
    )
    nodes = [
        _slice_node('lo', 'X', 'x_lo'),
        _slice_node('hi', 'X', 'x_hi'),
    ]

    if order == 'neg_hi_lo':
        nodes += [
            OH.make_node('Neg', ['x_hi'], ['n_hi']),
            OH.make_node('Concat', ['n_hi', 'x_lo'], ['rot'], axis=concat_axis),
        ]
    elif order == 'hi_neg_lo':
        nodes += [
            OH.make_node('Neg', ['x_lo'], ['n_lo']),
            OH.make_node('Concat', ['x_hi', 'n_lo'], ['rot'], axis=concat_axis),
        ]
    elif order == 'lo_neg_hi':
        nodes += [
            OH.make_node('Neg', ['x_hi'], ['n_hi']),
            OH.make_node('Concat', ['x_lo', 'n_hi'], ['rot'], axis=concat_axis),
        ]
    else:
        raise ValueError(f'unknown order {order!r}')

    nodes += [
        OH.make_node('Mul', ['X', 'cos'], ['xcos']),
        OH.make_node('Mul', ['rot', 'sin'], ['rsin']),
        OH.make_node('Add', ['xcos', 'rsin'], ['Y']),
    ]
    return _finish(nodes, inits, [batch_size, head_count, seq_len, head_dim], 'Y', tag)


class _TorchExportHalfSplitRoPE(torch.nn.Module):
    """Canonical HuggingFace-style RoPE used to exercise the real exporter."""

    def __init__(self, seq_len, head_dim):
        super().__init__()
        cos, sin = _rope_tables(seq_len, head_dim // 2)
        self.register_buffer('cos', torch.from_numpy(cos))
        self.register_buffer('sin', torch.from_numpy(sin))

    def forward(self, x):
        split = x.shape[-1] // 2
        x1 = x[..., :split]
        # The open-ended upper slice is the regression target: both the
        # TorchScript and dynamo exporters encode its end as INT64_MAX.
        x2 = x[..., split:]
        rotate_half = torch.cat((-x2, x1), dim=-1)
        return x * self.cos + rotate_half * self.sin


class _TorchExportChunkRoPE(torch.nn.Module):
    """RoPE using ``torch.chunk(2, dim=-1)`` / ``torch.split``."""

    def __init__(self, seq_len, head_dim, *, mode='chunk', swapped=False):
        super().__init__()
        cos, sin = _rope_tables(seq_len, head_dim // 2)
        self.register_buffer('cos', torch.from_numpy(cos))
        self.register_buffer('sin', torch.from_numpy(sin))
        self.mode = mode
        self.swapped = swapped

    def forward(self, x):
        if self.mode == 'chunk':
            x1, x2 = x.chunk(2, dim=-1)
        else:
            x1, x2 = torch.split(x, x.shape[-1] // 2, dim=-1)
        rotate_half = torch.cat((x2, -x1), dim=-1) if self.swapped else torch.cat((-x2, x1), dim=-1)
        return x * self.cos + rotate_half * self.sin


class _TorchExportNativeRoPE(torch.nn.Module):
    """Export-only ONNX opset-23 path available in recent PyTorch releases."""

    def __init__(self, seq_len, head_dim):
        super().__init__()
        freq = torch.from_numpy(_freqs(seq_len, head_dim // 2).astype(np.float32))
        self.register_buffer('cos', torch.cos(freq).unsqueeze(0))
        self.register_buffer('sin', torch.sin(freq).unsqueeze(0))

    def forward(self, x):
        return torch.onnx.ops.rotary_embedding(x, self.cos, self.sin)


def _export_torch_half_split_rope(batch_size, head_count, seq_len, head_dim, *, dynamo):
    if dynamo:
        if 'dynamo' not in inspect.signature(torch.onnx.export).parameters:
            pytest.skip('installed torch.onnx.export has no dynamo exporter')
        pytest.importorskip('onnxscript')

    fd, path = tempfile.mkstemp(prefix='test_rope_torch_export_', suffix='.onnx')
    os.close(fd)
    kwargs = dict(
        opset_version=18,
        input_names=['X'],
        output_names=['Y'],
        do_constant_folding=True,
    )
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        kwargs['dynamo'] = dynamo
    torch.onnx.export(
        _TorchExportHalfSplitRoPE(seq_len, head_dim).eval(),
        (torch.randn(batch_size, head_count, seq_len, head_dim),),
        path,
        **kwargs,
    )
    return path


def _export_torch_native_rope(batch_size, head_count, seq_len, head_dim):
    if 'dynamo' not in inspect.signature(torch.onnx.export).parameters:
        pytest.skip('installed torch.onnx.export has no dynamo exporter')
    if not hasattr(torch.onnx, 'ops') or not hasattr(torch.onnx.ops, 'rotary_embedding'):
        pytest.skip('installed PyTorch has no torch.onnx.ops.rotary_embedding')
    pytest.importorskip('onnxscript')

    fd, path = tempfile.mkstemp(prefix='test_rope_torch_native_', suffix='.onnx')
    os.close(fd)
    torch.onnx.export(
        _TorchExportNativeRoPE(seq_len, head_dim).eval(),
        (torch.randn(batch_size, head_count, seq_len, head_dim),),
        path,
        opset_version=23,
        dynamo=True,
        input_names=['X'],
        output_names=['Y'],
    )
    return path


def build_partial(batch_size, head_count, seq_len, head_dim, rotary_dim, tag, *, tail_first=False):
    """Partial RoPE: outer Slice+Concat wrapping the canonical 7-op core.

    ``tail_first=True`` swaps the outer Concat operand order to
    ``Concat([tail, rotated])``. This produces a DIFFERENT output layout
    (``[X[rotary_dim:], rotated(X[:rotary_dim])]``) that no ONNX RotaryEmbedding attribute
    can express — the matcher must reject it, not fuse it.
    """
    half = rotary_dim // 2
    cos, sin = _rope_tables(seq_len, half)

    inits = (
        _slice_consts('head', 0, rotary_dim)
        + _slice_consts('tail', rotary_dim, head_dim)
        + _slice_consts('lo', 0, half)
        + _slice_consts('hi', half, rotary_dim)
        + [
            OH.make_tensor('cos', TP.FLOAT, cos.shape, cos.flatten().tolist()),
            OH.make_tensor('sin', TP.FLOAT, sin.shape, sin.flatten().tolist()),
        ]
    )
    nodes = [
        _slice_node('head', 'X', 'x_rot'),
        _slice_node('tail', 'X', 'x_pass'),
        _slice_node('lo', 'x_rot', 'x_lo'),
        _slice_node('hi', 'x_rot', 'x_hi'),
        OH.make_node('Neg', ['x_hi'], ['n_hi']),
        OH.make_node('Concat', ['n_hi', 'x_lo'], ['rot'], axis=-1),
        OH.make_node('Mul', ['x_rot', 'cos'], ['xcos']),
        OH.make_node('Mul', ['rot', 'sin'], ['rsin']),
        OH.make_node('Add', ['xcos', 'rsin'], ['rotated']),
        (
            OH.make_node('Concat', ['x_pass', 'rotated'], ['Y'], axis=-1)
            if tail_first
            else OH.make_node('Concat', ['rotated', 'x_pass'], ['Y'], axis=-1)
        ),
    ]
    return _finish(nodes, inits, [batch_size, head_count, seq_len, head_dim], 'Y', tag)


def build_interleaved(batch_size, head_count, seq_len, head_dim, tag, cos_sin=None, *, even_first=False):
    """GPT-J interleaved: Reshape -> Slice/Slice -> Neg -> Concat -> Reshape.

    The rotation pairs adjacent elements, so ``rot`` ends up as
    ``[-x1, x0, -x3, x2, ...]`` and the tables must be repeat_interleaved.

    ``even_first=True`` swaps the concat operand order to
    ``Concat([x_even, -x_odd])``, which reshapes back to a different pair
    layout (pair (2k, 2k+1) = (x_even, -x_odd) instead of (-x_odd, x_even))
    — not the interleaved RoPE variant. The matcher must reject it.
    """
    half = head_dim // 2
    cos, sin = cos_sin if cos_sin is not None else _rope_tables_interleaved(seq_len, half)

    inits = (
        _slice_consts('even', 0, 1)
        + _slice_consts('odd', 1, 2)
        + [
            OH.make_tensor('shape_pair', TP.INT64, [5], [batch_size, head_count, seq_len, half, 2]),
            OH.make_tensor('shape_flat', TP.INT64, [4], [batch_size, head_count, seq_len, head_dim]),
            OH.make_tensor('cos', TP.FLOAT, cos.shape, cos.flatten().tolist()),
            OH.make_tensor('sin', TP.FLOAT, sin.shape, sin.flatten().tolist()),
        ]
    )
    concat_inputs = ['x_even', 'n_odd'] if even_first else ['n_odd', 'x_even']
    nodes = [
        OH.make_node('Reshape', ['X', 'shape_pair'], ['x_pair']),
        _slice_node('even', 'x_pair', 'x_even'),
        _slice_node('odd', 'x_pair', 'x_odd'),
        OH.make_node('Neg', ['x_odd'], ['n_odd']),
        OH.make_node('Concat', concat_inputs, ['rot_pair'], axis=-1),
        OH.make_node('Reshape', ['rot_pair', 'shape_flat'], ['rot']),
        OH.make_node('Mul', ['X', 'cos'], ['xcos']),
        OH.make_node('Mul', ['rot', 'sin'], ['rsin']),
        OH.make_node('Add', ['xcos', 'rsin'], ['Y']),
    ]
    return _finish(nodes, inits, [batch_size, head_count, seq_len, head_dim], 'Y', tag)


def build_not_rope(batch_size, head_count, seq_len, head_dim, tag):
    """A 2-input Add with two parameter Muls but no rotate_half anywhere."""
    inits = [
        OH.make_tensor('w0', TP.FLOAT, [1], [1.5]),
        OH.make_tensor('w1', TP.FLOAT, [1], [2.5]),
    ]
    nodes = [
        OH.make_node('Mul', ['X', 'w0'], ['a']),
        OH.make_node('Mul', ['X', 'w1'], ['b']),
        OH.make_node('Add', ['a', 'b'], ['Y']),
    ]
    return _finish(nodes, inits, [batch_size, head_count, seq_len, head_dim], 'Y', tag)


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------
def _load(path):
    graph = load_onnx_graph(path)
    dispatch_graph(graph, platform=TargetPlatform.ESPDL_INT8, dispatcher='conservative')
    return graph


def _run(graph, x):
    return TorchExecutor(graph=graph, device='cpu').forward(inputs=x)[0]


def _trace(graph, x):
    executor = TorchExecutor(graph=graph, device='cpu')
    executor.tracing_operation_meta(inputs=[x])
    return executor


def _op_types(graph):
    return [op.type for op in graph.operations.values()]


def _fuse_and_compare(path, xshape, *, expect_fused=True, seed=0, atol=1e-5):
    """Load, run, fuse, re-run. Returns (graph, pass_, max_abs_diff).

    The before/after comparison is the point: a matcher that fires on a
    pattern it should not would still leave a shape-correct graph.
    """
    graph = _load(path)
    torch.manual_seed(seed)
    x = torch.randn(*xshape)

    ref = _run(graph, x)
    pass_ = RotaryEmbeddingFusionPass(verbose=False)
    executor = _trace(graph, x)
    pass_.optimize(graph, dataloader=None, executor=executor, inputs=[x])

    fused_present = ROPE in _op_types(graph)
    assert fused_present == expect_fused, f'expected fused={expect_fused}, got ops {_op_types(graph)}'

    out = _run(graph, x)
    diff = (ref - out).abs().max().item()
    assert diff < atol, f'numerics changed across fusion: max|diff| = {diff:.3e}'
    return graph, pass_, diff


# ---------------------------------------------------------------------------
# Positive cases — must fuse AND stay numerically identical
# ---------------------------------------------------------------------------
def test_canonical_fusion():
    """Concat([-x2, x1], -1) — the HF-LLaMA / GPT-NeoX form."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='canonical')
    graph, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))

    assert pass_.fused_count == 1
    assert pass_.fused_swapped_count == 0
    # every primitive of the subgraph is gone
    assert not (PRIMITIVE_TYPES & set(_op_types(graph))), _op_types(graph)

    rope = next(op for op in graph.operations.values() if op.type == ROPE)
    assert rope.attributes.get('interleaved', 0) in (0, None)
    assert int(rope.inputs[1].value.shape[-1]) == head_dim // 2, 'cache must be half-width'
    assert rope.platform == TargetPlatform.FP32

    os.unlink(path)
    print(f'PASS: test_canonical_fusion (max|diff| = {diff:.2e})')


def test_mixed_platform_pattern_is_not_fused():
    shape = (1, 2, 8, 16)
    path = build_rotate_half(*shape, order='neg_hi_lo', tag='mixed_platform')
    try:
        graph = _load(path)
        next(op for op in graph.operations.values() if op.type == 'Neg').platform = TargetPlatform.UNSPECIFIED
        x = torch.randn(*shape)
        before = _run(graph, x)
        pass_ = RotaryEmbeddingFusionPass()
        pass_.optimize(graph, executor=_trace(graph, x), inputs=[x])
        after = _run(graph, x)
        assert pass_.fused_count == 0
        assert ROPE not in _op_types(graph)
        assert torch.equal(after, before)
    finally:
        os.unlink(path)


def test_slice_axis_normalization_and_boundaries():
    """Only a one-axis, unit-step split of the traced final dimension matches."""
    shape = (1, 2, 8, 16)

    # PyTorch may encode Slice.axes=[3] while Concat keeps axis=-1.
    path_pos = build_rotate_half(
        *shape,
        order='neg_hi_lo',
        tag='positive_last_axis',
        slice_axis=3,
    )
    try:
        _, pass_, _ = _fuse_and_compare(path_pos, shape)
        assert pass_.fused_count == 1
    finally:
        os.unlink(path_pos)

    # A single positive axis is not automatically equivalent to -1. This
    # malformed RoPE-looking graph used to be a false positive.
    path_wrong = build_rotate_half(
        *shape,
        order='neg_hi_lo',
        tag='wrong_positive_axis',
        slice_axis=2,
    )
    try:
        graph = _load(path_wrong)
        graph.inputs['X'].shape = list(shape)
        pass_ = RotaryEmbeddingFusionPass()
        pass_.optimize(graph)
        assert pass_.fused_count == 0
        assert ROPE not in _op_types(graph)
    finally:
        os.unlink(path_wrong)

    # Slice.end larger than D has the same positive-step clipping semantics
    # as PyTorch's INT64_MAX open tail.
    for end in (shape[-1] + 7, np.iinfo(np.int32).max):
        path = build_rotate_half(*shape, order='neg_hi_lo', tag=f'clipped_end_{end}')
        try:
            model = onnx.load(path)
            init = next(t for t in model.graph.initializer if t.name == 'hi_ends')
            init.CopyFrom(onnx.numpy_helper.from_array(np.asarray([end], dtype=np.int64), init.name))
            onnx.save(model, path)
            _, pass_, _ = _fuse_and_compare(path, shape)
            assert pass_.fused_count == 1
        finally:
            os.unlink(path)


def test_multi_axis_slice_is_not_absorbed():
    """Even a currently full-range extra axis is kept rather than guessed away."""
    shape = (1, 2, 8, 16)
    path = build_rotate_half(*shape, order='neg_hi_lo', tag='multi_axis_slice')
    try:
        model = onnx.load(path)
        values = {
            'lo_starts': [0, 0],
            'lo_ends': [shape[2], shape[3] // 2],
            'lo_axes': [2, 3],
            'lo_steps': [1, 1],
            'hi_starts': [0, shape[3] // 2],
            'hi_ends': [shape[2], shape[3]],
            'hi_axes': [2, 3],
            'hi_steps': [1, 1],
        }
        for init in model.graph.initializer:
            if init.name in values:
                init.CopyFrom(onnx.numpy_helper.from_array(np.asarray(values[init.name], dtype=np.int64), init.name))
        onnx.checker.check_model(model)
        onnx.save(model, path)
        _, pass_, _ = _fuse_and_compare(path, shape, expect_fused=False)
        assert pass_.fused_count == 0
    finally:
        os.unlink(path)


@pytest.mark.parametrize(
    ('updates', 'tag'),
    [
        ({'hi_steps': [2]}, 'non_unit_step'),
        ({'hi_ends': [-1]}, 'negative_end'),
        ({'lo_ends': [7], 'hi_starts': [7]}, 'unbalanced_halves'),
    ],
)
def test_invalid_slice_boundaries_are_not_fused(updates, tag):
    shape = (1, 2, 8, 16)
    path = build_rotate_half(*shape, order='neg_hi_lo', tag=tag)
    try:
        model = onnx.load(path)
        for init in model.graph.initializer:
            if init.name in updates:
                init.CopyFrom(onnx.numpy_helper.from_array(np.asarray(updates[init.name], dtype=np.int64), init.name))
        onnx.save(model, path)
        graph = _load(path)
        graph.inputs['X'].shape = list(shape)
        pass_ = RotaryEmbeddingFusionPass()
        pass_.optimize(graph)
        assert pass_.fused_count == 0
        assert ROPE not in _op_types(graph)
    finally:
        os.unlink(path)


def test_per_batch_static_cache_fusion():
    """[B,1,L,D] is unambiguously batch-specific once X is traced."""
    shape = (2, 3, 8, 16)
    cos, sin = _rope_tables(shape[2], shape[3] // 2)
    cos = np.repeat(cos, shape[0], axis=0)
    sin = np.repeat(sin, shape[0], axis=0)
    path = build_rotate_half(
        *shape,
        order='neg_hi_lo',
        tag='per_batch_cache',
        cos_sin=(cos, sin),
    )
    try:
        graph, pass_, _ = _fuse_and_compare(path, shape)
        rope = next(op for op in graph.operations.values() if op.type == ROPE)
        assert pass_.fused_count == 1
        assert tuple(rope.inputs[1].value.shape) == (shape[0], shape[2], shape[3] // 2)
    finally:
        os.unlink(path)


@pytest.mark.parametrize('dynamo', [False, True], ids=['legacy', 'dynamo'])
def test_real_pytorch_export_open_ended_slice_fusion(dynamo):
    """Fuse the graph emitted from the canonical PyTorch ``x[..., split:]``.

    PyTorch writes the upper Slice end as INT64_MAX, which ONNX clips to D.
    Hand-authored tests historically used an explicit D and therefore missed
    this exporter/matcher compatibility boundary.
    """
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = _export_torch_half_split_rope(batch_size, head_count, seq_len, head_dim, dynamo=dynamo)
    try:
        graph, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
        assert pass_.fused_count == 1
        assert _op_types(graph).count(ROPE) == 1
        assert not (PRIMITIVE_TYPES & set(_op_types(graph))), _op_types(graph)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    print(
        f'PASS: test_real_pytorch_export_open_ended_slice_fusion['
        f'{"dynamo" if dynamo else "legacy"}] (max|diff| = {diff:.2e})'
    )


@pytest.mark.parametrize(
    ('mode', 'dynamo', 'swapped'),
    [
        ('chunk', False, False),
        ('chunk', True, False),
        ('chunk', True, True),
        ('split', False, False),
        ('split', True, False),
    ],
)
def test_real_pytorch_export_split_rotate_half_fusion(mode, dynamo, swapped):
    """Fuse ``Split``-based ``rotate_half`` emitted by ``torch.chunk`` / ``torch.split``."""
    if dynamo:
        if 'dynamo' not in inspect.signature(torch.onnx.export).parameters:
            pytest.skip('installed torch.onnx.export has no dynamo exporter')
        pytest.importorskip('onnxscript')

    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    fd, path = tempfile.mkstemp(prefix=f'test_rope_{mode}_', suffix='.onnx')
    os.close(fd)
    try:
        kwargs = dict(
            opset_version=18,
            input_names=['X'],
            output_names=['Y'],
            do_constant_folding=True,
        )
        if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
            kwargs['dynamo'] = dynamo
        torch.onnx.export(
            _TorchExportChunkRoPE(seq_len, head_dim, mode=mode, swapped=swapped).eval(),
            (torch.randn(batch_size, head_count, seq_len, head_dim),),
            path,
            **kwargs,
        )
        if mode == 'chunk' and not dynamo:
            from onnxsim import simplify

            simplified, ok = simplify(onnx.load(path))
            assert ok
            onnx.save(simplified, path)
        graph, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
        assert pass_.fused_count == 1
        assert pass_.fused_swapped_count == (1 if swapped else 0)
        assert _op_types(graph) == [ROPE]
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.parametrize(
    ('order', 'splits', 'axis'),
    [
        ('lo_neg_hi', [8, 8], -1),
        ('neg_lo_hi', [8, 8], -1),
        ('neg_hi_lo', [6, 10], -1),
        ('neg_hi_lo', [8, 8], 2),
    ],
)
def test_invalid_split_rotate_half_is_not_fused(order, splits, axis):
    """Reject mirrored Split output orders, unequal Split sizes, or non-last Split axis."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    cos, sin = _rope_tables(seq_len, head_dim // 2)
    inits = [
        OH.make_tensor('split_sizes', TP.INT64, [len(splits)], splits),
        OH.make_tensor('cos', TP.FLOAT, cos.shape, cos.flatten().tolist()),
        OH.make_tensor('sin', TP.FLOAT, sin.shape, sin.flatten().tolist()),
    ]
    nodes = [OH.make_node('Split', ['X', 'split_sizes'], ['x_lo', 'x_hi'], axis=axis)]
    if order == 'neg_hi_lo':
        nodes += [
            OH.make_node('Neg', ['x_hi'], ['n_hi']),
            OH.make_node('Concat', ['n_hi', 'x_lo'], ['rot'], axis=-1),
        ]
    elif order == 'lo_neg_hi':
        nodes += [
            OH.make_node('Neg', ['x_hi'], ['n_hi']),
            OH.make_node('Concat', ['x_lo', 'n_hi'], ['rot'], axis=-1),
        ]
    elif order == 'neg_lo_hi':
        nodes += [
            OH.make_node('Neg', ['x_lo'], ['n_lo']),
            OH.make_node('Concat', ['n_lo', 'x_hi'], ['rot'], axis=-1),
        ]
    nodes += [
        OH.make_node('Mul', ['X', 'cos'], ['xcos']),
        OH.make_node('Mul', ['rot', 'sin'], ['rsin']),
        OH.make_node('Add', ['xcos', 'rsin'], ['Y']),
    ]
    path = _finish(nodes, inits, [batch_size, head_count, seq_len, head_dim], 'Y', f'bad_split_{order}_{axis}')
    try:
        graph = _load(path)
        graph.inputs['X'].shape = [batch_size, head_count, seq_len, head_dim]
        pass_ = RotaryEmbeddingFusionPass()
        pass_.optimize(graph)
        assert pass_.fused_count == 0
        assert ROPE not in _op_types(graph)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_real_pytorch_native_rotary_embedding_export():
    """The recommended dynamo/opset-23 path emits one backend-ready node."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = _export_torch_native_rope(batch_size, head_count, seq_len, head_dim)
    try:
        graph, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
        assert pass_.fused_count == 0
        assert _op_types(graph) == [ROPE]
        errors, _ = check_rotary_embedding(graph, strict=True)
        assert not errors
    finally:
        if os.path.exists(path):
            os.unlink(path)

    print(f'PASS: test_real_pytorch_native_rotary_embedding_export (max|diff| = {diff:.2e})')


def test_negated_sin_convention_fusion():
    """Concat([x2, -x1], -1) — same rotation, opposite sin sign.

    Must fuse with a freshly negated sin cache. Regression guard: an earlier
    revision matched Concat([x1, -x2]) here instead, which is not a rotation
    at all, and "fixed" it by negating sin — silently wrong numbers.
    """
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='hi_neg_lo', tag='negsin')
    graph, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))

    assert pass_.fused_count == 1
    assert pass_.fused_swapped_count == 1, 'should be counted as the swapped variant'

    # The fused sin cache must be the negation of the canonical one, and the
    # source table must not have been mutated in place.
    rope = next(op for op in graph.operations.values() if op.type == ROPE)
    _, sin_full = _rope_tables(seq_len, head_dim // 2)
    expected_sin, why = halve_rope_cache(sin_full)
    assert expected_sin is not None, why
    got_sin = rope.inputs[2].value
    got_sin = got_sin.numpy() if hasattr(got_sin, 'numpy') else np.asarray(got_sin)
    assert np.allclose(got_sin, -np.asarray(expected_sin), atol=1e-6), (
        'sin cache should be negated for the Concat([x2, -x1]) convention'
    )

    os.unlink(path)
    print(f'PASS: test_negated_sin_convention_fusion (max|diff| = {diff:.2e})')


def test_partial_rope_wrapper_is_not_absorbed():
    """Partial RoPE is outside the vision subset, so the outer Slice+Concant
    wrapper must survive.

    The inner 7-op core rotates all of ``x_rot`` (width ``rotary_dim``), which IS a valid
    full-rotation RoPE, so it still fuses — just on ``x_rot`` rather than on
    X, and without a ``rotary_embedding_dim`` attribute. The tail passthrough
    stays as graph ops. Both operand orders behave the same way.
    """
    batch_size, head_count, seq_len, head_dim, rotary_dim = 1, 2, 8, 32, 16
    for tag, tail_first in [('partial', False), ('partial_swap', True)]:
        path = build_partial(batch_size, head_count, seq_len, head_dim, rotary_dim, tag=tag, tail_first=tail_first)
        graph, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))

        remaining = _op_types(graph)
        assert 'Concat' in remaining and 'Slice' in remaining, f'outer wrapper must survive intact, got {remaining}'
        assert 'RotaryEmbedding' in remaining, remaining

        rope = next(op for op in graph.operations.values() if op.type == ROPE)
        assert int(rope.attributes.get('rotary_embedding_dim', 0) or 0) == 0, (
            'the fused node must consume x_rot directly and rotate all of it'
        )
        assert int(rope.inputs[1].value.shape[-1]) == rotary_dim // 2

        # The fused node is a legal vision-subset node.
        errors, _ = check_rotary_embedding(graph, strict=True)
        assert not errors, errors

        os.unlink(path)
        print(f'PASS: test_partial_rope_wrapper_is_not_absorbed[{tag}] (max|diff| = {diff:.2e})')


def test_interleaved_graphs_are_never_fused():
    """The GPT-J adjacent-pair layout is outside the vision subset.

    No interleaved graph shape may fuse — not the canonical
    ``Concat([-x_odd, x_even])``, not the mirrored order, and not one
    carrying a rotate_half-style table. Each must be left as the primitive
    subgraph, which stays numerically correct.
    """
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    cases = [
        ('interleaved', dict()),
        ('il_reverse', dict(even_first=True)),
        ('il_badtable', dict(cos_sin=_rope_tables(seq_len, head_dim // 2))),
    ]
    for tag, kwargs in cases:
        path = build_interleaved(batch_size, head_count, seq_len, head_dim, tag=tag, **kwargs)
        _, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim), expect_fused=False)
        assert pass_.fused_count == 0

        os.unlink(path)
        print(f'PASS: test_interleaved_graphs_are_never_fused[{tag}] (max|diff| = {diff:.2e})')


# ---------------------------------------------------------------------------
# Negative cases — must NOT fuse, must NOT raise, must NOT change numerics
# ---------------------------------------------------------------------------
def test_mirrored_concat_is_rejected():
    """Concat([x1, -x2], -1) is an elementwise scaling, not a rotation.

    ``X*cos + concat(x1, -x2)*sin`` = ``(x1*(c+s), x2*(c-s))``. No sin-sign
    flip makes that RoPE, so the matcher must leave the subgraph alone.
    """
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='lo_neg_hi', tag='mirrored')
    graph, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim), expect_fused=False)

    assert pass_.fused_count == 0
    assert 'Add' in _op_types(graph), 'primitive subgraph must survive intact'

    os.unlink(path)
    print(f'PASS: test_mirrored_concat_is_rejected (max|diff| = {diff:.2e})')


def test_per_head_cache_is_rejected():
    """A [1, H, L, D] table cannot be expressed as an ONNX cache."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    half = head_dim // 2
    cos, sin = _rope_tables(seq_len, half)
    # broadcast the table across the head axis -> leading dims [1, H]
    cos = np.repeat(cos, head_count, axis=1)
    sin = np.repeat(sin, head_count, axis=1)

    path = build_rotate_half(
        batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='perhead', cos_sin=(cos, sin)
    )
    _, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim), expect_fused=False)
    assert pass_.fused_count == 0

    os.unlink(path)
    print(f'PASS: test_per_head_cache_is_rejected (max|diff| = {diff:.2e})')


def test_unequal_halves_cache_is_rejected():
    """Halves that are not bit-identical break the half-width equivalence."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    half = head_dim // 2
    cos, sin = _rope_tables(seq_len, half)
    sin = sin.copy()
    sin[..., half:] += np.float32(1e-3)  # break the duplication

    path = build_rotate_half(
        batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='badhalves', cos_sin=(cos, sin)
    )
    _, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim), expect_fused=False)
    assert pass_.fused_count == 0

    os.unlink(path)
    print(f'PASS: test_unequal_halves_cache_is_rejected (max|diff| = {diff:.2e})')


def test_unrelated_add_is_not_matched():
    """A non-RoPE Add with two parameter Muls must not be misidentified."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = build_not_rope(batch_size, head_count, seq_len, head_dim, tag='notrope')
    _, pass_, diff = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim), expect_fused=False)
    assert pass_.fused_count == 0

    os.unlink(path)
    print(f'PASS: test_unrelated_add_is_not_matched (max|diff| = {diff:.2e})')


# ---------------------------------------------------------------------------
# halve_rope_cache contract
# ---------------------------------------------------------------------------
def test_halve_rope_cache_contract():
    seq, half = 8, 6
    cos, _ = _rope_tables(seq, half)  # [1, 1, seq, 2*half]

    cache, why = halve_rope_cache(cos)
    assert cache is not None, why
    assert cache.shape == (1, seq, half)
    assert np.allclose(cache[0], cos[0, 0, :, :half])

    # rank-2 [L, D] and rank-3 [1, L, D] are both accepted
    for table in (cos[0, 0], cos[0]):
        c, why = halve_rope_cache(table)
        assert c is not None, why
        assert c.shape == (1, seq, half)

    # single-token table needs a seq_len hint to materialize
    one = cos[:, :, :1, :]
    assert halve_rope_cache(one)[0] is None, 'no seq_len -> must refuse'
    c, why = halve_rope_cache(one, seq_len=seq)
    assert c is not None and c.shape == (1, seq, half), why

    # odd width has no midpoint
    assert halve_rope_cache(np.zeros((1, 4, 5), dtype=np.float32))[0] is None

    # A rank-4 [B,1,L,D] table is batch-specific only when X batch proves it.
    batched = np.repeat(cos, 2, axis=0)
    assert halve_rope_cache(batched)[0] is None
    c, why = halve_rope_cache(batched, batch_size=2)
    assert c is not None and c.shape == (2, seq, half), why
    assert halve_rope_cache(batched, batch_size=3)[0] is None

    # Rank-3 non-singleton leading axes are heads under 4D broadcasting;
    # rank>4 would change the Mul/Add output rank. Both stay unfused.
    assert halve_rope_cache(np.repeat(cos[0], 2, axis=0), batch_size=2)[0] is None
    assert halve_rope_cache(cos[None], batch_size=1)[0] is None

    # torch tensors round-trip as torch tensors
    t, why = halve_rope_cache(torch.from_numpy(cos))
    assert isinstance(t, torch.Tensor) and t.shape == (1, seq, half), why

    print('PASS: test_halve_rope_cache_contract')


# ---------------------------------------------------------------------------
# Backend support check (module-level function, not a Pass)
# ---------------------------------------------------------------------------
def test_support_check_accepts_fused_graph():
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='supportok')
    graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))

    errors, _ = check_rotary_embedding(graph, strict=True)
    assert not errors

    os.unlink(path)
    print('PASS: test_support_check_accepts_fused_graph')


def test_support_check_simd_hints_vision_shapes():
    """Vision dimensions divisible by the mixed 8-pair width hit SIMD."""
    batch_size, head_count, seq_len = 1, 2, 8

    # 1. Standard and mixed-8 dimensions are SIMD-eligible.
    for tag, head_dim in [
        ('vit_d32', 32),
        ('vit_d48', 48),
        ('vit_d64', 64),
        ('vit_d80', 80),
        ('vit_d96', 96),
    ]:
        path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag=f'simd_{tag}')
        graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
        errors, hints = check_rotary_embedding(graph, strict=True)
        assert not errors, errors
        assert not hints, f'D={head_dim} should be SIMD-eligible, got hints: {hints}'
        os.unlink(path)

    # 2. half=20 misses both mixed widths, but scalar fallback remains legal.
    path = build_rotate_half(batch_size, head_count, seq_len, 40, order='neg_hi_lo', tag='simd_d40')
    graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, 40))
    errors, hints = check_rotary_embedding(graph, strict=True)
    assert not errors, 'D=40 must pass the support check (scalar is legal)'
    assert len(hints) == 1
    assert 'half=20' in hints[0]
    os.unlink(path)

    print('PASS: test_support_check_simd_hints_vision_shapes')


def _mock_quantize_caches(op, dtype):
    """Cast this fused RoPE op's cos/sin cache variables to ``dtype`` in
    place, simulating a post-calibration state. check_rotary_embedding
    reads Variable.value dtype directly, so this is the smallest hook
    that flips the SIMD variant selector without re-running calibration."""
    for slot in (1, 2):
        v = op.inputs[slot]
        arr = v.value
        if arr is None:
            continue
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.asarray(arr))
        v.value = arr.to(dtype)


def test_support_check_int16_cache_hints_track_kernel_gates():
    """int16 caches now have two SIMD paths on P4:
       * int16×3  — needs half % 8  == 0
       * mixed    — needs half % 8  == 0 (only relevant with int8 X)
    int8 caches still hit only the int8×3 gate."""
    batch_size, head_count, seq_len = 1, 2, 8

    # 1. int16 caches, half % 8 == 0 → at least one variant is eligible.
    for tag, head_dim in [('int16_d48', 48), ('int16_d64', 64), ('int16_d80', 80), ('int16_d128', 128)]:
        path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag=tag)
        graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
        rope = next(op for op in graph.operations.values() if op.type == ROPE)
        _mock_quantize_caches(rope, torch.int16)

        errors, hints = check_rotary_embedding(graph, strict=True)
        assert not errors, errors
        assert not hints, f'D={head_dim} with int16 caches should hit SIMD on both variants, got hints: {hints}'
        os.unlink(path)

    # 2. int16 caches whose half dimension misses the 8-pair width fall back.
    for tag, head_dim in [('int16_d40', 40), ('int16_d56', 56)]:
        path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag=tag)
        graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
        rope = next(op for op in graph.operations.values() if op.type == ROPE)
        _mock_quantize_caches(rope, torch.int16)

        errors, hints = check_rotary_embedding(graph, strict=True)
        assert not errors, errors
        assert len(hints) == 1, hints
        assert 'int16×3/mixed SIMD gate' in hints[0], hints[0]
        os.unlink(path)

    # 3. int8 caches with half=24 / half=40 → int8×3 gate hint, no
    #    mention of int16 (int8 cache never routes to an int16 kernel).
    for tag, head_dim in [('int8_d48', 48), ('int8_d80', 80)]:
        path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag=tag)
        graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
        rope = next(op for op in graph.operations.values() if op.type == ROPE)
        _mock_quantize_caches(rope, torch.int8)

        errors, hints = check_rotary_embedding(graph, strict=True)
        assert not errors
        assert len(hints) == 1, hints
        assert 'int8×3 SIMD gate' in hints[0], hints[0]
        assert 'int16' not in hints[0], f'int8-cache hint should not mention int16 kernels; got {hints[0]!r}'
        os.unlink(path)

    # 4. int8 caches with half=32 (D=64) → SIMD-eligible, no hint.
    path = build_rotate_half(batch_size, head_count, seq_len, 64, order='neg_hi_lo', tag='int8_d64')
    graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, 64))
    rope = next(op for op in graph.operations.values() if op.type == ROPE)
    _mock_quantize_caches(rope, torch.int8)

    errors, hints = check_rotary_embedding(graph, strict=True)
    assert not errors and not hints, f'int8 caches with half=32 should hit SIMD, got hints={hints}'
    os.unlink(path)

    print('PASS: test_support_check_int16_cache_hints_track_kernel_gates')


def test_support_check_flags_bad_cache():
    """A cache whose last dim contradicts rotary_embedding_dim must be caught."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='supportbad')
    graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))

    rope = next(op for op in graph.operations.values() if op.type == ROPE)
    # half-width is D/2 = 8; claim a rotated width that cannot match it
    rope.attributes['rotary_embedding_dim'] = 4

    try:
        check_rotary_embedding(graph, strict=True)
        raise AssertionError('strict mode should have raised on the bad cache width')
    except ValueError as e:
        assert 'RotaryEmbedding' in str(e)

    # warn mode records the same problem without raising
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert errors, 'warn mode should still collect the problem'

    os.unlink(path)
    print('PASS: test_support_check_flags_bad_cache')


def test_support_check_rejects_out_of_subset_configs():
    """A hand-authored opset-23 node asking for anything the device kernel
    does not implement must be a hard error, not a soft SIMD hint.

    These four configurations are legal ONNX but outside the ESP-DL vision
    subset, and the kernel rejects each of them at inference time. Catching
    them during quantization turns a silent runtime failure into a build
    error.
    """
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16

    def _fresh_rope():
        """A fused, fully-supported node. PPQ leaves activation shapes unset,
        so X's shape is filled in here — several subset rules compare an
        attribute against head_dim (D) and are simply undecidable without it."""
        path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='subset')
        graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
        os.unlink(path)
        rope = next(op for op in graph.operations.values() if op.type == ROPE)
        rope.inputs[0].shape = [batch_size, head_count, seq_len, head_dim]
        return graph, rope

    # Baseline: untouched, this node passes.
    graph, rope = _fresh_rope()
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert not errors, errors

    # 1. GPT-J interleaved layout.
    graph, rope = _fresh_rope()
    rope.attributes['interleaved'] = 1
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any('interleaved' in e for e in errors), errors

    # 2. Partial rotation: rotate only the front half of each head. The
    #    caches shrink to match, so the only complaint must be about
    #    partial rotation itself.
    graph, rope = _fresh_rope()
    rope.attributes['rotary_embedding_dim'] = head_dim // 2
    for slot in (1, 2):
        rope.inputs[slot].value = rope.inputs[slot].value[..., : head_dim // 4]
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert len(errors) == 1 and 'partial rotation' in errors[0], errors

    # 3. Full rotation spelled out explicitly is legal, not partial.
    graph, rope = _fresh_rope()
    rope.attributes['rotary_embedding_dim'] = head_dim
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert not errors, errors

    # 4. 3D X [B, L, H*D].
    graph, rope = _fresh_rope()
    rope.inputs[0].shape = [batch_size, seq_len, head_count * head_dim]
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any('4D' in e for e in errors), errors

    # 5. The spec's optional 4th input (position_ids).
    graph, rope = _fresh_rope()
    pos = Variable(
        name='position_ids', value=np.zeros((batch_size, seq_len), dtype=np.int64), is_parameter=True, dest_ops=[rope]
    )
    graph.append_variable(pos)
    rope.inputs.append(pos)
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any('position_ids' in e for e in errors), errors

    print('PASS: test_support_check_rejects_out_of_subset_configs')


def test_fusion_and_legalize_passes_are_separate():
    """Fusion only rewrites; the always-on legalize pass owns validation."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='fusion_off')
    graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))

    rope = next(op for op in graph.operations.values() if op.type == ROPE)
    rope.attributes['rotary_embedding_dim'] = 4  # incompatible with cache last dim

    # A second fusion run neither validates nor legalizes the existing node.
    fusion_pass = RotaryEmbeddingFusionPass()
    fusion_pass.optimize(graph, dataloader=None, executor=None)
    assert rope.type == ROPE
    assert not hasattr(fusion_pass, 'strict')
    assert not hasattr(fusion_pass, 'errors')

    legalize_pass = EspdlRotaryEmbeddingLegalizePass(strict=True)
    try:
        legalize_pass.optimize(graph, dataloader=None, executor=None)
    except ValueError as e:
        assert 'RotaryEmbedding' in str(e)
    else:
        raise AssertionError('LegalizePass must reject an unsupported native node')

    # Warn mode still collects the same backend error without raising.
    legalize_pass = EspdlRotaryEmbeddingLegalizePass(strict=False)
    legalize_pass.optimize(graph, dataloader=None, executor=None)
    assert legalize_pass.errors, legalize_pass.errors

    os.unlink(path)
    print('PASS: test_fusion_and_legalize_passes_are_separate')


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------
def test_pass_is_auto_attached():
    """Fusion is optional; LegalizePass is always attached after it."""
    from esp_ppq.executor import TorchExecutor as _Exec
    from esp_ppq.quantization.quantizer import EspdlQuantizer

    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='pipeline')
    graph = _load(path)

    setting = QuantizationSettingFactory.espdl_setting()
    assert setting.rotary_embedding_fusion is True
    assert setting.rotary_embedding_fusion_setting.strict is True

    quantizer = EspdlQuantizer(graph=graph)

    pipeline = list(quantizer.build_prequant_pipeline(setting, executor=_Exec(graph=graph, device='cpu')))
    fusion = [p for p in pipeline if isinstance(p, RotaryEmbeddingFusionPass)]
    legalize = [p for p in pipeline if isinstance(p, EspdlRotaryEmbeddingLegalizePass)]
    assert len(fusion) == 1
    assert len(legalize) == 1
    assert pipeline.index(fusion[0]) < pipeline.index(legalize[0])
    assert not hasattr(fusion[0], 'strict')
    assert legalize[0].strict is True

    # Disabling fusion removes only FusionPass; backend legalization remains.
    setting.rotary_embedding_fusion = False
    pipeline_off = list(quantizer.build_prequant_pipeline(setting, executor=_Exec(graph=graph, device='cpu')))
    assert not any(isinstance(p, RotaryEmbeddingFusionPass) for p in pipeline_off)
    legalize_off = [p for p in pipeline_off if isinstance(p, EspdlRotaryEmbeddingLegalizePass)]
    assert len(legalize_off) == 1 and legalize_off[0].strict is True

    os.unlink(path)
    print('PASS: test_pass_is_auto_attached')


# ---------------------------------------------------------------------------
# Export-time cache dedupe
# ---------------------------------------------------------------------------
def test_shared_caches_are_deduped():
    """Two layers with identical caches collapse to one canonical pair."""
    from esp_ppq.parser.espdl.espdl_typedef import ExporterPatternInfo
    from esp_ppq.parser.espdl_exporter import dedupe_espdl_shareable_parameters

    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    half = head_dim // 2
    cos, sin = _rope_tables(seq_len, half)

    # 1. Single RoPE layer with distinct cos/sin has nothing to merge.
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='dedupe_single')
    graph = _load(path)
    sample = torch.randn(batch_size, head_count, seq_len, head_dim)
    executor = _trace(graph, sample)
    RotaryEmbeddingFusionPass().optimize(graph, dataloader=None, executor=executor, inputs=[sample])

    rope = next(op for op in graph.operations.values() if op.type == ROPE)
    before = sum(1 for v in graph.variables.values() if v.is_parameter)

    stats = dedupe_espdl_shareable_parameters(graph)
    assert stats['variables_merged'] == 0, stats
    assert sum(1 for v in graph.variables.values() if v.is_parameter) == before
    assert rope.inputs[1].value is not None and rope.inputs[2].value is not None
    os.unlink(path)

    # 2. Two RoPE layers sharing identical cos/sin tables -> merged into one pair.
    inits = (
        _slice_consts('lo0', 0, half)
        + _slice_consts('hi0', half, head_dim)
        + _slice_consts('lo1', 0, half)
        + _slice_consts('hi1', half, head_dim)
        + [
            OH.make_tensor('cos0', TP.FLOAT, cos.shape, cos.flatten().tolist()),
            OH.make_tensor('sin0', TP.FLOAT, sin.shape, sin.flatten().tolist()),
            OH.make_tensor('cos1', TP.FLOAT, cos.shape, cos.flatten().tolist()),
            OH.make_tensor('sin1', TP.FLOAT, sin.shape, sin.flatten().tolist()),
        ]
    )
    nodes = [
        _slice_node('lo0', 'X', 'x_lo0'),
        _slice_node('hi0', 'X', 'x_hi0'),
        OH.make_node('Neg', ['x_hi0'], ['n_hi0']),
        OH.make_node('Concat', ['n_hi0', 'x_lo0'], ['rot0'], axis=-1),
        OH.make_node('Mul', ['X', 'cos0'], ['xcos0']),
        OH.make_node('Mul', ['rot0', 'sin0'], ['rsin0']),
        OH.make_node('Add', ['xcos0', 'rsin0'], ['H0']),
        _slice_node('lo1', 'H0', 'x_lo1'),
        _slice_node('hi1', 'H0', 'x_hi1'),
        OH.make_node('Neg', ['x_hi1'], ['n_hi1']),
        OH.make_node('Concat', ['n_hi1', 'x_lo1'], ['rot1'], axis=-1),
        OH.make_node('Mul', ['H0', 'cos1'], ['xcos1']),
        OH.make_node('Mul', ['rot1', 'sin1'], ['rsin1']),
        OH.make_node('Add', ['xcos1', 'rsin1'], ['Y']),
    ]
    path2 = _finish(nodes, inits, [batch_size, head_count, seq_len, head_dim], 'Y', 'dedupe_two_layers')
    graph2 = _load(path2)
    sample2 = torch.randn(batch_size, head_count, seq_len, head_dim)
    executor2 = _trace(graph2, sample2)
    RotaryEmbeddingFusionPass().optimize(graph2, dataloader=None, executor=executor2, inputs=[sample2])
    ropes = [op for op in graph2.operations.values() if op.type == ROPE]
    assert len(ropes) == 2
    assert ropes[0].inputs[1] is not ropes[1].inputs[1]

    # Equal integer payloads with different final exponents represent
    # different real values. They must remain separate even though the raw
    # bytes, dtype, and shape all match.
    info = ExporterPatternInfo()
    for slot in (1, 2):
        info.add_var_exponents(ropes[0].inputs[slot].name, [-15])
        info.add_var_exponents(ropes[1].inputs[slot].name, [-14])
    mismatch_stats = dedupe_espdl_shareable_parameters(graph2)
    assert mismatch_stats['variables_merged'] == 0, mismatch_stats
    assert ropes[0].inputs[1] is not ropes[1].inputs[1]
    assert ropes[0].inputs[2] is not ropes[1].inputs[2]

    for slot in (1, 2):
        info.add_var_exponents(ropes[1].inputs[slot].name, [-15])
    # A parameter exposed as a graph output must not be popped by dedupe.
    boundary_cos = ropes[1].inputs[1]
    graph2.mark_variable_as_graph_output(boundary_cos)
    stats_boundary = dedupe_espdl_shareable_parameters(graph2)
    assert stats_boundary['variables_merged'] == 1, stats_boundary
    assert graph2.variables.get(boundary_cos.name) is boundary_cos
    assert graph2.outputs.get(boundary_cos.name) is boundary_cos
    assert ropes[0].inputs[2] is ropes[1].inputs[2]
    graph2.outputs.pop(boundary_cos.name)

    stats2 = dedupe_espdl_shareable_parameters(graph2)
    assert stats2['variables_merged'] == 1, stats2
    assert stats2['groups_merged'] == 1, stats2
    assert ropes[0].inputs[1] is ropes[1].inputs[1]
    assert ropes[0].inputs[2] is ropes[1].inputs[2]

    info.reset()
    os.unlink(path2)
    print('PASS: test_shared_caches_are_deduped')


def test_interior_exclusivity_and_shared_parameter_safety():
    """Intermediate nodes with external consumers or in graph.outputs must not fuse,
    and fusing a RoPE node must not wipe a parameter shared with another op."""
    batch_size, head_count, seq_len, head_dim = 1, 2, 8, 16
    half = head_dim // 2
    cos, sin = _rope_tables(seq_len, half)

    # 1. Intermediate tensor ('rot') is also a graph output -> skip fusion safely.
    inits = (
        _slice_consts('lo', 0, half)
        + _slice_consts('hi', half, head_dim)
        + [
            OH.make_tensor('cos', TP.FLOAT, cos.shape, cos.flatten().tolist()),
            OH.make_tensor('sin', TP.FLOAT, sin.shape, sin.flatten().tolist()),
        ]
    )
    nodes = [
        _slice_node('lo', 'X', 'x_lo'),
        _slice_node('hi', 'X', 'x_hi'),
        OH.make_node('Neg', ['x_hi'], ['n_hi']),
        OH.make_node('Concat', ['n_hi', 'x_lo'], ['rot'], axis=-1),
        OH.make_node('Mul', ['X', 'cos'], ['xcos']),
        OH.make_node('Mul', ['rot', 'sin'], ['rsin']),
        OH.make_node('Add', ['xcos', 'rsin'], ['Y']),
    ]
    g_pb = OH.make_graph(
        nodes,
        'rope_graph_out',
        [OH.make_tensor_value_info('X', TP.FLOAT, [batch_size, head_count, seq_len, head_dim])],
        [
            OH.make_tensor_value_info('Y', TP.FLOAT, [batch_size, head_count, seq_len, head_dim]),
            OH.make_tensor_value_info('rot', TP.FLOAT, [batch_size, head_count, seq_len, head_dim]),
        ],
        inits,
    )
    path_out = _save(OH.make_model(g_pb, opset_imports=[OH.make_opsetid('', 17)]), 'interior_out')
    g_out = _load(path_out)
    g_out.inputs['X'].shape = [batch_size, head_count, seq_len, head_dim]
    p_out = RotaryEmbeddingFusionPass()
    p_out.optimize(g_out)
    assert p_out.fused_count == 0
    os.unlink(path_out)

    # 2. Shared cos parameter also feeds an external Mul -> RoPE fuses AND external
    #    Mul's cos parameter retains its value (not wiped by remove_operation).
    nodes_shared = list(nodes) + [
        OH.make_node('Mul', ['Y', 'cos'], ['Z'], name='ext_mul'),
    ]
    path_shared = _finish(nodes_shared, inits, [batch_size, head_count, seq_len, head_dim], 'Z', 'shared_cos')
    g_shared = _load(path_shared)
    # Manually merge ext_mul's cos parameter onto the RoPE Mul's cos Variable to
    # simulate a shared parameter Variable across ops:
    rope_mul_cos = next(op for op in g_shared.operations.values() if op.type == 'Mul' and op.name != 'ext_mul')
    ext_mul = g_shared.operations['ext_mul']
    shared_cos_var = rope_mul_cos.inputs[1]
    old_ext_cos = ext_mul.inputs[1]
    if old_ext_cos is not shared_cos_var:
        ext_mul.inputs[1] = shared_cos_var
        shared_cos_var.dest_ops.append(ext_mul)
        old_ext_cos.dest_ops.clear()
        g_shared.remove_variable(old_ext_cos)
    p_shared = RotaryEmbeddingFusionPass()
    # This intentionally shared parameter cannot be deployed to TorchExecutor
    # before fusion (PPQ rejects multi-destination parameters), so inject only
    # the static input metadata the fusion admission gate requires.
    g_shared.inputs['X'].shape = [batch_size, head_count, seq_len, head_dim]
    p_shared.optimize(g_shared)
    assert p_shared.fused_count == 1
    assert shared_cos_var.value is not None
    assert ext_mul in shared_cos_var.dest_ops
    os.unlink(path_shared)

    # 3. Constant-backed cos/sin and Slice bounds also fuse cleanly.
    const_nodes = [
        OH.make_node('Constant', [], ['lo_starts'], value=OH.make_tensor('v1', TP.INT64, [1], [0])),
        OH.make_node('Constant', [], ['lo_ends'], value=OH.make_tensor('v2', TP.INT64, [1], [half])),
        OH.make_node('Constant', [], ['lo_axes'], value=OH.make_tensor('v3', TP.INT64, [1], [-1])),
        OH.make_node('Constant', [], ['lo_steps'], value=OH.make_tensor('v4', TP.INT64, [1], [1])),
        OH.make_node('Constant', [], ['hi_starts'], value=OH.make_tensor('v5', TP.INT64, [1], [half])),
        OH.make_node('Constant', [], ['hi_ends'], value=OH.make_tensor('v6', TP.INT64, [1], [head_dim])),
        OH.make_node('Constant', [], ['hi_axes'], value=OH.make_tensor('v7', TP.INT64, [1], [-1])),
        OH.make_node('Constant', [], ['hi_steps'], value=OH.make_tensor('v8', TP.INT64, [1], [1])),
        OH.make_node('Constant', [], ['cos'], value=OH.make_tensor('vc', TP.FLOAT, cos.shape, cos.flatten().tolist())),
        OH.make_node('Constant', [], ['sin'], value=OH.make_tensor('vs', TP.FLOAT, sin.shape, sin.flatten().tolist())),
        _slice_node('lo', 'X', 'x_lo'),
        _slice_node('hi', 'X', 'x_hi'),
        OH.make_node('Neg', ['x_hi'], ['n_hi']),
        OH.make_node('Concat', ['n_hi', 'x_lo'], ['rot'], axis=-1),
        OH.make_node('Mul', ['X', 'cos'], ['xcos']),
        OH.make_node('Mul', ['rot', 'sin'], ['rsin']),
        OH.make_node('Add', ['xcos', 'rsin'], ['Y']),
    ]
    path_const = _finish(const_nodes, [], [batch_size, head_count, seq_len, head_dim], 'Y', 'const_rope')
    g_const, p_const, _ = _fuse_and_compare(path_const, (batch_size, head_count, seq_len, head_dim), expect_fused=True)
    assert p_const.fused_count == 1
    assert 'Constant' not in _op_types(g_const)
    os.unlink(path_const)

    print('PASS: test_interior_exclusivity_and_shared_parameter_safety')


def test_2d_and_batch_cache_validation_and_forward():
    """Verify 2D [L, half] and 3D [B, L, half] caches in RotaryEmbedding_forward
    and check_rotary_embedding, plus batch/seq mismatch rejection."""
    batch_size, head_count, seq_len, head_dim = 2, 2, 8, 16
    path = build_rotate_half(batch_size, head_count, seq_len, head_dim, order='neg_hi_lo', tag='cache2d')
    graph, _, _ = _fuse_and_compare(path, (batch_size, head_count, seq_len, head_dim))
    os.unlink(path)

    rope = next(op for op in graph.operations.values() if op.type == ROPE)
    x = torch.randn(batch_size, head_count, seq_len, head_dim)
    TorchExecutor(graph=graph, device='cpu').tracing_operation_meta(inputs=[x])
    ref = _run(graph, x)

    # 1. 2D [L, half] cache executes and passes check_rotary_embedding.
    cos_3d = rope.inputs[1].value.clone()
    sin_3d = rope.inputs[2].value.clone()
    rope.inputs[1].value = cos_3d.squeeze(0)
    rope.inputs[2].value = sin_3d.squeeze(0)
    errors, _ = check_rotary_embedding(graph, strict=True)
    assert not errors
    out_2d = _run(graph, x)
    assert (ref - out_2d).abs().max().item() < 1e-6

    # 2. 3D [B, L, half] per-batch cache executes and passes check_rotary_embedding.
    rope.inputs[1].value = cos_3d.expand(batch_size, -1, -1).contiguous()
    rope.inputs[2].value = sin_3d.expand(batch_size, -1, -1).contiguous()
    errors, _ = check_rotary_embedding(graph, strict=True)
    assert not errors
    out_b = _run(graph, x)
    assert (ref - out_b).abs().max().item() < 1e-6

    # 3. Cache batch mismatch (cb=3 != B=2 and != 1) is caught by check_rotary_embedding.
    rope.inputs[1].value = cos_3d.expand(3, -1, -1).contiguous()
    rope.inputs[2].value = sin_3d.expand(3, -1, -1).contiguous()
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any('cache batch' in e for e in errors), errors

    # 4. Cache sequence mismatch (seq=4 != L=8) is caught by check_rotary_embedding.
    rope.inputs[1].value = cos_3d[:, :4, :].contiguous()
    rope.inputs[2].value = sin_3d[:, :4, :].contiguous()
    errors, _ = check_rotary_embedding(graph, strict=False)
    assert any('cache sequence' in e for e in errors), errors

    print('PASS: test_2d_and_batch_cache_validation_and_forward')


def test_shape_awareness_and_float_export():
    """Verify that prequant tracing and EspdlExporter.prepare_graph_for_float_export
    correctly skip 3D X and fuse 4D X."""
    from esp_ppq.parser.espdl_exporter import EspdlExporter

    # 1. 3D X [B, L, D] must be skipped automatically when traced via executor/inputs.
    batch_size, seq_len, head_dim = 1, 8, 16
    half = head_dim // 2
    f = _freqs(seq_len, half)
    cos3 = np.concatenate([np.cos(f), np.cos(f)], axis=-1)[None].astype(np.float32)
    sin3 = np.concatenate([np.sin(f), np.sin(f)], axis=-1)[None].astype(np.float32)
    inits = (
        _slice_consts('lo', 0, half)
        + _slice_consts('hi', half, head_dim)
        + [
            OH.make_tensor('cos', TP.FLOAT, cos3.shape, cos3.flatten().tolist()),
            OH.make_tensor('sin', TP.FLOAT, sin3.shape, sin3.flatten().tolist()),
        ]
    )
    nodes = [
        _slice_node('lo', 'X', 'x_lo'),
        _slice_node('hi', 'X', 'x_hi'),
        OH.make_node('Neg', ['x_hi'], ['n_hi']),
        OH.make_node('Concat', ['n_hi', 'x_lo'], ['rot'], axis=-1),
        OH.make_node('Mul', ['X', 'cos'], ['xcos']),
        OH.make_node('Mul', ['rot', 'sin'], ['rsin']),
        OH.make_node('Add', ['xcos', 'rsin'], ['Y']),
    ]
    path_3d = _finish(nodes, inits, [batch_size, seq_len, head_dim], 'Y', 'rope_3d')
    g_3d = _load(path_3d)
    p_3d = RotaryEmbeddingFusionPass()
    p_3d.optimize(
        g_3d, executor=TorchExecutor(graph=g_3d, device='cpu'), inputs=[torch.randn(batch_size, seq_len, head_dim)]
    )
    assert p_3d.fused_count == 0
    os.unlink(path_3d)

    # 2. prepare_graph_for_float_export fuses 4D RoPE on the float export path.
    head_count = 2
    path_4d = build_rotate_half(batch_size, head_count, seq_len, 32, order='neg_hi_lo', tag='float_export')
    g_4d = load_onnx_graph(path_4d)
    sample = [torch.randn(batch_size, head_count, seq_len, 32)]
    EspdlExporter.prepare_graph_for_float_export(
        graph=g_4d,
        platform=TargetPlatform.ESPDL_INT8,
        inputs=sample,
        device='cpu',
    )
    assert any(op.type == ROPE for op in g_4d.operations.values())

    # 3. End-to-end quantize_onnx_model on ESPDL_INT8 fuses RoPE, calibrates,
    #    and runs post-calibration SIMD exponent hint checks.
    from esp_ppq.api import quantize_onnx_model

    q_graph = quantize_onnx_model(
        onnx_import_file=path_4d,
        calib_dataloader=[sample, [t.clone() for t in sample]],
        calib_steps=2,
        input_shape=None,
        platform=TargetPlatform.ESPDL_INT8,
        inputs=sample,
        setting=QuantizationSettingFactory.espdl_setting(),
        device='cpu',
        verbose=0,
    )
    q_rope = next(op for op in q_graph.operations.values() if op.type == ROPE)
    assert hasattr(q_rope, 'input_quant_config') and len(q_rope.input_quant_config) == 3
    cos_cfg, sin_cfg = q_rope.input_quant_config[1:3]
    assert cos_cfg.num_of_bits == 16 and sin_cfg.num_of_bits == 16
    assert sin_cfg.dominated_by == cos_cfg, "cos/sin caches must share one exponent"
    assert torch.equal(cos_cfg.scale, sin_cfg.scale)

    # 4. The final export gate sees the calibrated RoPE contract before
    # parameter baking. A successful FlatBuffer write proves that its
    # int8-X/int16-cache/int8-Y contract survives the exporter patterns.
    espdl_path = path_4d + ".espdl"
    EspdlExporter().export(file_path=espdl_path, graph=q_graph, export_config=False)
    assert os.path.isfile(espdl_path) and os.path.getsize(espdl_path) > 0
    os.unlink(espdl_path)

    os.unlink(path_4d)

    print('PASS: test_shape_awareness_and_float_export')


def test_fusion_trace_requires_only_equivalence_shape_metadata():
    """Fusion needs X shape; prequant separately traces dtype for Legalize."""
    shape = (1, 2, 8, 16)
    path = build_rotate_half(*shape, order='neg_hi_lo', tag='trace_add_output')
    try:
        graph = _load(path)
        _trace(graph, torch.randn(*shape))
        x = graph.inputs['X']
        y = graph.outputs['Y']
        assert y.shape is not None and y.dtype is not None

        # Intermediate shapes/dtypes and Y shape are not needed to prove this
        # rewrite: static X plus static cache values determine all geometry.
        for op in graph.operations.values():
            if op.type in {'Slice', 'Neg', 'Concat', 'Mul'}:
                for output in op.outputs:
                    output.shape = None
                    output._dtype = None
        y.shape = None
        y._dtype = None
        assert not _rope_fusion_needs_trace(graph)

        # The configured prequant pipeline still traces once because the
        # fused node's X/Y dtype is required by the following LegalizePass.
        assert _rope_prequant_needs_trace(graph, do_fusion=True)
        assert not _rope_prequant_needs_trace(graph, do_fusion=False)

        # Once prospective X/Y dtypes are known, no intermediate metadata is
        # required. Fusion restores Y shape from the proven X geometry.
        y._dtype = x.dtype
        assert not _rope_prequant_needs_trace(graph, do_fusion=True)
        RotaryEmbeddingFusionPass().optimize(graph)
        assert any(op.type == ROPE for op in graph.operations.values())
        assert y.shape == list(shape)
    finally:
        os.unlink(path)


if __name__ == '__main__':
    test_canonical_fusion()
    test_negated_sin_convention_fusion()
    test_partial_rope_wrapper_is_not_absorbed()
    test_interleaved_graphs_are_never_fused()
    test_mirrored_concat_is_rejected()
    test_per_head_cache_is_rejected()
    test_unequal_halves_cache_is_rejected()
    test_unrelated_add_is_not_matched()
    test_halve_rope_cache_contract()
    test_support_check_accepts_fused_graph()
    test_support_check_simd_hints_vision_shapes()
    test_support_check_int16_cache_hints_track_kernel_gates()
    test_support_check_flags_bad_cache()
    test_support_check_rejects_out_of_subset_configs()
    test_fusion_and_legalize_passes_are_separate()
    test_pass_is_auto_attached()
    test_shared_caches_are_deduped()
    test_interior_exclusivity_and_shared_parameter_safety()
    test_2d_and_batch_cache_validation_and_forward()
    test_shape_awareness_and_float_export()
    test_fusion_trace_requires_only_equivalence_shape_metadata()
    print('\nAll RotaryEmbedding fusion tests passed.')
