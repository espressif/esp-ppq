"""Quantization helpers used by AutoQuant."""

import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import onnx
from onnxsim import simplify
from torch.utils.data import DataLoader

from esp_ppq.api.espdl_interface import espdl_quantize_onnx, get_target_platform
from esp_ppq.api.setting import QuantizationSetting, QuantizationSettingFactory
from esp_ppq.IR import BaseGraph
from esp_ppq.log import NaiveLogger
from esp_ppq.parser import NativeExporter
from esp_ppq.quantization.analyse import layerwise_error_analyse

logger = NaiveLogger.get_logger("AutoQuant")


def simplify_onnx(onnx_path: str) -> str:
    """Return a simplified ONNX path without overwriting the original model."""
    base, _ext = os.path.splitext(onnx_path)
    out_path = f"{base}_simplified.onnx"

    if not os.path.isfile(out_path) or os.path.getmtime(out_path) < os.path.getmtime(onnx_path):
        logger.info(f"Simplifying ONNX {onnx_path} -> {out_path}")
        model = onnx.load(onnx_path)
        model, ok = simplify(model)
        if not ok:
            raise RuntimeError(f"onnxsim failed on {onnx_path}")
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, out_path)

    return out_path


def _check_topk(name: str, topk: int, allow_all: bool) -> None:
    if not isinstance(topk, int) or isinstance(topk, bool):
        raise TypeError(f"{name}.topk must be int, got {type(topk).__name__} ({topk!r})")
    if topk == -1 and allow_all:
        return
    if topk <= 0:
        if allow_all:
            raise ValueError(f"{name}.topk must be -1 or > 0, got {topk}")
        raise ValueError(f"{name}.topk must be > 0, got {topk}")


def select_topk_layers(layerwise_error: Dict[str, float], topk: int) -> List[str]:
    """Return the ``topk`` layer names with the largest layerwise error.

    ``topk == -1`` returns an empty list, which PPQ's ``interested_layers``
    interprets as "apply the pass to all eligible layers". ``topk > 0`` returns
    the top-``topk`` highest-error layer names, ordered from highest to lowest
    error. Any other value raises ``ValueError``.
    """
    _check_topk("interested_layers", topk, allow_all=True)
    if topk == -1:
        return []
    return [name for name, _ in sorted(layerwise_error.items(), key=lambda x: x[1], reverse=True)[:topk]]


def get_next_node(onnx_path: str, layer_name: str) -> List[str]:
    """Return the names of operations that consume ``layer_name``'s outputs."""
    model = onnx.load(onnx_path)
    consumers: Dict[str, List[str]] = defaultdict(list)
    target = None
    for n in model.graph.node:
        for inp in n.input:
            consumers[inp].append(n.name)
        if n.name == layer_name:
            target = n
    if target is None:
        raise ValueError(f"Node not found in {onnx_path}: {layer_name}")

    next_names = set()
    for out in target.output:
        for name in consumers.get(out, []):
            next_names.add(name)
    return list(next_names)


def build_quant_setting(
    strategy: Dict[str, dict],
    sampled_param: Dict[str, dict],
    target: str,
    layerwise_error: Dict[str, float],
    simplified_onnx_path: str,
) -> Tuple[QuantizationSetting, int]:
    """Translate a sampled AutoQuant candidate into an ESP-DL quant setting."""
    num_of_bits = sampled_param["num_of_bits"]["value"]
    calib_algorithm = sampled_param["calib_algorithm"]["method"]

    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_setting.quantize_activation_setting.calib_algorithm = calib_algorithm
    quant_setting.equalization = strategy["weight_equalization"]["value"]
    quant_setting.weight_split = strategy["horizontal_layer_split"]["value"]
    quant_setting.bias_correct = strategy["bias_correction"]["value"]
    quant_setting.tqt_optimization = strategy["tqt"]["value"]
    quant_setting.fusion = True
    quant_setting.fusion_setting.align_quantization = strategy["fusion_alignment"]["value"]
    quant_setting.blockwise_reconstruction = strategy["blockwise_reconstruction"]["value"]

    if strategy["mixed_precision"]["value"]:
        mp_cfg = sampled_param.get("mixed_precision")
        if mp_cfg is None:
            raise ValueError("Missing sampled mixed_precision config")
        if not layerwise_error:
            raise ValueError("mixed_precision requires layerwise_error; compute it before calling build_quant_setting.")
        mp_topk = mp_cfg.get("topk", 1)
        _check_topk("mixed_precision", mp_topk, allow_all=False)
        interested_layers = select_topk_layers(layerwise_error, mp_topk)

        # Keep direct successors on the same higher-precision platform.
        mp_layers: List[str] = []
        for layer_name in interested_layers:
            mp_layers.append(layer_name)
            for nxt in get_next_node(simplified_onnx_path, layer_name):
                if nxt not in mp_layers:
                    mp_layers.append(nxt)
        for name in mp_layers:
            quant_setting.dispatching_table.append(name, get_target_platform(target, 16))
        logger.info(f"Mixed precision int16 layers: {mp_layers}")

    if strategy["bias_correction"]["value"]:
        bc_cfg = sampled_param["bias_correction"]
        if not layerwise_error:
            raise ValueError("bias_correction requires layerwise_error; compute it before calling build_quant_setting.")
        bc_topk = bc_cfg.get("topk", 1)
        _check_topk("bias_correction", bc_topk, allow_all=True)
        quant_setting.bias_correct_setting.interested_layers = select_topk_layers(layerwise_error, bc_topk)
        quant_setting.bias_correct_setting.block_size = bc_cfg["block_size"]
        quant_setting.bias_correct_setting.steps = bc_cfg["steps"]

    if strategy["horizontal_layer_split"]["value"]:
        hls_cfg = sampled_param["horizontal_layer_split"]
        if not layerwise_error:
            raise ValueError(
                "horizontal_layer_split requires layerwise_error; compute it before calling build_quant_setting."
            )
        hls_topk = hls_cfg.get("topk", 1)
        _check_topk("horizontal_layer_split", hls_topk, allow_all=False)
        quant_setting.weight_split_setting.interested_layers = select_topk_layers(layerwise_error, hls_topk)
        quant_setting.weight_split_setting.value_threshold = hls_cfg["value_threshold"]

    if strategy["weight_equalization"]["value"]:
        wq_cfg = sampled_param["weight_equalization"]
        quant_setting.equalization_setting.iterations = wq_cfg["iterations"]
        quant_setting.equalization_setting.value_threshold = wq_cfg["value_threshold"]
        quant_setting.equalization_setting.opt_level = wq_cfg["opt_level"]

    if strategy["tqt"]["value"]:
        tqt_cfg = sampled_param["tqt"]
        if not layerwise_error:
            raise ValueError("tqt requires layerwise_error; compute it before calling build_quant_setting.")
        tqt_setting = quant_setting.tqt_optimization_setting
        tqt_setting.lr = tqt_cfg["lr"]
        tqt_setting.steps = tqt_cfg["steps"]
        tqt_setting.block_size = tqt_cfg["block_size"]
        tqt_setting.is_scale_trainable = tqt_cfg["is_scale_trainable"]
        tqt_setting.gamma = tqt_cfg["gamma"]
        tqt_setting.int_lambda = tqt_cfg["int_lambda"]
        tqt_setting.collecting_device = tqt_cfg["collecting_device"]
        _check_topk("tqt", tqt_cfg["topk"], allow_all=True)
        tqt_setting.interested_layers = select_topk_layers(layerwise_error, tqt_cfg["topk"])

    if strategy["fusion_alignment"]["value"]:
        fa_cfg = sampled_param["fusion_alignment"]
        quant_setting.fusion_setting.align_elementwise_to = fa_cfg["elementwise_to"]
        quant_setting.fusion_setting.align_concat_to = fa_cfg["concat_to"]
        quant_setting.fusion_setting.align_avgpooling_to = fa_cfg["avgpooling_to"]
        quant_setting.fusion_setting.align_resize_to = fa_cfg["resize_to"]
        quant_setting.fusion_setting.force_alignment_overlap = fa_cfg["force_overlap"]

    if strategy["blockwise_reconstruction"]["value"]:
        br_cfg = sampled_param["blockwise_reconstruction"]
        if not layerwise_error:
            raise ValueError(
                "blockwise_reconstruction requires layerwise_error; compute it before calling build_quant_setting."
            )
        br_setting = quant_setting.blockwise_reconstruction_setting
        br_setting.is_scale_trainable = br_cfg["is_scale_trainable"]
        br_topk = br_cfg.get("topk", 1)
        _check_topk("blockwise_reconstruction", br_topk, allow_all=True)
        br_setting.interested_layers = select_topk_layers(layerwise_error, br_topk)
        br_setting.block_size = br_cfg["block_size"]
        br_setting.steps = br_cfg["steps"]
        br_setting.lr = br_cfg["lr"]
        br_setting.gamma = br_cfg["gamma"]
        br_setting.collecting_device = br_cfg["collecting_device"]

    return quant_setting, num_of_bits


def _default_collate(device: str) -> Callable[[Any], Any]:
    def _collate(batch):
        return batch.to(device)

    return _collate


def quant_model(
    onnx_import_file: str,
    espdl_export_file: str,
    input_shape: List[int],
    calib_dataloader: DataLoader,
    quant_setting: QuantizationSetting,
    num_of_bits: int,
    target: str,
    device: str,
    collate_fn: Callable = None,
    calib_steps: int = 32,
    verbose: int = 0,
) -> BaseGraph:
    """Run one ESP-DL quantization candidate and export its native graph."""
    simplified_onnx_path = simplify_onnx(onnx_import_file)
    collate = collate_fn or _default_collate(device)

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=simplified_onnx_path,
        espdl_export_file=espdl_export_file,
        calib_dataloader=calib_dataloader,
        calib_steps=calib_steps,
        input_shape=[1] + list(input_shape),
        target=target,
        num_of_bits=num_of_bits,
        collate_fn=collate,
        setting=quant_setting,
        device=device,
        error_report=False,
        skip_export=False,
        export_test_values=False,
        verbose=verbose,
        inputs=None,
    )

    native_path = os.path.splitext(espdl_export_file)[0] + ".native"
    NativeExporter().export(file_path=native_path, graph=quant_graph)

    return quant_graph


def compute_layerwise_error(
    onnx_import_file: str,
    espdl_export_file: str,
    input_shape: List[int],
    calib_dataloader: DataLoader,
    target: str,
    device: str,
    collate_fn: Callable = None,
    calib_steps: int = 32,
) -> Dict[str, float]:
    """Run baseline INT8 quantization and return the layerwise error report."""
    quant_setting = QuantizationSettingFactory.espdl_setting()
    quant_graph = quant_model(
        onnx_import_file=onnx_import_file,
        espdl_export_file=espdl_export_file,
        input_shape=input_shape,
        calib_dataloader=calib_dataloader,
        quant_setting=quant_setting,
        num_of_bits=8,
        target=target,
        device=device,
        collate_fn=collate_fn,
        calib_steps=calib_steps,
        verbose=0,
    )

    collate = collate_fn or _default_collate(device)
    return layerwise_error_analyse(
        graph=quant_graph,
        running_device=device,
        collate_fn=collate,
        dataloader=calib_dataloader,
    )
