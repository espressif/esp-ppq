"""Entry point and search loop for ESP-DL AutoQuant.

The public API is :func:`espdl_auto_quantize_onnx`. It evaluates quantization
strategy candidates, stores experiment artifacts, and returns the current Top-K
records. Failures from quantization or user callbacks are not swallowed; see
the function's ``Raises`` section for the exceptions a caller may observe.
"""

import hashlib
import itertools
import json
import math
import os
from numbers import Real
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from torch.utils.data import DataLoader

from esp_ppq.IR import BaseGraph
from esp_ppq.log import NaiveLogger
from esp_ppq.quantization.analyse import graphwise_error_analyse

from .candidates import (
    load_candidates,
    update_topk_candidates,
)
from .quantize import (
    build_quant_setting,
    compute_layerwise_error,
    quant_model,
    simplify_onnx,
)
from .save import (
    create_run_dir,
    load_summary,
    reset_run_dir,
    save_experiment,
    update_summary,
)
from .search import build_search_pipeline
from .setting import (
    AutoQuantSearchSetting,
)

logger = NaiveLogger.get_logger("AutoQuant")

# These passes choose interested layers from a baseline layerwise-error report.
_LAYERWISE_DEPENDENT_MODULES = (
    "bias_correction",
    "horizontal_layer_split",
    "blockwise_reconstruction",
    "mixed_precision",
    "tqt",
)

# Keys that AutoQuant writes into each record itself (from `save_experiment`
# or AutoQuant bookkeeping). `evaluate_fn` extras must not collide with them,
# otherwise persisted artifacts and Top-K sorting would silently use the wrong
# values.
_RESERVED_EXTRA_KEYS = frozenset({"score", "hash", "index", "folder", "files", "strategy", "params"})


# Param keys that must be excluded from the resume hash. ``collecting_device``
# is unconditionally overwritten by the top-level ``device`` argument, so
# including it would invalidate resume across device boundaries even though
# the underlying experiment is the same.
_HASH_IGNORED_PARAM_KEYS = frozenset({"collecting_device"})


def _strategy_hash(strategy: Dict[str, dict], sampled_param: Dict[str, dict] = None) -> str:
    """Return a stable hash for resume lookup."""
    payload: Dict[str, Any] = {"strategy": {k: v["value"] for k, v in strategy.items()}}
    if sampled_param is not None:
        payload["params"] = {
            module: (
                {k: v for k, v in params.items() if k not in _HASH_IGNORED_PARAM_KEYS}
                if isinstance(params, dict)
                else params
            )
            for module, params in sampled_param.items()
        }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()


def _iter_param_combos(strategy: Dict[str, dict]) -> Iterator[Dict[str, dict]]:
    """Yield all parameter samples for one strategy."""
    sets = [v["param_candidates"] if v["value"] else [None] for v in strategy.values()]
    for combo in itertools.product(*sets):
        yield dict(zip(strategy.keys(), combo))


def _split_param_candidates(
    strategy: Dict[str, dict], structural_knobs: Dict[str, List[str]]
) -> Tuple[List[Dict[str, dict]], Callable[[Dict[str, dict]], Iterator[Dict[str, dict]]]]:
    """Build the two-stage parameter iterator used by fast search."""
    keys = list(strategy.keys())

    def is_struct(module: str, knob: str) -> bool:
        return knob in structural_knobs.get(module, [])

    def gather(module: str):
        """Collect the per-knob value lists for one enabled module."""
        v = strategy[module]
        if not v["value"] or not v["param_candidates"]:
            return None
        first = v["param_candidates"][0]
        knob_keys = list(first.keys())
        values_per_knob: Dict[str, List[Any]] = {k: [] for k in knob_keys}
        for cand in v["param_candidates"]:
            for k in knob_keys:
                if cand[k] not in values_per_knob[k]:
                    values_per_knob[k].append(cand[k])
        return knob_keys, values_per_knob, first

    def expand_module(module: str, mode: str, fixed: Dict[str, dict] = None):
        info = gather(module)
        if info is None:
            return [None]
        knob_keys, values_per_knob, first = info
        sources: List[List[Any]] = []
        for k in knob_keys:
            structural = is_struct(module, k)
            if mode == "struct":
                sources.append(values_per_knob[k] if structural else [first[k]])
            else:  # mode == "knob"
                if structural:
                    sources.append([fixed[module][k]])
                else:
                    sources.append(values_per_knob[k])
        return [dict(zip(knob_keys, c)) for c in itertools.product(*sources)]

    struct_per_module = [expand_module(k, "struct", None) for k in keys]
    structural_grid = [dict(zip(keys, combo)) for combo in itertools.product(*struct_per_module)]

    def knob_iter(fixed_struct: Dict[str, dict]) -> Iterator[Dict[str, dict]]:
        knob_per_module = [expand_module(k, "knob", fixed_struct) for k in keys]
        for combo in itertools.product(*knob_per_module):
            yield dict(zip(keys, combo))

    return structural_grid, knob_iter


def _load_resume(resume: bool, run_dir: str, metric_key: str) -> Tuple[Set[str], Dict[str, float], int]:
    """Load completed experiment hashes and the next experiment index."""
    if not resume:
        return set(), {}, 0
    prev = load_summary(run_dir)
    done_hashes = {e["hash"] for e in prev if "hash" in e}
    done_metrics = {e["hash"]: e[metric_key] for e in prev if "hash" in e and metric_key in e}
    next_index = max((e.get("index", -1) for e in prev), default=-1) + 1
    logger.info(f"[Resume] {len(done_hashes)} experiments already done; next index={next_index}")
    return done_hashes, done_metrics, next_index


def _validate_resume_artifacts(run_dir: str, candidates_path: str) -> None:
    """Fail fast when resume files are missing, malformed, or inconsistent."""
    summary_path = os.path.join(run_dir, "summary.json")
    missing: List[str] = []
    if not os.path.isfile(summary_path):
        missing.append(summary_path)
    if not os.path.isfile(candidates_path):
        missing.append(candidates_path)
    if missing:
        raise ValueError(f"setting.resume=True requires both resume files to exist: {', '.join(missing)}")

    try:
        summary = load_summary(run_dir)
        candidates = load_candidates(candidates_path)
    except json.JSONDecodeError as e:
        raise ValueError("Resume files are malformed JSON. Please fix them or rerun with setting.resume=False.") from e

    if not isinstance(summary, list) or not isinstance(candidates, list):
        raise ValueError("Resume files are invalid: summary.json and candidates.json must both be JSON lists.")
    if any(not isinstance(x, dict) for x in summary):
        raise ValueError("summary.json is invalid: every item must be a JSON object.")
    if any(not isinstance(x, dict) for x in candidates):
        raise ValueError("candidates.json is invalid: every item must be a JSON object.")

    # If one side is empty while the other is non-empty, resume state is incomplete.
    if (len(summary) == 0) != (len(candidates) == 0):
        raise ValueError(
            "Resume files are inconsistent: summary.json and candidates.json "
            "must be both empty or both non-empty. "
            "Please rerun with setting.resume=False to rebuild artifacts."
        )


def _needs_layerwise_error(setting: AutoQuantSearchSetting) -> bool:
    """Whether the search space can enable a layerwise-error-dependent pass."""
    for name in _LAYERWISE_DEPENDENT_MODULES:
        if True in setting.strategy_space.get(name, []):
            return True
    return False


def _run_exhaustive(
    strategies: Iterable[Dict[str, dict]],
    eval_once: Callable[[Dict[str, dict], Dict[str, dict]], float],
) -> None:
    """Run exhaustive search over every strategy and parameter sample."""
    for s in strategies:
        logger.info("=== Strategy ===")
        logger.info(f"[strategy] { {k: v['value'] for k, v in s.items()} }")
        for p in _iter_param_combos(s):
            eval_once(s, p)


def _is_better(score: float, best: float, score_direction: str) -> bool:
    """True when ``score`` improves on ``best`` under the configured direction.

    ``score_direction`` follows the Optuna convention and must be either
    ``"maximize"`` or ``"minimize"``.
    """
    return score > best if score_direction == "maximize" else score < best


def _run_fast(
    strategies: Iterable[Dict[str, dict]],
    eval_once: Callable[[Dict[str, dict], Dict[str, dict]], float],
    top: int,
    t_max: int,
    structural_knobs: Dict[str, List[str]],
    score_direction: str,
) -> None:
    """Run structural-first search with early stop on non-improving samples."""
    scored: List[Tuple[float, Dict[str, dict], Dict[str, dict]]] = []
    for s in strategies:
        logger.info("=== Strategy (stage1) ===")
        logger.info(f"[strategy] { {k: v['value'] for k, v in s.items()} }")
        struct_grid, _ = _split_param_candidates(s, structural_knobs)
        best_score, best_struct = None, None
        for struct_sample in struct_grid:
            score = eval_once(s, struct_sample)
            if best_score is None or _is_better(score, best_score, score_direction):
                best_score, best_struct = score, struct_sample
        if best_score is not None:
            logger.info(
                f"[stage1] best_score={best_score} "
                f"struct={ {k: v for k, v in best_struct.items() if v is not None and k in structural_knobs} }"
            )
            scored.append((best_score, s, best_struct))

    scored.sort(key=lambda x: x[0], reverse=(score_direction == "maximize"))
    picked = scored[: max(int(top), 1)]
    logger.info(f"[fast] picked {len(picked)} of {len(scored)} strategies")

    t_max = max(int(t_max), 1)
    for best, s, fixed_struct in picked:
        logger.info("=== Strategy (stage2) ===")
        logger.info(f"[strategy] { {k: v['value'] for k, v in s.items()} }")
        logger.info(
            f"[stage2] fixed_struct={ {k: v for k, v in fixed_struct.items() if v is not None and k in structural_knobs} }"
        )
        _, knob_iter = _split_param_candidates(s, structural_knobs)
        no_improve = 0
        for p in knob_iter(fixed_struct):
            if p == fixed_struct:
                # Already evaluated in stage 1.
                continue
            score = eval_once(s, p)
            if _is_better(score, best, score_direction):
                best, no_improve = score, 0
            else:
                no_improve += 1
            if no_improve >= t_max:
                logger.info(f"[fast] early stop: {_strategy_hash(s)[:8]}")
                break


def espdl_auto_quantize_onnx(
    onnx_import_file: str,
    espdl_export_file: str,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    evaluate_fn: Optional[Callable[[BaseGraph], Tuple[float, Dict[str, float]]]] = None,
    target: str = "esp32p4",
    collate_fn: Callable = None,
    setting: AutoQuantSearchSetting = None,
    device: str = "cpu",
    verbose: int = 0,
) -> List[Dict[str, Any]]:
    """Search ESP-DL quantization strategies and return Top-K candidates.

    The first five positional arguments and the relative ordering of
    ``target`` / ``collate_fn`` / ``setting`` / ``device`` / ``verbose``
    match :func:`espdl_quantize_onnx`.

    Args:
        onnx_import_file: Input ONNX model path.
        espdl_export_file: Output .espdl path used as the artifact base name.
        calib_dataloader: Calibration dataloader.
        calib_steps: Calibration steps forwarded to ``espdl_quantize_onnx``.
        input_shape: Input shape without the batch dimension.
        evaluate_fn: Optional user evaluation callback. When provided, it
            must return ``(score, extras)`` where ``score`` is a finite
            real number (``int`` / ``float`` / ``numpy`` scalar / any
            ``numbers.Real``; ``bool`` is explicitly rejected, and
            ``nan`` / ``inf`` are rejected as well — ranking and JSON
            persistence both require a finite numeric score) used to
            rank candidates, and ``extras`` is a ``Dict[str, float]`` of
            additional metrics persisted into ``candidates.json`` /
            ``summary.json`` (e.g. ``{"top1": 0.74, "top5": 0.92}``).
            The score is also stored under the fixed ``score`` field in
            each record. ``extras`` must not contain any of the reserved
            keys ``{"score", "hash", "index", "folder", "files",
            "strategy", "params"}``; AutoQuant owns them. Any deviation
            from this contract aborts the search (fail-fast). When
            ``None`` (default), AutoQuant
            scores each candidate by the mean of the top-3 highest
            graphwise noise/signal-power ratios computed via
            :func:`graphwise_error_analyse` on ``calib_dataloader``;
            since lower is better, the run uses ``"minimize"`` as the
            sort direction regardless of ``setting.score_direction``
            (an info log is emitted when the two disagree, and
            ``setting.score_direction`` itself is left untouched so a
            reused setting still reflects the user's original value).
        target: Target chip, such as ``"esp32p4"``, ``"esp32s3"``, or ``"c"``.
        collate_fn: Optional calibration batch collate function.
        setting: Search-space and search-behavior setting. Defaults to
            ``AutoQuantSearchSetting()``. Universal fields:
            ``search_mode``, ``num_of_candidates``, ``candidate_filter``,
            ``score_direction`` (Optuna-style ``"maximize"`` /
            ``"minimize"``), ``run_dir`` and ``resume`` control the search
            loop, Top-K policy and experiment bookkeeping. The fast-only
            fields (``top_strategy``, ``early_stop_patience``,
            ``structural_knobs``) are honored only when ``search_mode ==
            "fast"``. See :class:`AutoQuantSearchSetting` for the full list.
        device: Execution device forwarded to ``espdl_quantize_onnx`` and
            the baseline layerwise-error pass. It also overwrites
            ``param_space[...]["collecting_device"]`` for tqt /
            blockwise_reconstruction so the collecting device always
            matches the execution device.
        verbose: Verbosity forwarded to ``espdl_quantize_onnx``.

    Returns:
        Top-K candidate records ranked by ``score``. The sort direction
        is ``setting.score_direction`` when ``evaluate_fn`` is supplied,
        and ``"minimize"`` when the built-in default evaluator is used
        (``setting.score_direction`` itself is left untouched). The
        actually-used direction is also recorded in each experiment's
        ``config.json`` (``runtime.score_direction``).

    Raises:
        ValueError: Invalid arguments; reserved-key collision in
            ``evaluate_fn`` extras; resume artifacts missing or
            inconsistent; or — when the built-in default evaluator is
            used — ``graphwise_error_analyse`` finds no quantable
            computing op to score.
        TypeError: Invalid ``evaluate_fn`` return shape or types.
    """
    if calib_dataloader is None:
        raise ValueError("calib_dataloader is required")
    if calib_steps < 1:
        raise ValueError(f"calib_steps must be >= 1, got {calib_steps}")

    if setting is None:
        setting = AutoQuantSearchSetting()

    if setting.search_mode not in ("exhaustive", "fast"):
        raise ValueError(f"setting.search_mode must be 'exhaustive' or 'fast', got {setting.search_mode!r}")
    if setting.num_of_candidates < 1:
        raise ValueError(f"setting.num_of_candidates must be >= 1, got {setting.num_of_candidates}")
    if setting.score_direction not in ("maximize", "minimize"):
        raise ValueError(f"setting.score_direction must be 'maximize' or 'minimize', got {setting.score_direction!r}")

    # Default evaluator: when the caller does not supply `evaluate_fn`, score
    # each candidate by the mean of the top-3 highest graphwise noise/signal
    # power ratios on `calib_dataloader`. SNR is a loss (lower is better), so
    # this run sorts under "minimize" regardless of `setting.score_direction`;
    # we keep that as a *local* override so the user's setting is not
    # mutated across calls.
    if evaluate_fn is None:
        _default_top_n = 3

        def evaluate_fn(quant_graph: BaseGraph) -> Tuple[float, Dict[str, float]]:
            errors = (
                graphwise_error_analyse(
                    graph=quant_graph,
                    running_device=device,
                    dataloader=calib_dataloader,
                    collate_fn=collate_fn,
                    method="snr",
                    verbose=False,
                )
                or {}
            )
            if not errors:
                # No quantable computing op was analysed: the graph has
                # nothing to score. Fail-fast (AutoQuant convention) rather
                # than fabricating a sentinel score; the caller should fix
                # their model / dispatch instead of letting the search
                # silently rank a degenerate candidate.
                raise ValueError(
                    "Default evaluator: graphwise_error_analyse returned no "
                    "layers (no QuantableOperation with is_computing_op=True). "
                    "Provide a custom evaluate_fn or check the input ONNX / "
                    "dispatch configuration."
                )
            top = sorted(errors.values(), reverse=True)[:_default_top_n]
            avg = float(sum(top) / len(top))
            return avg, {
                "topn_mean_error": avg,
                "topn_used": len(top),
                "num_layers": len(errors),
            }

        effective_score_direction = "minimize"
        if setting.score_direction != "minimize":
            logger.info(
                f"Using built-in graphwise top-{_default_top_n} SNR evaluator; "
                f"this run sorts under 'minimize' (lower is better) instead of "
                f"setting.score_direction={setting.score_direction!r}. "
                f"setting.score_direction itself is left untouched."
            )
    else:
        effective_score_direction = setting.score_direction

    # `device` is the single knob: overwrite tqt / blockwise_reconstruction
    # `collecting_device` so it always matches the execution device.
    for _module in ("tqt", "blockwise_reconstruction"):
        if _module in setting.param_space:
            setting.param_space[_module]["collecting_device"] = [device]

    candidates_path = os.path.join(setting.run_dir, "candidates.json")
    if setting.resume:
        # Validate before creating `run_dir` so a failed resume does not
        # leave an empty directory on disk.
        _validate_resume_artifacts(setting.run_dir, candidates_path)
    run_dir = create_run_dir(setting.run_dir)
    done_hashes, done_metrics, next_index = _load_resume(setting.resume, run_dir, "score")
    if not setting.resume:
        reset_run_dir(run_dir)

    runtime_snapshot = {
        "onnx_import_file": onnx_import_file,
        "espdl_export_file": espdl_export_file,
        "input_shape": list(input_shape),
        "target": target,
        "device": device,
        "num_of_candidates": setting.num_of_candidates,
        "search_mode": setting.search_mode,
        # Record the direction this run actually sorted under, not the
        # user's raw `setting.score_direction`: the default evaluator
        # forces "minimize" locally without mutating the setting, so
        # logging the raw value here would mislead post-hoc analysis.
        "score_direction": effective_score_direction,
        "calib_steps": calib_steps,
    }
    if setting.search_mode == "fast":
        # Persist fast-only knobs so a saved config can be reproduced exactly.
        runtime_snapshot["top_strategy"] = setting.top_strategy
        runtime_snapshot["early_stop_patience"] = setting.early_stop_patience

    simplified_onnx_path = simplify_onnx(onnx_import_file)

    if _needs_layerwise_error(setting):
        layerwise_error = compute_layerwise_error(
            onnx_import_file=onnx_import_file,
            espdl_export_file=espdl_export_file,
            input_shape=input_shape,
            calib_dataloader=calib_dataloader,
            target=target,
            device=device,
            collate_fn=collate_fn,
            calib_steps=calib_steps,
        )
    else:
        layerwise_error = {}

    searcher, _rules = build_search_pipeline(setting.strategy_space, setting.param_space)
    strategies = list(searcher.search())

    state = {"exp_index": next_index}

    def _eval_once(strategy: Dict[str, dict], sampled_param: Dict[str, dict]) -> float:
        h = _strategy_hash(strategy, sampled_param)
        if h in done_hashes:
            if h not in done_metrics:
                raise ValueError(
                    f"Hash {h[:8]} exists in summary but missing 'score' field. "
                    "The summary is not compatible with the current AutoQuant format; "
                    "rerun with setting.resume=False to rebuild it."
                )
            logger.info(f"[skip] already done: {h[:8]}")
            return done_metrics[h]

        quant_setting, num_of_bits = build_quant_setting(
            strategy=strategy,
            sampled_param=sampled_param,
            target=target,
            layerwise_error=layerwise_error,
            simplified_onnx_path=simplified_onnx_path,
        )
        quant_graph = quant_model(
            onnx_import_file=onnx_import_file,
            espdl_export_file=espdl_export_file,
            input_shape=input_shape,
            calib_dataloader=calib_dataloader,
            quant_setting=quant_setting,
            num_of_bits=num_of_bits,
            target=target,
            device=device,
            collate_fn=collate_fn,
            calib_steps=calib_steps,
            verbose=verbose,
        )

        result = evaluate_fn(quant_graph)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "evaluate_fn must return a 2-tuple (score, extras_dict), "
                f"got {type(result).__name__}" + (f" of length {len(result)}" if isinstance(result, tuple) else "")
            )
        score, extras = result
        # Accept any real-number type (including numpy.float32, Decimal, ...)
        # via numbers.Real, but reject bool explicitly: True/False are
        # int subclasses and would otherwise be silently scored as 1.0/0.0.
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError(
                f"evaluate_fn score (tuple[0]) must be a real number "
                f"(int / float / numpy scalar), got {type(score).__name__}"
            )
        if not isinstance(extras, dict):
            raise TypeError(f"evaluate_fn extras (tuple[1]) must be Dict[str, float], got {type(extras).__name__}")
        reserved = set(extras) & _RESERVED_EXTRA_KEYS
        if reserved:
            raise ValueError(
                f"evaluate_fn extras must not contain reserved keys "
                f"{sorted(reserved)}; reserved key set is "
                f"{sorted(_RESERVED_EXTRA_KEYS)}"
            )
        score = float(score)
        # nan / +-inf would break: (a) `_is_better` comparisons, since
        # `nan < x` and `nan > x` are both False, so the strategy never
        # updates `best_score` and stage2 early-stops immediately; (b)
        # `json.dumps(nan)` emits non-standard JSON ("NaN") that downstream
        # tooling cannot parse back. Fail-fast on the evaluator instead.
        if not math.isfinite(score):
            raise ValueError(
                f"evaluate_fn score must be finite, got {score!r}. "
                "AutoQuant cannot rank candidates with nan / inf scores "
                "(comparisons are undefined and the value cannot be "
                "persisted as standard JSON). Fix evaluate_fn so it "
                "returns a finite numeric score."
            )
        logger.info(f"[eval] score={score} extras={extras}")

        record = save_experiment(
            run_dir=run_dir,
            index=state["exp_index"],
            strategy=strategy,
            sampled_param=sampled_param,
            export_path=espdl_export_file,
            runtime_snapshot=runtime_snapshot,
        )
        record["score"] = score
        record.update(extras)
        # Commit the "done" record (summary.json) BEFORE updating the
        # candidates pool. Rationale: a crash between these two writes
        # used to leave the new entry in candidates.json but unrecorded
        # in summary.json; the next resume would not see the hash in
        # done_hashes, re-run the same (strategy, params), and append a
        # second entry to candidates.json — Top-K would then contain
        # duplicate hashes with different index/folder. With this order,
        # the worst case becomes "summary has the record but candidates
        # missed it", which is just a one-row gap in Top-K (and the
        # missing experiment's artifacts are still recoverable from
        # <run_dir>/<index>/), not a correctness violation.
        update_summary(run_dir, {"hash": h, **record})
        update_topk_candidates(
            candidates_path,
            record,
            setting.num_of_candidates,
            key="score",
            candidate_filter=setting.candidate_filter,
            score_direction=effective_score_direction,
        )

        done_hashes.add(h)
        done_metrics[h] = score
        state["exp_index"] += 1
        return score

    if setting.search_mode == "exhaustive":
        _run_exhaustive(strategies, _eval_once)
    else:
        _run_fast(
            strategies,
            _eval_once,
            top=setting.top_strategy,
            t_max=setting.early_stop_patience,
            structural_knobs=setting.structural_knobs,
            score_direction=effective_score_direction,
        )

    tried = state["exp_index"] - next_index
    logger.info(f"AutoQuant done. Experiments tried this run: {tried}")
    return load_candidates(candidates_path)
