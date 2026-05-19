"""Search-space settings for ESP-DL AutoQuant."""

from typing import Any, Dict, List, Optional

from .candidates import CandidateFilter, low_latency_candidate_filter


class AutoQuantSearchSetting:
    """Search-space and search-behavior setting for AutoQuant.

    Fields fall into three groups:

    * **Search space** (``strategy_space`` / ``param_space``): large dicts
      initialized inside ``__init__``. Users typically modify one entry
      (e.g. ``setting.strategy_space["mixed_precision"] = [True, False]``)
      rather than replace the whole dict, so they are not exposed as
      constructor arguments.
    * **Universal search behavior** (``num_of_candidates``, ``search_mode``,
      ``candidate_filter``, ``score_direction``, ``run_dir``, ``resume``):
      simple scalars / callables that affect both ``"exhaustive"`` and
      ``"fast"`` search modes. They are exposed as constructor arguments
      so a one-liner can configure the whole search:

      .. code-block:: python

          setting = AutoQuantSearchSetting(
              search_mode="fast",
              num_of_candidates=10,
              run_dir="outputs/my_run",
          )

      The execution device is configured separately via the top-level
      ``espdl_auto_quantize_onnx(..., device=...)`` argument; that
      argument also overwrites ``param_space[...]["collecting_device"]``
      for tqt / blockwise_reconstruction at the start of the search.

    * **fast-only knobs** (``top_strategy`` / ``early_stop_patience`` /
      ``structural_knobs``): only used when ``search_mode == "fast"``. Set
      them via attribute assignment after enabling fast mode:

      .. code-block:: python

          setting = AutoQuantSearchSetting(search_mode="fast")
          setting.top_strategy = 20
          setting.early_stop_patience = 5

    Each behavior field can still be reassigned after construction (e.g.
    ``setting.search_mode = "exhaustive"``); the entry-point re-validates
    the universal fields at the start of the search.

    ``topk`` validation rules (enforced at search time):

    * ``bias_correction`` / ``tqt`` / ``blockwise_reconstruction``: ``topk``
      must be ``-1`` or a positive int. ``-1`` applies the pass to all
      eligible layers; a positive value selects the top-K highest-error
      layers.
    * ``horizontal_layer_split`` / ``mixed_precision``: ``topk`` must be a
      positive int.
    """

    def __init__(
        self,
        num_of_candidates: int = 5,
        search_mode: str = "exhaustive",
        candidate_filter: Optional[CandidateFilter] = low_latency_candidate_filter,
        score_direction: str = "maximize",
        run_dir: str = "outputs/auto_quant",
        resume: bool = False,
    ) -> None:
        # Per-module switch candidates.
        self.strategy_space: Dict[str, List[bool]] = {
            "num_of_bits": [True],
            "calib_algorithm": [True],
            "weight_equalization": [True, False],
            "horizontal_layer_split": [False],
            "bias_correction": [True, False],
            "mixed_precision": [False],
            "tqt": [True, False],
            "fusion_alignment": [True],
            "blockwise_reconstruction": [True, False],
        }

        # Per-module parameter grid. Values are enumerated when their module is enabled.
        self.param_space: Dict[str, Dict[str, Any]] = {
            "num_of_bits": {
                "value": [8],
            },
            "calib_algorithm": {
                "method": ["kl", "percentile", "mse"],
            },
            "weight_equalization": {
                "iterations": [10, 50, 100],
                "value_threshold": [0.1, 0.5, 2],
                "opt_level": [2],
            },
            "horizontal_layer_split": {
                "topk": [1],
                "value_threshold": [0.1, 0.2],
            },
            "bias_correction": {
                "block_size": [4, 2, 1],
                "steps": [32],
                "topk": [-1, 1, 2, 3],
            },
            "mixed_precision": {
                "topk": [1],
            },
            "tqt": {
                "lr": [1e-5, 5e-5],
                "steps": [500, 1000, 2000],
                "block_size": [4, 2, 1],
                "is_scale_trainable": [True],
                "gamma": [0.0],
                "int_lambda": [0.0],
                "collecting_device": ["cpu"],
                "topk": [-1],
            },
            "fusion_alignment": {
                "elementwise_to": ["Align to Large", "Align to Output"],
                "concat_to": ["Align to Large", "Align to Output"],
                "avgpooling_to": ["Align to Input", "Align to Output"],
                "resize_to": ["Align to Input", "Align to Output"],
                "force_overlap": [False],
            },
            "blockwise_reconstruction": {
                "is_scale_trainable": [False],
                "topk": [-1, 1, 2, 3],
                "block_size": [4, 2],
                "steps": [1000],
                "lr": [1e-4],
                "gamma": [1.0],
                "collecting_device": ["cpu"],
            },
        }

        # Universal search behavior (constructor arguments; see class docstring).
        self.num_of_candidates: int = num_of_candidates
        self.search_mode: str = search_mode
        self.candidate_filter: Optional[CandidateFilter] = candidate_filter
        # `score_direction` follows Optuna convention. Set to "minimize"
        # when the score is a loss / error metric (e.g. MSE, FID, L2
        # quantization error). Must be either "maximize" or "minimize";
        # validated in `espdl_auto_quantize_onnx`.
        self.score_direction: str = score_direction

        # Run-directory bookkeeping.
        # `run_dir` holds `summary.json` / `candidates.json` / per-experiment
        # subfolders. `resume` skips experiments already recorded in
        # `summary.json` so that interrupted searches can continue.
        self.run_dir: str = run_dir
        self.resume: bool = resume

        # ------------------------------------------------------------------
        # fast-only knobs (only used when `search_mode == "fast"`).
        # Assign them after construction when opting into fast mode.
        # ------------------------------------------------------------------
        # stage1 best-strategy retention count.
        self.top_strategy: int = 10
        # stage2 patience: stop a strategy after N non-improving samples.
        self.early_stop_patience: int = 3
        # Knobs swept first in fast mode stage1 (the "structural" sub-grid).
        self.structural_knobs: Dict[str, List[str]] = {
            "calib_algorithm": ["method"],
            "num_of_bits": ["value"],
        }
