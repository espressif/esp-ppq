"""Top-K candidate bookkeeping for AutoQuant."""

import json
import os
from typing import Any, Callable, Dict, List, Optional

CandidateFilter = Callable[[Dict[str, Any]], bool]


def load_candidates(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_candidates(path: str, candidates: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        json.dump(candidates, f, indent=4)


def low_latency_candidate_filter(record: Dict[str, Any]) -> bool:
    """Prefer candidates without precision promotion or layer splitting."""
    s = record.get("strategy", {})
    return (not s.get("mixed_precision", False)) and (not s.get("horizontal_layer_split", False))


def _sort_by_score(records: List[Dict[str, Any]], key: str, score_direction: str) -> List[Dict[str, Any]]:
    """Sort records by ``key`` according to ``score_direction``.

    ``score_direction`` follows the Optuna convention:
        * ``"maximize"``: best first means **largest** ``key`` first
          (descending).
        * ``"minimize"``: best first means **smallest** ``key`` first
          (ascending).

    Ties on ``key`` are broken by ascending ``index`` regardless of
    ``score_direction``, so the output order is deterministic.
    """
    if score_direction not in ("maximize", "minimize"):
        raise ValueError(f"score_direction must be 'maximize' or 'minimize', got {score_direction!r}")
    # Two-pass stable sort: secondary key first, then primary key.
    out = sorted(records, key=lambda x: x.get("index", 10**12))
    out.sort(key=lambda x: x[key], reverse=(score_direction == "maximize"))
    return out


def _valid(records: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    return [c for c in records if key in c and isinstance(c[key], (int, float))]


def _balanced_topk(
    candidates: List[Dict[str, Any]],
    k: int,
    key: str,
    candidate_filter: CandidateFilter,
    score_direction: str,
) -> List[Dict[str, Any]]:
    """Balance Top-K between filter-matching and non-matching pools."""
    if k <= 0:
        return []

    valid = _valid(candidates, key)
    if not valid:
        return []

    target = min(k, len(valid))
    preferred = _sort_by_score([c for c in valid if candidate_filter(c)], key, score_direction)
    others = _sort_by_score([c for c in valid if not candidate_filter(c)], key, score_direction)

    if not others:
        return preferred[:target]
    if not preferred:
        return others[:target]

    n_pref = target // 2
    n_other = target - n_pref

    picked_pref = preferred[:n_pref]
    picked_other = others[:n_other]
    selected = picked_pref + picked_other

    if len(picked_pref) < n_pref:
        need = n_pref - len(picked_pref)
        selected += others[n_other : n_other + need]
    if len(picked_other) < n_other:
        need = n_other - len(picked_other)
        selected += preferred[n_pref : n_pref + need]

    return _sort_by_score(selected, key, score_direction)[:target]


def update_topk_candidates(
    path: str,
    record: Dict[str, Any],
    k: int,
    key: str,
    candidate_filter: Optional[CandidateFilter] = None,
    score_direction: str = "maximize",
) -> List[Dict[str, Any]]:
    """Append one record and rewrite the candidate file with the current Top-K.

    Args:
        score_direction: Optuna-style direction. ``"maximize"`` (default) keeps
            the largest scores, ``"minimize"`` keeps the smallest (use it when
            ``key`` is a loss / error metric).
    """
    candidates = load_candidates(path)
    candidates.append(record)

    if candidate_filter is None:
        valid = _valid(candidates, key)
        candidates = _sort_by_score(valid, key, score_direction)[: max(int(k), 0)]
    else:
        candidates = _balanced_topk(candidates, int(k), key, candidate_filter, score_direction)

    save_candidates(path, candidates)
    return candidates
