"""Strategy and parameter-space expansion for AutoQuant."""

import itertools
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, List, Tuple


def _range_round_digits(*values: Any) -> int:
    """Return enough decimal places to preserve range inputs while hiding float drift."""
    digits = 0
    for value in values:
        exponent = Decimal(str(value)).as_tuple().exponent
        if exponent < 0:
            digits = max(digits, -exponent)
    return max(digits, 12)


class StrategyCompiler:
    """Expand ``strategy_space`` and ``param_space`` into concrete strategies."""

    def __init__(self, strategy_space: Dict[str, List[bool]], param_space: Dict[str, Dict[str, Any]]) -> None:
        self.strategy_space = strategy_space
        self.param_space = param_space

    def _expand_param(self, name: str) -> List[Dict[str, Any]]:
        """Expand one module's parameter candidates."""
        if name not in self.param_space:
            return [{}]

        raw = self.param_space[name]
        keys = list(raw.keys())
        dim_values: List[List[Any]] = []

        for k in keys:
            v = raw[k]
            if isinstance(v, list):
                dim_values.append(v)
            elif isinstance(v, tuple):
                start, end, step = v
                if step <= 0:
                    raise ValueError(f"Step must be positive for range param {name}.{k}, got step={step}")
                ndigits = _range_round_digits(start, end, step)
                tolerance = abs(step) * 1e-12
                samples: List[Any] = []
                idx = 0
                while True:
                    value = start + idx * step
                    if value > end + tolerance:
                        break
                    samples.append(round(value, ndigits))
                    idx += 1
                dim_values.append(samples)
            else:
                raise TypeError(f"Unsupported param format: {name}.{k}={v!r}; must be list or (start, end, step)")

        return [dict(zip(keys, combo)) for combo in itertools.product(*dim_values)]

    def build(self) -> List[Dict[str, dict]]:
        """Return the fully expanded strategy list."""
        expanded: List[Dict[str, dict]] = []
        keys = list(self.strategy_space.keys())
        values = list(self.strategy_space.values())

        for combo in itertools.product(*values):
            base = dict(zip(keys, combo))
            strategy: Dict[str, dict] = {}
            for k, v in base.items():
                if v is False:
                    strategy[k] = {"value": False, "param_candidates": None}
                else:
                    strategy[k] = {
                        "value": v,
                        "param_candidates": self._expand_param(k),
                    }
            expanded.append(strategy)
        return expanded


class StrategyRules:
    """Simple rule list for pruning invalid strategies."""

    def __init__(self) -> None:
        self.rules: List[Callable[[Dict[str, dict]], bool]] = []

    def add(self, rule_fn: Callable[[Dict[str, dict]], bool]) -> None:
        self.rules.append(rule_fn)

    def valid(self, strategy: Dict[str, dict]) -> bool:
        return all(rule(strategy) for rule in self.rules)


class ExhaustiveStrategySearcher:
    """Yield strategies that satisfy every registered rule."""

    def __init__(
        self,
        strategies: List[Dict[str, dict]],
        rules: "StrategyRules | None" = None,
    ) -> None:
        self.strategies = strategies
        self.rules = rules or StrategyRules()

    def search(self) -> Iterable[Dict[str, dict]]:
        for strategy in self.strategies:
            if self.rules.valid(strategy):
                yield strategy


def build_search_pipeline(
    strategy_space: Dict[str, List[bool]],
    param_space: Dict[str, Dict[str, Any]],
) -> Tuple[ExhaustiveStrategySearcher, StrategyRules]:
    """Build a searcher and its mutable rule set."""
    compiler = StrategyCompiler(strategy_space, param_space)
    rules = StrategyRules()
    searcher = ExhaustiveStrategySearcher(compiler.build(), rules)
    return searcher, rules
