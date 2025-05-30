#!/usr/bin/env python3
"""Unified hyperparameter optimization service."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable
import itertools
import logging
try:
    import optuna  # type: ignore
    OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore
    OPTUNA_AVAILABLE = False
from common.metrics import MetricsCollector
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    _HYPEROPT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    fmin = tpe = hp = Trials = STATUS_OK = None
    _HYPEROPT_AVAILABLE = False


class HyperOptService:
    """Run hyperparameter optimization using Hyperopt."""

    def __init__(self, max_evals: int = 50):
        if not _HYPEROPT_AVAILABLE:
            raise ImportError("hyperopt is required for HyperOptService")
        self.max_evals = max_evals

    def optimize(self, objective_fn, search_space: Dict[str, Any]) -> Dict[str, Any]:
        trials = Trials()
        best_params = fmin(
            fn=objective_fn,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
        )
        # Convert ints
        final_params: Dict[str, Any] = {}
        for k, v in best_params.items():
            if isinstance(v, float) and k.startswith("n_"):
                final_params[k] = int(v)
            else:
                final_params[k] = v
        return final_params


logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Orchestrate hyperparameter search using grid search or Optuna."""

    def __init__(
        self,
        metrics: MetricsCollector | None = None,
        direction: str = "maximize",
    ) -> None:
        self.metrics = metrics or MetricsCollector("ml_models.hyperopt")
        self.direction = direction

    def _iter_grid(self, search_space: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        keys = list(search_space.keys())
        for values in itertools.product(*(search_space[k] for k in keys)):
            yield dict(zip(keys, values))

    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Any],
        method: str = "bayesian",
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Run optimization.

        Args:
            objective: Function evaluating a parameter set and returning a score.
            search_space: Parameter ranges.
            method: "grid" or "bayesian".
            n_trials: Number of trials for Bayesian search.
        """
        if method == "grid":
            best_score = float("-inf") if self.direction == "maximize" else float("inf")
            best_params: Dict[str, Any] | None = None
            for params in self._iter_grid(search_space):
                score = objective(params)
                self.metrics.gauge("trial_score", score)
                self.metrics.increment("trials_total")
                if (
                    self.direction == "maximize" and score > best_score
                ) or (
                    self.direction == "minimize" and score < best_score
                ):
                    best_score = score
                    best_params = params
            logger.info("Grid search best params: %s score %.4f", best_params, best_score)
            return best_params or {}

        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required for Bayesian optimization")

        def _objective(trial: optuna.Trial) -> float:
            params = {}
            for name, values in search_space.items():
                if isinstance(values, list):
                    params[name] = trial.suggest_categorical(name, values)
                elif isinstance(values, tuple) and len(values) == 2:
                    low, high = values
                    if isinstance(low, int) and isinstance(high, int):
                        params[name] = trial.suggest_int(name, low, high)
                    else:
                        params[name] = trial.suggest_float(name, low, high)
                else:
                    raise ValueError(f"Unsupported search space for {name}: {values}")
            score = objective(params)
            self.metrics.observe("trial_score", score)
            return -score if self.direction == "minimize" else score

        study = optuna.create_study(direction=self.direction)
        study.optimize(_objective, n_trials=n_trials)
        self.metrics.increment("trials_total", len(study.trials))
        logger.info("Optuna best params: %s score %.4f", study.best_params, study.best_value)
        return study.best_params
