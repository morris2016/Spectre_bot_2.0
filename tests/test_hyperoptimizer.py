import asyncio
import pandas as pd

from config import Config
from ml_models.hyperopt import HyperparameterOptimizer
from ml_models.training import ModelTrainer


def test_hyperoptimizer_grid():
    opt = HyperparameterOptimizer(direction="maximize")

    def objective(params):
        return params["x"]

    best = opt.optimize(objective, {"x": [1, 2, 3]}, method="grid")
    assert best["x"] == 3


def test_trainer_integration():
    cfg = Config({"ml_models": {"optimize": True, "optimize_method": "grid", "optimize_trials": 1}})
    trainer = ModelTrainer(cfg)
    X = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])

    params = asyncio.run(
        trainer._get_hyperparameters(
            "random_forest",
            None,
            True,
            X,
            y,
            X,
            y,
            "accuracy",
        )
    )
    assert "n_estimators" in params
