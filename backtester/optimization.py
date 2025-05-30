from typing import Any, Dict, List, Optional, Callable

class ParameterOptimizer:
    """Simplified parameter optimizer used by walk forward analysis."""
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    async def optimize(
        self,
        strategy_id: str,
        base_parameters: Dict[str, Any],
        target_parameters: List[str],
        start_date: str,
        end_date: str,
        assets: List[str],
        platform: str,
        optimization_metric: str = "sharpe_ratio",
    ) -> Dict[str, Any]:
        # Placeholder implementation returning base parameters
        return {
            "best_parameters": base_parameters,
            "best_metrics": {},
            "best_equity_curve": [],
        }

class Optimization:
    """General optimization entry point used by the backtester service."""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    async def optimize(
        self,
        data: Any,
        strategy_name: str,
        param_space: Dict[str, Any],
        method: str,
        objective: str,
        n_trials: int,
        engine: Any,
        performance: Any,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        # Minimal stub selecting first value of each parameter
        best_params = {k: (v[0] if isinstance(v, list) else v) for k, v in param_space.items()}
        if progress_callback:
            progress_callback(1.0)
        return {"best_params": best_params, "best_value": 0.0, "all_trials": []}
