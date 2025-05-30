from typing import Any, Dict, List, Optional
import pandas as pd

class PerformanceAnalyzer:
    """Simple performance metrics calculator."""

    def compute(self, trades: pd.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        metrics = metrics or ["total_return", "win_rate"]
        result: Dict[str, Any] = {}
        if trades.empty:
            for m in metrics:
                result[m] = 0
            return result

        if "profit_pct" in trades.columns:
            if "total_return" in metrics:
                result["total_return"] = trades["profit_pct"].sum()
            if "win_rate" in metrics:
                result["win_rate"] = float((trades["profit_pct"] > 0).mean() * 100)
        return result

class Performance:
    """Wrapper providing async API for performance analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.analyzer = PerformanceAnalyzer()

    async def analyze(self, backtest_result: Dict[str, Any], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        trades = pd.DataFrame(backtest_result.get("trades", []))
        return self.analyzer.compute(trades, metrics)
