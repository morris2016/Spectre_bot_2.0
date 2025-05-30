import asyncio

from collections import deque

from config import Config
from strategy_brains.order_flow_brain import OrderFlowBrain


def _build_brain():
    cfg = Config()
    brain = OrderFlowBrain(cfg, "TEST", "1m")
    return brain


def test_is_significant_candle_volume():
    brain = _build_brain()

    # Preload volume history
    brain.delta_history = deque(
        [{"timestamp": i, "delta": 0, "close": 0, "volume": 100} for i in range(20)],
        maxlen=brain.params["memory_length"],
    )

    candle = {
        "open": 1.0,
        "close": 1.8,
        "high": 2.0,
        "low": 1.0,
        "volume": 200,
    }
    assert brain._is_significant_candle(candle)

    candle["volume"] = 50
    assert not brain._is_significant_candle(candle)


def test_analyze_transactions_volume_spike():
    brain = _build_brain()
    brain.recent_cluster_volumes.extend([50] * 10)

    now = 1000000
    cluster_big = [
        {"timestamp": now + i * 1000, "price": 1.0, "size": 100, "side": "buy"}
        for i in range(6)
    ]
    cluster_small = [
        {"timestamp": now + 20000 + i * 1000, "price": 1.0, "size": 4, "side": "buy"}
        for i in range(6)
    ]

    transactions = cluster_big + cluster_small

    asyncio.run(brain._analyze_transactions(transactions))

    assert brain.significant_transactions
    entry = brain.significant_transactions[-1]
    assert "volume_spike" in entry["reasons"]
