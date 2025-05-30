import asyncio
import sys
import types

fake_exceptions = types.ModuleType("common.exceptions")


def _getattr(name):
    return Exception


fake_exceptions.__getattr__ = _getattr
sys.modules.setdefault("common.exceptions", fake_exceptions)

from intelligence.loophole_detection.microstructure import MicrostructureAnalyzer  # noqa: E402


fake_cross_asset = types.ModuleType("feature_service.features.cross_asset")
fake_cross_asset.compute_pair_correlation = lambda *args, **kwargs: 0.0
fake_cross_asset.cointegration_score = lambda *args, **kwargs: 0.0
sys.modules.setdefault("feature_service.features.cross_asset", fake_cross_asset)

from intelligence.loophole_detection.microstructure import MicrostructureAnalyzer  # noqa: E402


from intelligence.loophole_detection.microstructure import MicrostructureAnalyzer  # noqa: E402

# Patch problematic cross_asset module used during MicrostructureAnalyzer import
fake_cross_asset = types.ModuleType("feature_service.features.cross_asset")
sys.modules.setdefault("feature_service.features.cross_asset", fake_cross_asset)


from intelligence.loophole_detection.microstructure import MicrostructureAnalyzer  # noqa: E402

def _getattr(name):
    return Exception

fake_exceptions.__getattr__ = _getattr
sys.modules.setdefault("common.exceptions", fake_exceptions)

from intelligence.loophole_detection.microstructure import MicrostructureAnalyzer

class DummyRepo:
    pass


class DummyOrderFlow:
    pass


class DummyVolume:
    pass


def _build_analyzer():
    cfg = {
        "liquidity_trap_threshold": 0.001,
        "liquidity_trap_window": 5,
        "trap_volume_ratio": 0.6,
    }
    return MicrostructureAnalyzer(DummyRepo(), DummyOrderFlow(), DummyVolume(), cfg)


def _generate_trades(prices, volumes):
    trades = []
    for price, volume in zip(prices, volumes):
        trades.append({"price": price, "amount": volume, "side": "buy"})

    for p, v in zip(prices, volumes):
        trades.append({"price": p, "amount": v, "side": "buy"})
    return trades


def test_detect_liquidity_trap():
    analyzer = _build_analyzer()

    # Pre-window trades around price 100
    pre_prices = [100.0 for _ in range(5)]
    pre_volumes = [10 for _ in range(5)]

    # Trap trades spike then revert with lower volume
    trap_prices = [100.5, 100.6, 100.0, 100.0, 100.1]
    trap_volumes = [3 for _ in range(5)]

    trades = _generate_trades(pre_prices + trap_prices, pre_volumes + trap_volumes)
    asyncio.run(analyzer.update_trades("BTCUSDT", trades))
    signals = asyncio.run(analyzer._detect_liquidity_traps("BTCUSDT"))
    assert signals
    assert signals[0].signal_type == "liquidity_trap"
