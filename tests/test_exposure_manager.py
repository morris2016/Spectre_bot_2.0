from decimal import Decimal

import pandas as pd
import pytest

from common.constants import Exchange
from risk_manager.exposure import ExposureManager, MarketDataRepository


@pytest.mark.asyncio
async def test_update_exposure_with_market_repo(monkeypatch):
    async def dummy_get_ohlcv_data(self, asset_id, timeframe, *args, **kwargs):
        closes = [100 + i for i in range(5)]
        return pd.DataFrame({'close': closes})

    monkeypatch.setattr(MarketDataRepository, 'get_ohlcv_data', dummy_get_ohlcv_data)

    manager = ExposureManager({'correlation_lookback': 5})

    positions = [
        {'symbol': 'BTCUSDT', 'platform': Exchange.BINANCE, 'position_value': Decimal('100')},
        {'symbol': 'ETHUSDT', 'platform': Exchange.BINANCE, 'position_value': Decimal('50')},
    ]
    balances = {Exchange.BINANCE: Decimal('200'), Exchange.DERIV: Decimal('0')}

    result = await manager.update_exposure(positions, balances)

    assert result['total'] == Decimal('150')
    assert result['percentage']['total'] == Decimal('0.75')
    assert result['assets']['BTCUSDT'] == Decimal('100')
    assert result['assets']['ETHUSDT'] == Decimal('50')
