import pandas as pd
import numpy as np
from common.ta_candles import cdl_pattern

from feature_service.feature_extraction import FeatureExtractor


def build_sample_df():
    close = [100 + 0.5 * i for i in range(30)]
    data = pd.DataFrame({
        "open": [c - 0.2 for c in close],
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
        "volume": [1000 + i * 10 for i in range(30)],
    })
    return data


EXPECTED_SMA = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan, 103.25, 103.75, 104.25,
                104.75, 105.25, 105.75, 106.25, 106.75, 107.25, 107.75, 108.25,
                108.75, 109.25, 109.75, 110.25, 110.75, 111.25]

EXPECTED_EMA = [100.0, 100.066667, 100.191111, 100.36563, 100.583546, 100.839073,
                101.127197, 101.44357, 101.784428, 102.146504, 102.52697,
                102.923374, 103.333591, 103.755779, 104.188342, 104.629896,
                105.079243, 105.535344, 105.997298, 106.464325, 106.935748,
                107.410982, 107.889518, 108.370915, 108.854793, 109.340821,
                109.828711, 110.318217, 110.809121, 111.301238]

EXPECTED_RSI = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 100, 100, 100,
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

EXPECTED_MACD = [0.0, 0.039886, 0.110567, 0.20457, 0.315774, 0.439187, 0.570759,
                 0.707225, 0.845968, 0.984916, 1.12244, 1.257282, 1.388485,
                 1.515342, 1.637349, 1.754165, 1.865583, 1.971503, 2.071907,
                 2.166845, 2.25642, 2.340772, 2.420069, 2.494504, 2.564281,
                 2.629613, 2.690718, 2.747814, 2.80112, 2.850848]

EXPECTED_ATR = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0,
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]


def test_indicator_calculations_match_expected():
    df = build_sample_df()
    extractor = FeatureExtractor(["sma", "ema", "rsi", "macd", "atr"])
    result = extractor.extract_features(df)

    assert np.allclose(result["sma"].values, EXPECTED_SMA, equal_nan=True)
    assert np.allclose(result["ema"].values, EXPECTED_EMA, equal_nan=True)
    assert np.allclose(result["rsi"].values, EXPECTED_RSI, equal_nan=True)
    assert np.allclose(result["macd"].values, EXPECTED_MACD, equal_nan=True)
    assert np.allclose(result["atr"].values, EXPECTED_ATR, equal_nan=True)


def test_cdl_pattern_detection():
    df = pd.DataFrame({
        "open": [1.0, 1.2],
        "high": [1.5, 1.3],
        "low": [0.5, 1.1],
        "close": [1.0, 1.25],
        "volume": [100, 120],
    })
    pattern_series = cdl_pattern(df, name="doji")
    assert isinstance(pattern_series, pd.Series)
    assert pattern_series.iloc[0] != 0

