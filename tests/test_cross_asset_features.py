import numpy as np
import pandas as pd
from feature_service.features.cross_asset import compute_pair_correlation, cointegration_score


def test_pair_correlation_basic():
    s1 = pd.Series(range(30))
    s2 = pd.Series([x + 0.1 for x in range(30)])
    corr = compute_pair_correlation(s1, s2, window=5)
    assert isinstance(corr, pd.Series)
    assert corr.iloc[-1] > 0.9


def test_cointegration_score_range():
    s1 = pd.Series(range(30))
    s2 = pd.Series(range(30)) + 1
    pval = cointegration_score(s1, s2)
    assert 0 <= pval <= 1
def build_series(n=100):
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    s1 = pd.DataFrame({"close": np.arange(n)}, index=index)
    s2 = pd.DataFrame({"close": np.arange(n) + np.random.normal(0, 0.1, n)}, index=index)
    return s1, s2


def test_pair_correlation_high():
    s1, s2 = build_series()
    corr = compute_pair_correlation(s1, s2)
    assert corr > 0.99


def test_cointegration_pvalue_low():
    s1, s2 = build_series()
    pvalue = cointegration_score(s1, s2)
    assert pvalue < 0.05

import pandas as pd
import numpy as np

from feature_service.features.cross_asset import compute_pair_correlation, cointegration_score
from feature_service.feature_extraction import FeatureExtractor


def build_datasets():
    idx = pd.date_range("2022-01-01", periods=50, freq="D")
    base = pd.Series(np.arange(50), index=idx)
    df1 = pd.DataFrame({
        "open": base,
        "high": base + 1,
        "low": base - 1,
        "close": base,
        "volume": np.ones(50),
    })
    df2 = pd.DataFrame({
        "open": base * 1.5,
        "high": base * 1.5 + 1,
        "low": base * 1.5 - 1,
        "close": base * 1.5 + 0.5,
        "volume": np.ones(50) * 2,
    })
    return df1, df2


def test_cross_asset_utilities():
    df1, df2 = build_datasets()
    corr = compute_pair_correlation(df1, df2)
    pval = cointegration_score(df1, df2)
    assert corr > 0.99
    assert pval < 0.05


def test_feature_extractor_cross_asset():
    df1, df2 = build_datasets()
    extractor = FeatureExtractor(["pair_correlation", "cointegration_pvalue"])
    result = extractor.extract_features(df1, params={"pair_data": df2})
    assert result["pair_correlation"].iloc[0] > 0.99
    assert result["cointegration_pvalue"].iloc[0] < 0.05

