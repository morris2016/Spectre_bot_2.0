"""Utility functions for basic candlestick pattern detection without pandas_ta."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_series(df: pd.DataFrame | None, open_, high, low, close):
    if df is not None:
        open_ = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
    if any(v is None for v in [open_, high, low, close]):
        raise ValueError('OHLC data required')
    return open_, high, low, close


def _doji(open_, high, low, close):
    body = (close - open_).abs()
    rng = high - low
    cond = (rng != 0) & (body / rng <= 0.1)
    return pd.Series(np.where(cond, 100, 0), index=open_.index)


def _hammer(open_, high, low, close, bullish=True):
    body = (close - open_).abs()
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    cond = (lower >= 2 * body) & (upper <= body)
    if bullish:
        cond &= close > open_
        val = 100
    else:
        cond &= close < open_
        val = -100
    return pd.Series(np.where(cond, val, 0), index=open_.index)


def _inverted_hammer(open_, high, low, close, bullish=True):
    body = (close - open_).abs()
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    cond = (upper >= 2 * body) & (lower <= body)
    if bullish:
        cond &= close > open_
        val = 100
    else:
        cond &= close < open_
        val = -100
    return pd.Series(np.where(cond, val, 0), index=open_.index)


def _marubozu(open_, high, low, close):
    body = (close - open_).abs()
    rng = high - low
    cond = (rng > 0) & (body / rng >= 0.9)
    sign = np.sign(close - open_)
    return pd.Series(np.where(cond, 100 * sign, 0), index=open_.index)


def _spinning_top(open_, high, low, close):
    body = (close - open_).abs()
    rng = high - low
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    cond = (rng > 0) & (body / rng <= 0.4) & (upper > body) & (lower > body)
    sign = np.sign(close - open_)
    return pd.Series(np.where(cond, 100 * sign, 0), index=open_.index)


def _engulfing(open_, high, low, close):
    result = np.zeros(len(open_))
    for i in range(1, len(open_)):
        prev_o, prev_c = open_.iat[i-1], close.iat[i-1]
        curr_o, curr_c = open_.iat[i], close.iat[i]
        if curr_c > curr_o and prev_c < prev_o and curr_o <= prev_c and curr_c >= prev_o:
            result[i] = 100
        elif curr_c < curr_o and prev_c > prev_o and curr_o >= prev_c and curr_c <= prev_o:
            result[i] = -100
    return pd.Series(result, index=open_.index)


def _harami(open_, high, low, close):
    result = np.zeros(len(open_))
    for i in range(1, len(open_)):
        prev_o, prev_c = open_.iat[i-1], close.iat[i-1]
        curr_o, curr_c = open_.iat[i], close.iat[i]
        prev_body_high = max(prev_o, prev_c)
        prev_body_low = min(prev_o, prev_c)
        curr_body_high = max(curr_o, curr_c)
        curr_body_low = min(curr_o, curr_c)
        if (curr_body_high <= prev_body_high and curr_body_low >= prev_body_low and curr_c > curr_o and prev_c < prev_o):
            result[i] = 100
        elif (curr_body_high <= prev_body_high and curr_body_low >= prev_body_low and curr_c < curr_o and prev_c > prev_o):
            result[i] = -100
    return pd.Series(result, index=open_.index)


def _piercing_line(open_, high, low, close):
    result = np.zeros(len(open_))
    for i in range(1, len(open_)):
        if close.iat[i-1] < open_.iat[i-1] and close.iat[i] > open_.iat[i]:
            midpoint = (open_.iat[i-1] + close.iat[i-1]) / 2
            if open_.iat[i] < close.iat[i-1] and close.iat[i] > midpoint:
                result[i] = 100
    return pd.Series(result, index=open_.index)


def _dark_cloud_cover(open_, high, low, close):
    result = np.zeros(len(open_))
    for i in range(1, len(open_)):
        if close.iat[i-1] > open_.iat[i-1] and close.iat[i] < open_.iat[i]:
            midpoint = (open_.iat[i-1] + close.iat[i-1]) / 2
            if open_.iat[i] > close.iat[i-1] and close.iat[i] < midpoint:
                result[i] = -100
    return pd.Series(result, index=open_.index)


def _tweezer_bottom(open_, high, low, close):
    result = np.zeros(len(open_))
    for i in range(1, len(open_)):
        if close.iat[i-1] < open_.iat[i-1] and close.iat[i] > open_.iat[i]:
            if abs(low.iat[i-1] - low.iat[i]) <= (high.iat[i-1] - low.iat[i-1]) * 0.1:
                result[i] = 100
    return pd.Series(result, index=open_.index)


def _tweezer_top(open_, high, low, close):
    result = np.zeros(len(open_))
    for i in range(1, len(open_)):
        if close.iat[i-1] > open_.iat[i-1] and close.iat[i] < open_.iat[i]:
            if abs(high.iat[i-1] - high.iat[i]) <= (high.iat[i-1] - low.iat[i-1]) * 0.1:
                result[i] = -100
    return pd.Series(result, index=open_.index)


_PATTERN_MAP = {
    'doji': _doji,
    'hammer': lambda o, h, l, c: _hammer(o, h, l, c, True),
    'hanging_man': lambda o, h, l, c: _hammer(o, h, l, c, False),
    'inverted_hammer': lambda o, h, l, c: _inverted_hammer(o, h, l, c, True),
    'shooting_star': lambda o, h, l, c: _inverted_hammer(o, h, l, c, False),
    'marubozu': _marubozu,
    'spinning_top': _spinning_top,
    'engulfing': _engulfing,
    'harami': _harami,
    'piercing_line': _piercing_line,
    'dark_cloud_cover': _dark_cloud_cover,
    'tweezer_bottom': _tweezer_bottom,
    'tweezer_top': _tweezer_top,
}


def cdl_pattern(df: pd.DataFrame | None = None, *, open_=None, high=None, low=None, close=None, name: str) -> pd.Series:
    """Basic replacement for pandas_ta.cdl_pattern."""
    open_, high, low, close = _get_series(df, open_, high, low, close)
    func = _PATTERN_MAP.get(name.lower())
    if func is None:
        return pd.Series(np.zeros(len(open_)), index=open_.index)
    return func(open_, high, low, close)
