import math
from decimal import Decimal
import numpy as np
from common.utils import (
    round_to_tick_size,
    dict_to_namedtuple,
    calculate_volatility,
    calculate_correlation,
    calculate_drawdown,
    calculate_liquidation_price,
    calculate_arbitrage_profit,
    calculate_position_size,
    calculate_correlation_matrix,
    calculate_risk_reward,
    calculate_confidence_score,
    normalize_probability,
    weighted_average,
    time_weighted_average,
    validate_signal,
)


def test_round_to_tick_size_decimal():
    assert round_to_tick_size(Decimal("10.125"), Decimal("0.01")) == Decimal("10.13")


def test_dict_to_namedtuple():
    nt = dict_to_namedtuple("Test", {"a": 1, "b": 2})
    assert nt.a == 1 and nt.b == 2


def test_calculate_volatility():
    prices = [100, 101, 102, 103, 104]
    # manual calculation
    returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    mean = sum(returns) / len(returns)
    variance = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
    expected = math.sqrt(variance) * math.sqrt(len(returns))
    result = calculate_volatility(prices)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_calculate_correlation():
    s1 = [1, 2, 3, 4, 5]
    s2 = [2, 4, 6, 8, 10]
    assert math.isclose(calculate_correlation(s1, s2), 1.0, rel_tol=1e-9)


def test_calculate_correlation_matrix():
    data = {
        "A": np.array([1, 2, 3]),
        "B": np.array([2, 4, 6]),
        "C": np.array([1, 0, 1]),
    }
    matrix = calculate_correlation_matrix(data)
    assert math.isclose(matrix.loc["A", "B"], 1.0, rel_tol=1e-9)


def test_calculate_drawdown():
    equity = [100, 120, 110, 105, 130]
    max_dd, current_dd = calculate_drawdown(equity)
    assert math.isclose(max_dd, 12.5, rel_tol=1e-9)
    assert math.isclose(current_dd, 0.0, rel_tol=1e-9)


def test_calculate_liquidation_price():
    assert math.isclose(calculate_liquidation_price("long", 100, 10), 90.5, rel_tol=1e-9)
    assert math.isclose(calculate_liquidation_price("short", 100, 10), 109.5, rel_tol=1e-9)


def test_calculate_arbitrage_profit():
    profit = calculate_arbitrage_profit(100, 105, 10, 0.001, 0.001)
    assert math.isclose(profit, 47.95, rel_tol=1e-9)


def test_calculate_position_size():
    size = calculate_position_size(1000, 0.02, 0.05)
    assert math.isclose(size, 400.0, rel_tol=1e-9)


def test_calculate_risk_reward():
    assert math.isclose(calculate_risk_reward(100, 95, 110), 2.0, rel_tol=1e-9)
    assert math.isclose(
        calculate_risk_reward("sell", 100, 105, 95), 2.0, rel_tol=1e-9
    )


def test_calculate_confidence_score():
    votes = {"buy": 0.6, "sell": 0.4}
    reasoning = {
        "c1": {"action": "buy", "confidence": 0.8},
        "c2": {"action": "sell", "confidence": 0.6},
    }
    result = calculate_confidence_score(votes, reasoning)
    assert math.isclose(result, 0.72, rel_tol=1e-9)


def test_normalize_probability():
    assert math.isclose(normalize_probability(0.75), 0.75, rel_tol=1e-9)
    assert math.isclose(normalize_probability(75), 0.75, rel_tol=1e-9)
    assert math.isclose(normalize_probability(-5), 0.0, rel_tol=1e-9)
    assert math.isclose(normalize_probability(150), 1.0, rel_tol=1e-9)


def test_weighted_average():
    values = [1, 2, 3]
    weights = [1, 1, 2]
    assert math.isclose(weighted_average(values, weights), 2.25, rel_tol=1e-9)


def test_time_weighted_average():
    values = [10, 20, 30]
    times = [0, 1, 3]
    assert math.isclose(time_weighted_average(values, times), 22.5, rel_tol=1e-9)


def test_validate_signal():
    valid_signal = {
        "symbol": "BTCUSD",
        "action": "buy",
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "take_profit": 120.0,
        "confidence": 0.8,
    }
    assert validate_signal(valid_signal)

    invalid_signal = {"symbol": "BTCUSD", "confidence": 1.2}
    assert not validate_signal(invalid_signal)
