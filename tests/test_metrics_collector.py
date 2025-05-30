import pytest

from common.metrics import MetricsCollector


def test_metrics_collector_namespace_with_subsystem():
    collector = MetricsCollector(namespace="test", subsystem="unit")
    assert collector.namespace == "test.unit"
    collector.increment("example")
    assert collector.counters["test.unit.example"] == 1
