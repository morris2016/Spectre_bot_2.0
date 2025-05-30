import asyncio
import time
import pytest

from common.metrics import MetricsCollector, calculate_timing


def test_calculate_timing_sync():
    collector = MetricsCollector("timing_sync")

    @calculate_timing(metric_name="sample", collector=collector)
    def sample_function():
        return "ok"

    result = sample_function()
    assert result == "ok"
    key = "timing_sync.sample"
    assert key in collector.timers
    assert len(collector.timers[key]) == 1
    assert collector.timers[key][0] >= 0


@pytest.mark.asyncio
async def test_calculate_timing_async():
    collector = MetricsCollector("timing_async")

    @calculate_timing(metric_name="sample_async", collector=collector)
    async def sample_async():
        await asyncio.sleep(0.01)
        return "ok"

    result = await sample_async()
    assert result == "ok"
    key = "timing_async.sample_async"
    assert key in collector.timers
    assert len(collector.timers[key]) == 1
    assert collector.timers[key][0] >= 0
