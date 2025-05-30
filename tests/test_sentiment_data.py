import tempfile
import shutil

from feature_service.features.sentiment import SentimentFeatures
from data_storage.time_series import TimeSeriesStorage


def test_get_sentiment_data_fallback():
    temp_dir = tempfile.mkdtemp()
    try:
        TimeSeriesStorage._instance = None
        ts = TimeSeriesStorage.get_instance({'backend': 'pandas', 'data_path': temp_dir})
        features = SentimentFeatures(ts)
        data = features._get_sentiment_data('BTCUSDT', 4, ['twitter'])
        assert 'twitter' in data
        assert isinstance(data['twitter'], list)
        assert len(data['twitter']) > 0
    finally:
        shutil.rmtree(temp_dir)

