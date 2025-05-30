import pandas as pd
from ml_models.models.time_series import ProphetModel, TimeSeriesConfig

def build_series():
    dates = pd.date_range('2022-01-01', periods=20)
    values = pd.Series(range(20), index=dates)
    return pd.DataFrame({'y': values})

def test_prophet_fit_save_load(tmp_path):
    df = build_series()
    cfg = TimeSeriesConfig()
    model = ProphetModel(cfg)
    model.fit(df)
    model.save(tmp_path)

    loaded = ProphetModel(cfg).load(tmp_path)
    preds = loaded.predict(2)
    assert len(preds) == 2

