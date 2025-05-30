import pandas as pd
from feature_service.processor_utils import cudf


def test_merge():
    left = pd.DataFrame({'id': [1, 2, 3], 'a': [10, 20, 30]})
    right = pd.DataFrame({'id': [2, 3, 4], 'b': [200, 300, 400]})
    expected = pd.merge(left, right, on='id', how='inner')
    result = cudf.merge(left, right, on='id', how='inner')
    pd.testing.assert_frame_equal(result, expected)


def test_join():
    left = pd.DataFrame({'id': [1, 2, 3], 'a': [10, 20, 30]}).set_index('id')
    right = pd.DataFrame({'id': [2, 3, 4], 'b': [200, 300, 400]}).set_index('id')
    expected = left.join(right, how='left')
    result = cudf.join(left, right, how='left')
    pd.testing.assert_frame_equal(result, expected)


def test_groupby():
    df = pd.DataFrame({'key': ['A', 'B', 'A', 'B'], 'val': [1, 2, 3, 4]})
    expected = df.groupby('key')['val'].sum()
    result = cudf.groupby(df, 'key')['val'].sum()
    pd.testing.assert_series_equal(result, expected)

