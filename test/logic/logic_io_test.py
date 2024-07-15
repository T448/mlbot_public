from for_test.generate_data import ohlcv_with_datetime_index
from logic.sample_logic_limit import sample_logic_limit
import pandas as pd


class TestClass:

    def test_sample_logic_limit(self):
        """
        以下の項目を満たすDataFrameを返すこと
        - open, high, low, close, volume, buy, sell の7カラムを持つ
        - DatetimeIndex を持つ
        """
        df = ohlcv_with_datetime_index()
        df_signal = sample_logic_limit(df, 1, 0.1)
        expected_columns = ["open", "high", "low", "close", "volume", "buy", "sell"]
        actual_columns = df_signal.columns
        actual_column_set = set(actual_columns)

        actual_index_type = type(df_signal.index)

        assert sum([i in actual_column_set for i in expected_columns]) == len(expected_columns)
        assert actual_index_type is pd.DatetimeIndex
