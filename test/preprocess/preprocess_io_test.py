from for_test.generate_data import ohlcv_and_features_with_datetime_index
from preprocess.preprocess import calc_standard_scaler
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TestClass:

    def test_calc_preprocessing(self):
        """
        以下の2つを返すこと
        - StandardScaler (A)
        - DataFrame
            - open, high, low, close, volume の5カラムを持つ (B)
            - 上記の5カラム以外にもカラムを持つ (C)
            - preprocess 時点では signal は未計算を想定、そのため buy, sell カラムがない (D)
            - DatetimeIndex を持つ (E)
        """
        df = ohlcv_and_features_with_datetime_index()
        ss, df_standard = calc_standard_scaler(df)

        expected_columns = ["open", "high", "low", "close", "volume"]
        excluded_columns = ["buy", "sell"]
        actual_column_set = set(df_standard.columns)

        actual_index_type = type(df_standard.index)

        # (A)
        assert type(ss) is StandardScaler
        # (B)
        assert sum([i in actual_column_set for i in expected_columns]) == len(expected_columns)
        # (C)
        assert len(actual_column_set) > len(expected_columns)
        # (D)
        assert sum([i in actual_column_set for i in excluded_columns]) == 0
        # (E)
        assert actual_index_type is pd.DatetimeIndex
