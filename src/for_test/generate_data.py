# TODO 本来はtestコード用フォルダに配置すべきだが、PATHの設定がうまくいかなかったためsrc配下に置いている。修正方法が分かり次第移動する。
import numpy as np
import pandas as pd
import datetime


def ohlcv_with_datetime_index() -> pd.DataFrame:
    """
    OHLCVの5カラムとdatetime_indexを持つpandas.DataFrameを生成する

    Returns:
        pd.DataFrame:
    """
    columns = ["open", "high", "low", "close", "volume"]
    start = datetime.datetime.strptime("2024/01/01 00:00:00", "%Y/%m/%d %H:%M:%S")
    index = pd.DatetimeIndex([start + datetime.timedelta(minutes=i * 15) for i in range(10)])
    data = np.full((len(index), len(columns)), 1)
    df = pd.DataFrame(data, columns=columns, index=index)

    return df


def ohlcv_and_features_with_datetime_index() -> pd.DataFrame:
    """
    OHLCVとダミー特徴量の計10カラムとdatetime_indexを持つpandas.DataFrameを生成する

    Returns:
        pd.DataFrame:
    """
    columns = ["open", "high", "low", "close", "volume", "feature0", "feature1", "feature2", "feature3", "feature4"]
    start = datetime.datetime.strptime("2024/01/01 00:00:00", "%Y/%m/%d %H:%M:%S")
    index = pd.DatetimeIndex([start + datetime.timedelta(minutes=i * 15) for i in range(10)])
    data = np.full((len(index), len(columns)), 1)
    df = pd.DataFrame(data, columns=columns, index=index)

    return df
