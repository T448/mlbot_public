import pandas as pd


def sample_logic_limit(_df: pd.DataFrame, entry_length: int, entry_point: float) -> pd.DataFrame:
    """
    過去 entry_length 分のローソク足 の high,low から entry_point[%] だけ離れたところに指値を置く

    Args:
        _df (pd.DataFrame): DatetimeIndexで、カラムがopen, high, low, close であるDataFrame
        entry_length (int): 計算に使用するwindowの長さ
        entry_point (float): high,lowからの乖離率 [%]

    Returns:
        _type_: inputにbuy, sell の指値額のカラムを追加したDataFrame
    """
    df = _df.copy()

    df["sell"] = df["high"].rolling(entry_length).max() * (1 + entry_point / 100)
    df["buy"] = df["low"].rolling(entry_length).min() * (1 - entry_point / 100)

    return df[entry_length:]
