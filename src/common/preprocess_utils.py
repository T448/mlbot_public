import pandas as pd


def get_label(df: pd.DataFrame, binary=True):
    # 指値位置補正
    df["limit"] = df[["open"]].shift(-1)
    df["buy"] = df[["buy", "limit"]].min(axis="columns")
    df["sell"] = df[["sell", "limit"]].max(axis="columns")

    # 約定判断
    # ヒットしなければすぐ削除
    df["long"] = df["low"] < df["buy"].shift(1)
    df["short"] = df["high"] > df["sell"].shift(1)

    # 約定価格
    # 約定しなければ0
    df["exec_price_buy"] = df["buy"].where(df["long"].shift(-1), 0)
    df["exec_price_sell"] = df["sell"].where(df["short"].shift(-1), 0)

    # ラベル
    # 約定し、次の足で利益になる場合1
    df["y_buy"] = 2 * (df["exec_price_buy"] < df["close"].shift(-1)) - 1
    df["y_buy"] = df["y_buy"] * (1 * df["exec_price_buy"] != 0)

    df["y_sell"] = 2 * df["exec_price_sell"] > df["close"].shift(-1) - 1
    df["y_sell"] = df["y_sell"] * (1 * (df["exec_price_sell"] != 0))

    if binary:
        # 2値化
        df["y_buy"] = 1 * (df["y_buy"] == 1)
        df["y_sell"] = 1 * (df["y_sell"] == 1)

    return df.dropna()
