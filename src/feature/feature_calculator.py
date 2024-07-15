from typing import List
import pandas as pd
import numpy as np
import time
import pandas_ta as ta
from scipy.stats import rankdata


def get_feature_names() -> List[str]:
    return [
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
        "open_log_diff",
        "high_log_diff",
        "low_log_diff",
        "close_log_diff",
        "volume_log_diff",
        "SMA10",
        "SMA20",
        "SMA50",
        "SMA100",
        "SMA200",
        "EMA10",
        "EMA20",
        "EMA50",
        "EMA100",
        "EMA200",
        "REG5",
        "REG10",
        "BBL_20_1.0",
        "BBM_20_1.0",
        "BBU_20_1.0",
        "BBL_20_2.0",
        "BBM_20_2.0",
        "BBU_20_2.0",
        "BBL_20_3.0",
        "BBM_20_3.0",
        "BBU_20_3.0",
        "MOM7",
        "MOM14",
        "ROC10",
        "ROC20",
        "ATR14",
        "ATR28",
        "RSI5",
        "RSI10",
        "RSI20",
        "RCI9",
        "RCI28",
        "RCI56",
        "BB_20_1.0",
        "BB_20_2.0",
        "BB_20_3.0",
        "MACD_12_26_9",
        "MACD_h12_26_9",
        "MACDs_12_26_9",
        "CCI14",
        "CCI28",
        "SKEW10",
        "SKEW20",
        "SKEW50",
        "SKEW100",
        "SKEW200",
        "KURT10",
        "KURT20",
        "KURT50",
        "KURT100",
        "KURT200",
        # "close_normal_lag10",
        # "EMA200_lag10",
        # "RCI9_lag10",
        # "RCI28_lag10",
        # "BB_20_2.0_lag10",
        # "SKEW20_lag10",
        # "KURT20_lag10",
        # "1",
        # "close_normal^2",
        # "close_normal EMA200",
        # "close_normal RCI28",
        # "close_normal BB_20_2.0",
        # "close_normal SKEW20",
        # "close_normal KURT20",
        # "EMA200^2",
        # "EMA200 RCI28",
        # "EMA200 BB_20_2.0",
        # "EMA200 SKEW20",
        # "EMA200 KURT20",
        # "RCI28^2",
        # "RCI28 BB_20_2.0",
        # "RCI28 SKEW20",
        # "RCI28 KURT20",
        # "BB_20_2.0^2",
        # "BB_20_2.0 SKEW20",
        # "BB_20_2.0 KURT20",
        # "SKEW20^2",
        # "SKEW20 KURT20",
        # "KURT20^2",
        # "close_normal^3",
        # "close_normal^2 EMA200",
        # "close_normal^2 RCI28",
        # "close_normal^2 BB_20_2.0",
        # "close_normal^2 SKEW20",
        # "close_normal^2 KURT20",
        # "close_normal EMA200^2",
        # "close_normal EMA200 RCI28",
        # "close_normal EMA200 BB_20_2.0",
        # "close_normal EMA200 SKEW20",
        # "close_normal EMA200 KURT20",
        # "close_normal RCI28^2",
        # "close_normal RCI28 BB_20_2.0",
        # "close_normal RCI28 SKEW20",
        # "close_normal RCI28 KURT20",
        # "close_normal BB_20_2.0^2",
        # "close_normal BB_20_2.0 SKEW20",
        # "close_normal BB_20_2.0 KURT20",
        # "close_normal SKEW20^2",
        # "close_normal SKEW20 KURT20",
        # "close_normal KURT20^2",
        # "EMA200^3",
        # "EMA200^2 RCI28",
        # "EMA200^2 BB_20_2.0",
        # "EMA200^2 SKEW20",
        # "EMA200^2 KURT20",
        # "EMA200 RCI28^2",
        # "EMA200 RCI28 BB_20_2.0",
        # "EMA200 RCI28 SKEW20",
        # "EMA200 RCI28 KURT20",
        # "EMA200 BB_20_2.0^2",
        # "EMA200 BB_20_2.0 SKEW20",
        # "EMA200 BB_20_2.0 KURT20",
        # "EMA200 SKEW20^2",
        # "EMA200 SKEW20 KURT20",
        # "EMA200 KURT20^2",
        # "RCI28^3",
        # "RCI28^2 BB_20_2.0",
        # "RCI28^2 SKEW20",
        # "RCI28^2 KURT20",
        # "RCI28 BB_20_2.0^2",
        # "RCI28 BB_20_2.0 SKEW20",
        # "RCI28 BB_20_2.0 KURT20",
        # "RCI28 SKEW20^2",
        # "RCI28 SKEW20 KURT20",
        # "RCI28 KURT20^2",
        # "BB_20_2.0^3",
        # "BB_20_2.0^2 SKEW20",
        # "BB_20_2.0^2 KURT20",
        # "BB_20_2.0 SKEW20^2",
        # "BB_20_2.0 SKEW20 KURT20",
        # "BB_20_2.0 KURT20^2",
        # "SKEW20^3",
        # "SKEW20^2 KURT20",
        # "SKEW20 KURT20^2",
        # "KURT20^3",
    ]


def spearman(x, y):
    x = np.array(x)
    x = rankdata(x)
    N = len(x)

    return 1 - (6 * sum((x - y) ** 2) / (N * (N**2 - 1)))


def calc_feature(_df: pd.DataFrame, target: str = "close"):
    start = time.time()

    df = _df.copy()

    df["day_sin"] = [np.sin(i.day) for i in df.index]
    df["day_cos"] = [np.cos(i.day) for i in df.index]
    df["hour_sin"] = [np.sin(i.hour) for i in df.index]
    df["hour_cos"] = [np.cos(i.hour) for i in df.index]

    # 移動平均
    print("SMA")
    df["SMA10"] = df[target].rolling(10).mean()
    df["SMA20"] = df[target].rolling(20).mean()
    df["SMA50"] = df[target].rolling(50).mean()
    df["SMA100"] = df[target].rolling(100).mean()
    df["SMA200"] = df[target].rolling(200).mean()

    # EMA
    print("EMA")
    df["EMA10"] = df[target].ewm(10).mean()
    df["EMA20"] = df[target].ewm(20).mean()
    df["EMA50"] = df[target].ewm(50).mean()
    df["EMA100"] = df[target].ewm(100).mean()
    df["EMA200"] = df[target].ewm(200).mean()

    # モメンタム
    print("MOM")
    df["MOM7"] = ta.mom(df[target], length=7)
    df["MOM14"] = ta.mom(df[target], length=14)

    # ROC
    print("ROC")
    df["ROC10"] = ta.roc(df[target], length=10)
    df["ROC20"] = ta.roc(df[target], length=20)

    # 回帰
    print("REG")
    df["REG5"] = ta.linreg(df[target], length=5)
    df["REG10"] = ta.linreg(df[target], length=10)

    if target == "close":
        # ATR
        print("ATR")
        df["ATR14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["ATR28"] = ta.atr(df["high"], df["low"], df["close"], length=28)

    # RSI
    print("RSI")
    df["RSI5"] = ta.rsi(df[target], length=5)
    df["RSI10"] = ta.rsi(df[target], length=10)
    df["RSI20"] = ta.rsi(df[target], length=20)

    # RCI
    print("RCI")
    linspace9 = np.linspace(1, 9, 9)
    df["RCI9"] = df[target].rolling(9).apply(spearman, args=(linspace9,))
    linspace28 = np.linspace(1, 28, 28)
    df["RCI28"] = df[target].rolling(28).apply(spearman, args=(linspace28,))
    linspace56 = np.linspace(1, 56, 56)
    df["RCI56"] = df[target].rolling(56).apply(spearman, args=(linspace56,))

    del linspace9
    del linspace28
    del linspace56

    # BB
    print("BB")
    bbands1 = ta.bbands(df[target], length=20, std=1)
    bbands2 = ta.bbands(df[target], length=20, std=2)
    bbands3 = ta.bbands(df[target], length=20, std=3)

    df["BBL_20_1.0"] = bbands1["BBL_20_1.0"]
    df["BBM_20_1.0"] = bbands1["BBM_20_1.0"]
    df["BBU_20_1.0"] = bbands1["BBU_20_1.0"]
    df["BB_20_1.0"] = bbands1["BBU_20_1.0"] - bbands1["BBL_20_1.0"]

    df["BBL_20_2.0"] = bbands2["BBL_20_2.0"]
    df["BBM_20_2.0"] = bbands2["BBM_20_2.0"]
    df["BBU_20_2.0"] = bbands2["BBU_20_2.0"]
    df["BB_20_2.0"] = bbands2["BBU_20_2.0"] - bbands2["BBL_20_2.0"]

    df["BBL_20_3.0"] = bbands3["BBL_20_3.0"]
    df["BBM_20_3.0"] = bbands3["BBM_20_3.0"]
    df["BBU_20_3.0"] = bbands3["BBU_20_3.0"]
    df["BB_20_3.0"] = bbands3["BBU_20_3.0"] - bbands3["BBL_20_3.0"]

    del bbands1
    del bbands2
    del bbands3

    # MACD
    print("MACD")
    macd = ta.macd(df[target], fast=12, slow=26, signal=9)
    df["MACD_12_26_9"] = macd["MACD_12_26_9"]
    df["MACD_h12_26_9"] = macd["MACDh_12_26_9"]
    df["MACDs_12_26_9"] = macd["MACDs_12_26_9"]

    del macd

    if target == "close":
        # CCI
        print("CCI")
        df["CCI14"] = ta.cci(df["high"], df["low"], df["close"], length=14)
        df["CCI28"] = ta.cci(df["high"], df["low"], df["close"], length=28)

    # 歪度
    print("SKEW")
    df["SKEW10"] = df[target].rolling(10).skew()
    df["SKEW20"] = df[target].rolling(20).skew()
    df["SKEW50"] = df[target].rolling(50).skew()
    df["SKEW100"] = df[target].rolling(100).skew()
    df["SKEW200"] = df[target].rolling(200).skew()

    # 尖度
    print("KURT")
    df["KURT10"] = df[target].rolling(10).kurt()
    df["KURT20"] = df[target].rolling(20).kurt()
    df["KURT50"] = df[target].rolling(50).kurt()
    df["KURT100"] = df[target].rolling(100).kurt()
    df["KURT200"] = df[target].rolling(200).kurt()

    if target != "close":
        df.columns = [f"{target}_{i}" for i in df.columns]  # type: ignore

    # 対数差分
    df["open_log_diff"] = np.log(df["open"]).diff(1)
    df["high_log_diff"] = np.log(df["high"]).diff(1)
    df["low_log_diff"] = np.log(df["low"]).diff(1)
    df["close_log_diff"] = np.log(df["close"]).diff(1)
    df["volume_log_diff"] = np.log(df["volume"]).diff(1)
    df["SMA10"] = np.log(df["SMA10"]).diff(1)
    df["SMA20"] = np.log(df["SMA20"]).diff(1)
    df["SMA50"] = np.log(df["SMA50"]).diff(1)
    df["SMA100"] = np.log(df["SMA100"]).diff(1)
    df["SMA200"] = np.log(df["SMA200"]).diff(1)
    df["EMA10"] = np.log(df["EMA10"]).diff(1)
    df["EMA20"] = np.log(df["EMA20"]).diff(1)
    df["EMA50"] = np.log(df["EMA50"]).diff(1)
    df["EMA100"] = np.log(df["EMA100"]).diff(1)
    df["EMA200"] = np.log(df["EMA200"]).diff(1)
    df["REG5"] = np.log(df["REG5"]).diff(1)
    df["REG10"] = np.log(df["REG10"]).diff(1)
    df["BBL_20_1.0"] = np.log(df["BBL_20_1.0"]).diff(1)
    df["BBM_20_1.0"] = np.log(df["BBM_20_1.0"]).diff(1)
    df["BBU_20_1.0"] = np.log(df["BBU_20_1.0"]).diff(1)
    df["BBL_20_2.0"] = np.log(df["BBL_20_2.0"]).diff(1)
    df["BBM_20_2.0"] = np.log(df["BBM_20_2.0"]).diff(1)
    df["BBU_20_2.0"] = np.log(df["BBU_20_2.0"]).diff(1)
    df["BBL_20_3.0"] = np.log(df["BBL_20_3.0"]).diff(1)
    df["BBM_20_3.0"] = np.log(df["BBM_20_3.0"]).diff(1)
    df["BBU_20_3.0"] = np.log(df["BBU_20_3.0"]).diff(1)

    end = time.time()

    print(f"処理時間 : {np.round(end-start)} [s]")

    return df.dropna()
