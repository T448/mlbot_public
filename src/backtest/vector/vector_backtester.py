import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from entity.backtest_params import BackTestParams
from entity.optimize_params import OptimizeParams

# pandasのwarning非表示用
warnings.simplefilter("ignore")


class VectorBacktester:
    def __init__(self, df: pd.DataFrame, backtest_params: BackTestParams, optimize_params: OptimizeParams):
        self.df = df
        self.backtest_params = backtest_params
        self.optimize_params = optimize_params

    def run(self):
        # 指値位置補正
        self.df["limit"] = self.df[["open"]].shift(-1)

        self.df["buy"] = self.df[["buy", "limit"]].min(axis="columns")
        self.df["sell"] = self.df[["sell", "limit"]].max(axis="columns")

        # 約定判断
        # ヒットしなければすぐ削除
        self.df["long"] = self.df["low"] < self.df["buy"].shift(1)
        self.df["short"] = self.df["high"] > self.df["sell"].shift(1)

        # ポジション計算部分(ドテン&ピラミッディング)
        self.df["order"] = 0
        self.df["order"] = self.df["order"].where(~self.df["long"], 1)
        self.df["order"] = self.df["order"].where(~self.df["short"], -1)
        self.df["order"] = self.df["order"].where(~(self.df["long"] == self.df["short"]), 0)

        self.df["pos"] = (
            self.df["order"]
            .where(
                self.df["order"] != 0,
            )
            .ffill()
            .fillna(0)
        )

        self.df["pos"] = self.df.groupby((self.df["pos"] * self.df["pos"].shift(1) < 0).cumsum().fillna(0))[
            "order"
        ].cumsum()
        self.df["pos"] = self.df["pos"].where(
            self.df["pos"] <= self.optimize_params.pyramiding, self.optimize_params.pyramiding
        )
        self.df["pos"] = self.df["pos"].where(
            self.df["pos"] >= -self.optimize_params.pyramiding, -self.optimize_params.pyramiding
        )

        # 約定価格
        self.df["exec_price"] = self.df["close"]
        self.df["exec_price"] = self.df["buy"].where(self.df["long"].shift(-1), self.df["exec_price"])
        self.df["exec_price"] = self.df["sell"].where(self.df["short"].shift(-1), self.df["exec_price"])

        # 損益計算
        self.df["pos"] /= self.optimize_params.pyramiding

        # lot調整
        self.df["pos"] *= self.backtest_params.lot

        # 不要かもしれない
        # # 最低取引枚数以上でないといけないので置換する
        # self.df.loc[(0 < self.df["pos"]) & (self.df["pos"] < backtest_params.lot_min), "pos"]
        # = backtest_params.lot_min
        # self.df.loc[(-backtest_params.lot_min < self.df["pos"]) & (self.df["pos"] < 0), "pos"]
        # = -backtest_params.lot_min
        self.df["return"] = self.df["exec_price"].diff(1).fillna(0)
        self.df["commision"] = (
            self.df["exec_price"] * self.backtest_params.commision / 100 * abs(self.df["pos"] - self.df["pos"].shift(1))
        )
        self.df["profit"] = self.df["return"] * self.df["pos"] - self.df["commision"]
        self.df["pnl"] = self.df["profit"].cumsum()

    def make_graph(self):
        plt.clf()
        plt.figure()

        plt.plot(self.df.index, self.df["pnl"].values)

        plt.xlabel("datetime")
        plt.ylabel("pnl")

        img = BytesIO()
        plt.savefig(img)

        return Image.open(img)

    def get_metrics(self):
        if self.optimize_params.optimize_target == "pnl":
            return self.df["pnl"].values[-1]
        elif self.optimize_params.optimize_target == "sr":
            return np.mean(self.df["profit"].values) / np.std(self.df["profit"].values)
        elif self.optimize_params.optimize_target == "max_dd":
            # TODO max_dd
            return 1
        else:
            raise ValueError("optimize_target は pnl, sr, max_dd から選択してください")
