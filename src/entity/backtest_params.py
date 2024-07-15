import numpy as np
from typing import Any, Dict


class BackTestParams:
    def __init__(
        self,
        params: Dict[str, Any]  # 親runで保存する
    ) -> None:
        self.JPY = float(params["backtest_params_jpy"])
        self.USDJPY = float(params["backtest_params_usdjpy"])
        self.lot = float(params["backtest_params_lot"])
        self.lot_min = float(params["backtest_params_lot_min"])
        self.commision = float(params["backtest_params_commision"])
        self.use_ML = bool(params["backtest_params_use_ml"])
        self.use_binary_label = bool(params["backtest_params_use_binary_label"])
        self.seed = int(params["backtest_params_seed"])

        self.USD = self.JPY / self.USDJPY

        # 最低取引数と最大取引数の関係から、どこまでピラミッディングできるかが決まる。
        self.pyramiding_max = int(np.floor(self.lot / self.lot_min))

        self.validation(self.JPY, self.USDJPY, self.lot, self.lot_min)

    def validation(self, JPY: float, USDJPY: float, lot: float, lot_min: float) -> None:
        if JPY < 0:
            raise ValueError("JPY > 0")
        if USDJPY < 0:
            raise ValueError("USDJPY > 0")
        if lot < 0.001:
            raise ValueError("lot >= 0.001")
        if lot_min < 0.001:
            raise ValueError("lot_min >= 0.001")
        if lot < lot_min:
            raise ValueError("lot >= lot_min")
