from typing import Literal
from typing import Any, Dict


class OptimizeParams:
    def __init__(self, params: Dict[str, Any]) -> None:  # 親runで保存する
        self.pyramiding = int(params["optimize_params_pyramiding"])
        self.optimize_target = params["optimize_params_optimize_target"]
        self.optimize_target_clf = params["optimize_params_optimize_target_clf"]
        self.n_trials = int(params["optimize_params_n_trials"])
        self.n_trials_clf = int(params["optimize_params_n_trials_clf"])
        self.class_weight: Literal["balanced", "None"] = params["optimize_params_class_weight"]

        self.n_splits = int(params["optimize_params_n_splits"])
        self.max_train_size: int | None = None
        try:
            self.max_train_size = int(params["optimize_params_max_train_size"])
        except Exception:
            pass

        self.test_size: int | None = None
        try:
            self.test_size = int(params["optimize_params_test_size"])
        except Exception:
            pass

        self.gap = int(params["optimize_params_gap"])
        self.evaluate_ratio = float(params["optimize_params_evaluate_ratio"])

        self.optimize_direction = self.get_direction(self.optimize_target)
        self.optimize_direction_clf = self.get_direction(self.optimize_target_clf)

        self.validation(self.pyramiding, self.n_trials, self.n_trials_clf)

    def validation(self, pyramiding: int, n_trials: int, n_trials_clf: int) -> None:
        if pyramiding < 1:
            raise ValueError("pyramiding > 0")
        if n_trials < 1:
            raise ValueError("n_trials >= 1")
        if n_trials_clf < 1:
            raise ValueError("n_trials_clf >= 1")

    def get_direction(self, target: Literal["sr", "pnl", "max_dd"]):
        if target == "sr" or target == "pnl":
            return "maximize"
        elif target == "max_dd":
            return "minimize"
        else:
            raise ValueError(f"{target} is not optimize-target")
