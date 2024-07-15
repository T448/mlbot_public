from typing import Any, Dict, Literal
import mlflow
import pandas as pd
from optuna.trial import BaseTrial
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from entity.backtest_params import BackTestParams
from entity.optimize_params import OptimizeParams
from logic.sample_logic_limit import sample_logic_limit

ARGUMENT_ERROR_MESSAGE = "trial もしくは mlflow_run_id を指定してください。"


def logic(
    df: pd.DataFrame,
    trial: BaseTrial | None = None,
    mlflow_run_id: str | None = None,
) -> pd.DataFrame:
    """
    pipeline,botで使用するロジックを定義する

    Args:
        df (pd.DataFrame): ohlcv,DatetimeIndexをもつDataFrame
        trial (BaseTrial | None, optional): パラメータチューニング時はこちらを使用
        mlflow_run_id (str | None, optional): パラメータチューニング時以外はこちらを使用

    Raises:
        TypeError: trial もしくは mlflow_run_id を指定する必要がある。それ以外の場合はTypeError。
    Returns:
        pd.DataFrame: ラベルのカラムが追加されたDataFrame
    """
    if trial is not None and mlflow_run_id is None:
        # logic変更時の変更箇所1
        entry_length = trial.suggest_int("entry_length", 5, 25)
        entry_point = trial.suggest_float("entry_point", 0.1, 15, log=True)
    elif trial is None and mlflow_run_id is not None:
        # logic変更時の変更箇所2
        client = mlflow.tracking.MlflowClient()
        entry_length = int(client.get_metric_history(mlflow_run_id, "best_entry_length")[0].value)
        entry_point = float(client.get_metric_history(mlflow_run_id, "best_entry_point")[0].value)
    else:
        raise TypeError(ARGUMENT_ERROR_MESSAGE)

    # ここでlogicの変更をする
    df_signal = sample_logic_limit(_df=df, entry_length=entry_length, entry_point=entry_point)

    return df_signal


def classifier(
    optimize_params: OptimizeParams,
    backtest_params: BackTestParams,
    trial: BaseTrial | None = None,
    mlflow_run_id: str | None = None,
    buy_sell_flag: Literal["buy", "sell", None] = None,
) -> BaseEstimator:
    """
    pipeline,botで使用する分類器を定義する

    Args:
        optimize_params (OptimizeParams): _description_
        backtest_params (BackTestParams): _description_
        trial (BaseTrial | None, optional): パラメータチューニング時はこちらを使用
        mlflow_run_id (str | None, optional): パラメータチューニング時以外はこちらを使用
        buy_sell_flag (Literal[&quot;buy&quot;, &quot;sell&quot;, None], optional): buy専用、sell専用モデルを作成する場合用
    Raises:
        TypeError: trial もしくは mlflow_run_id を指定する必要がある。それ以外の場合はTypeError。

    Returns:
        BaseEstimator: Classifier
    """
    num_leaves_name = "num_leaves" if buy_sell_flag is None else f"num_leaves_{buy_sell_flag}"
    learning_rate_name = "learning_rate" if buy_sell_flag is None else f"learning_rate_{buy_sell_flag}"
    colsample_bytree_name = "colsample_bytree" if buy_sell_flag is None else f"colsample_bytree_{buy_sell_flag}"
    min_child_samples_name = "min_child_samples" if buy_sell_flag is None else f"min_child_samples_{buy_sell_flag}"
    reg_alpha_name = "reg_alpha" if buy_sell_flag is None else f"reg_alpha_{buy_sell_flag}"
    reg_lambda_name = "reg_lambda" if buy_sell_flag is None else f"reg_lambda_{buy_sell_flag}"
    max_depth_name = "max_depth" if buy_sell_flag is None else f"max_depth_{buy_sell_flag}"

    if trial is not None and mlflow_run_id is None:
        # classifier 変更時の変更箇所1
        num_leaves = trial.suggest_int(num_leaves_name, 4, 128)
        learning_rate = trial.suggest_float(learning_rate_name, 0.5, 0.9)
        colsample_bytree = trial.suggest_float(colsample_bytree_name, 0.5, 0.9)
        min_child_samples = trial.suggest_int(min_child_samples_name, 4, 128)
        reg_alpha = trial.suggest_float(reg_alpha_name, 1e-1, 1e4, log=True)
        reg_lambda = trial.suggest_float(reg_lambda_name, 1e-1, 1e4, log=True)
        max_depth = trial.suggest_int(max_depth_name, 4, 64)
    elif trial is None and mlflow_run_id is not None:
        client = mlflow.tracking.MlflowClient()
        num_leaves = int(client.get_metric_history(mlflow_run_id, "best_" + num_leaves_name)[0].value)
        learning_rate = float(client.get_metric_history(mlflow_run_id, "best_" + learning_rate_name)[0].value)
        colsample_bytree = float(client.get_metric_history(mlflow_run_id, "best_" + colsample_bytree_name)[0].value)
        min_child_samples = int(client.get_metric_history(mlflow_run_id, "best_" + min_child_samples_name)[0].value)
        reg_alpha = float(client.get_metric_history(mlflow_run_id, "best_" + reg_alpha_name)[0].value)
        reg_lambda = float(client.get_metric_history(mlflow_run_id, "best_" + reg_lambda_name)[0].value)
        max_depth = int(client.get_metric_history(mlflow_run_id, "best_" + max_depth_name)[0].value)
    else:
        raise TypeError(ARGUMENT_ERROR_MESSAGE)

    # 分類器のパラメータ
    clf_params: Dict[str, Any] = {
        "num_leaves": num_leaves,
        "objective": "binary",
        "learning_rate": learning_rate,
        "metrics": "binary_logloss",
        "colsample_bytree": colsample_bytree,
        "min_child_samples": min_child_samples,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "max_depth": max_depth,
        "class_weight": optimize_params.class_weight,
        "seed": backtest_params.seed,
        "verbose": -1,
    }

    return LGBMClassifier(**clf_params)
