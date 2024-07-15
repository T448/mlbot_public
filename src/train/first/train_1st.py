import argparse
import datetime
from typing import Dict, Final
import numpy as np
import warnings
from entity.s3 import S3Client
import mlflow
import pandas as pd
import optuna
from optuna.trial import BaseTrial

from backtest.vector.vector_backtester import VectorBacktester
from entity.backtest_params import BackTestParams
from entity.optimize_params import OptimizeParams
from common.optimize_utils import get_tscv_index_dict
from joblib import Parallel, delayed
from config import logic

# pandasのwarning非表示用
warnings.simplefilter("ignore")

BUCKET_NAME = "mlflow"

EXCLUDED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

PROJECT_SOURCE_PATH: Final[str] = "~/project/src"

s3 = S3Client()


def objective_1st(
    df: pd.DataFrame,
    backtest_params: BackTestParams,
    optimize_params: OptimizeParams,
    tscv_index_dict: Dict[int, Dict[str, datetime.datetime]],
):
    """
    1次モデルの最適化

    Args:
        df (pd.DataFrame): ohlcを持つDataFrame
        backtest_params (BackTestParams): バックテスト関連のパラメータ
        optimize_params (OptimizeParams): 最適化関連のパラメータ
        tscv_index_dict (Dict[int, Dict[str, datetime.datetime]]): 時系列cvのindex
    """

    def objective(trial: BaseTrial):

        # シグナル生成
        df_signal = logic(df=df, trial=trial)

        def run_backtest(tscv_index: int):
            start = tscv_index_dict[tscv_index]["train_start_index"]
            end = tscv_index_dict[tscv_index]["train_end_index"]
            df_backtest = df_signal.loc[start:end]  # type: ignore
            backtester = VectorBacktester(df_backtest, backtest_params, optimize_params)
            backtester.run()
            return backtester.get_metrics()

        # バックテスト
        metrics_list = Parallel(n_jobs=-1, verbose=1)(
            delayed(run_backtest)(tscv_index=tscv_index) for tscv_index in tscv_index_dict
        )

        return np.mean(metrics_list)

    return objective


def mlflow_callback_1st(metrics_name: str, run_id: str):
    """
    1次モデル最適化用コールバック関数

    Args:
        metrics_name (str): 評価指標名
    """

    def mlflow_callback(study, trial):
        trial_value = trial.value if trial.value is not None else float("nan")
        with mlflow.start_run(run_name=study.study_name, run_id=run_id):
            params = trial.params
            params[metrics_name] = trial_value
            mlflow.log_metrics(params, run_id=run_id, step=trial.number)

    return mlflow_callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_run_id")
    parser.add_argument("--preprocess_run_id")
    parser.add_argument("--train_1st_run_id")
    args = parser.parse_args()

    if args.mlflow_run_id and args.preprocess_run_id and args.train_1st_run_id:
        params = mlflow.get_run(args.mlflow_run_id).data.params
        preprocess_params = mlflow.get_run(args.preprocess_run_id).data.params
        print("preprocess_params:", preprocess_params)
        # preprocess で加工したデータを取得
        df = s3.get_dataframe_with_datetime_index(preprocess_params["preprocessed_data_name"], BUCKET_NAME)

        # 実験条件の設定
        backtest_params = BackTestParams(params)

        optimize_params = OptimizeParams(params)

        # 時系列CVのIndexを取得
        tscv_idx_dict = get_tscv_index_dict(args.preprocess_run_id)

        # optimize 1st
        study = optuna.create_study(direction=optimize_params.optimize_direction)
        study.optimize(
            objective_1st(df, backtest_params, optimize_params, tscv_idx_dict),
            n_trials=optimize_params.n_trials,
            callbacks=[mlflow_callback_1st(optimize_params.optimize_target, args.train_1st_run_id)],
        )
        best_params = {}
        for key in study.best_params.keys():
            best_params["best_" + key] = study.best_params[key]
        mlflow.log_metrics(best_params, run_id=args.mlflow_run_id)


if __name__ == "__main__":
    main()
