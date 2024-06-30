import argparse
import datetime
from typing import Dict
import warnings
import mlflow
import numpy as np
import pandas as pd
import optuna
from optuna.trial import BaseTrial


from backtest.vector.vector_backtester import VectorBacktester
from config import classifier, logic
from entity.backtest_params import BackTestParams
from entity.optimize_params import OptimizeParams
from entity.s3 import S3Client
from common.preprocess_utils import get_label
from common.optimize_utils import get_tscv_index_dict

# pandasのwarning非表示用
warnings.simplefilter("ignore")

BUCKET_NAME = "mlflow"

EXCLUDED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "limit",
    "buy",
    "sell",
    "long",
    "short",
    "exec_price_buy",
    "exec_price_sell",
    "y_buy",
    "y_sell",
]

OFFSET: int = 10000

s3 = S3Client()


def objective_2nd(
    df: pd.DataFrame,
    backtest_params: BackTestParams,
    optimize_params: OptimizeParams,
    tscv_index_dict: Dict[int, Dict[str, datetime.datetime]],
):
    """
    2次モデルの最適化

    Args:
        df (pd.DataFrame): ohlcを持つDataFrame
        backtest_params (BackTestParams): バックテスト関連のパラメータ
        optimize_params (OptimizeParams): 最適化関連のパラメータ
        tscv_index_dict (Dict[int, Dict[str, datetime.datetime]]): 時系列cvのindex
    """

    def objective(trial: BaseTrial):
        df_features = df[[i for i in df.columns if i not in EXCLUDED_COLUMNS]]
        df_buy_label = df["y_buy"]
        df_sell_label = df["y_sell"]

        print("df_features (objective)", df_features)

        def fit_and_run(tscv_index: int):
            clf_buy = classifier(
                optimize_params=optimize_params, backtest_params=backtest_params, trial=trial, buy_sell_flag="buy"
            )
            clf_sell = classifier(
                optimize_params=optimize_params, backtest_params=backtest_params, trial=trial, buy_sell_flag="sell"
            )

            train_start = tscv_index_dict[tscv_index]["train_start_index"]
            train_end = tscv_index_dict[tscv_index]["train_end_index"]
            test_start = tscv_index_dict[tscv_index]["test_start_index"]
            test_end = tscv_index_dict[tscv_index]["test_end_index"]

            print("tscv_index_dict[tscv_index]", tscv_index_dict[tscv_index])

            df_train_features = df_features.loc[train_start:train_end]  # type: ignore
            df_train_buy_label = df_buy_label.loc[train_start:train_end]  # type: ignore
            df_train_sell_label = df_sell_label.loc[train_start:train_end]  # type: ignore

            print("df_train_features (objective)", df_train_features)

            df_test = df.loc[test_start:test_end]  # type: ignore
            df_test_features = df_features.loc[test_start:test_end]  # type: ignore

            print("df_test_features (objective)", df_test_features)

            clf_buy.fit(df_train_features, df_train_buy_label)
            clf_sell.fit(df_train_features, df_train_sell_label)

            buy_pred = clf_buy.predict(df_test_features)
            sell_pred = clf_sell.predict(df_test_features)

            df_test["buy"] = df_test["buy"].where(buy_pred != 0, df_test["buy"] - OFFSET)
            df_test["sell"] = df_test["sell"].where(sell_pred != 0, df_test["sell"] + OFFSET)

            # バックテスト
            backtester = VectorBacktester(df_test, backtest_params, optimize_params)
            backtester.run()

            return backtester.get_metrics()

        # NOTE 並列にすると遅くなる
        metrics_list = [fit_and_run(tscv_index) for tscv_index in tscv_index_dict]

        return np.mean(metrics_list)

    return objective


def mlflow_callback_2nd(metrics_name: str, run_id: str):
    """
    2次モデル最適化用コールバック関数

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
    parser.add_argument("--train_2nd_run_id")
    args = parser.parse_args()

    if args.mlflow_run_id and args.preprocess_run_id and args.train_1st_run_id and args.train_2nd_run_id:
        params = mlflow.get_run(args.mlflow_run_id).data.params
        preprocess_params = mlflow.get_run(args.preprocess_run_id).data.params

        # preprocess で加工したデータを取得
        df = s3.get_dataframe_with_datetime_index(preprocess_params["preprocessed_data_name"], BUCKET_NAME)

        # 1次モデルのbest_paramsを使用し、シグナル生成
        df_signal = logic(df=df, mlflow_run_id=args.mlflow_run_id)
        print("df_signal", df_signal)

        # 実験条件の設定
        backtest_params = BackTestParams(params)
        np.random.seed(backtest_params.seed)

        optimize_params = OptimizeParams(params)
        # ラベリング
        df_features_and_labels = get_label(df_signal, backtest_params.use_binary_label)
        print("df_features_and_labels", df_features_and_labels)

        # 時系列CVのIndexを取得
        tscv_idx_dict = get_tscv_index_dict(args.preprocess_run_id)

        # optimize 2nd
        study = optuna.create_study(direction=optimize_params.optimize_direction)
        study.optimize(
            objective_2nd(df_features_and_labels, backtest_params, optimize_params, tscv_idx_dict),
            n_trials=optimize_params.n_trials_clf,
            callbacks=[mlflow_callback_2nd(optimize_params.optimize_target, args.train_2nd_run_id)],
        )
        best_params = {}
        for key in study.best_params.keys():
            best_params["best_" + key] = study.best_params[key]
        mlflow.log_metrics(best_params, run_id=args.mlflow_run_id)


if __name__ == "__main__":
    main()
