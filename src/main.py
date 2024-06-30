import argparse

import mlflow
from logging import getLogger, StreamHandler, INFO

from common.postgres_utils import get_last_run_commit_hash

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


def main():
    logger.info("start pipeline")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_run_id")
    args = parser.parse_args()

    if args.mlflow_run_id:
        with mlflow.start_run(run_id=args.mlflow_run_id):

            # 実験条件の設定
            backtest_params = {
                "backtest_params_jpy": 5000,
                "backtest_params_usdjpy": 145,
                "backtest_params_lot": 0.003,
                "backtest_params_lot_min": 0.001,
                "backtest_params_commision": 0.01,
                "backtest_params_use_ml": True,
                "backtest_params_use_binary_label": True,
                "backtest_params_seed": 42,
            }
            optimize_params = {
                "optimize_params_pyramiding": 3,
                "optimize_params_optimize_target": "pnl",
                "optimize_params_optimize_target_clf": "sr",
                "optimize_params_n_trials": 200,
                "optimize_params_n_trials_clf": 100,
                "optimize_params_class_weight": "balanced",  # "balanced" or "None"
                # ↓↓↓↓↓ TimeSeriesSplit用 ↓↓↓↓↓
                "optimize_params_n_splits": 5,
                "optimize_params_max_train_size": None,
                "optimize_params_test_size": None,
                "optimize_params_gap": 5,
                # ↑↑↑↑↑ TimeSeriesSplit用 ↑↑↑↑↑
                "optimize_params_evaluate_ratio": 0.25,
            }
            re_calc_features = False  # influxdbからohlcvを取得し、特徴量を再計算するフラグ。再計算には時間がかかる、使用するデータセットを揃えるため、基本的にはFalseにする。

            mlflow.log_params(backtest_params)
            mlflow.log_params(optimize_params)
            mlflow.log_param(key="re_calc_features", value=re_calc_features)

            last_run_commit_hash = get_last_run_commit_hash()
            mlflow.log_param("last_run_commit_hash", last_run_commit_hash)

            logger.info("start preprocess")
            preprocess_run = mlflow.run(
                uri="./preprocess",
                run_name="preprocess",
                entry_point="preprocess",
                backend="local",
                parameters={
                    "mlflow_run_id": args.mlflow_run_id,
                },
            )
            preprocess_run_id = preprocess_run.run_id
            preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run_id)
            logger.info("complete preprocess")

            logger.info("start train_1st")
            train_1st_run = mlflow.run(
                uri="./train",
                run_name="train_1st",
                entry_point="train_1st",
                backend="local",
                parameters={"mlflow_run_id": args.mlflow_run_id, "preprocess_run_id": preprocess_run_id},
            )
            train_1st_run_id = train_1st_run.run_id
            train_1st_run = mlflow.tracking.MlflowClient().get_run(train_1st_run_id)
            logger.info("complete train_1st")

            logger.info("start train_2nd")
            train_2nd_run = mlflow.run(
                uri="./train",
                run_name="train_2nd",
                entry_point="train_2nd",
                backend="local",
                parameters={
                    "mlflow_run_id": args.mlflow_run_id,
                    "preprocess_run_id": preprocess_run_id,
                    "train_1st_run_id": train_1st_run_id,
                },
            )
            train_2nd_run_id = train_2nd_run.run_id
            train_2nd_run = mlflow.tracking.MlflowClient().get_run(train_2nd_run_id)
            logger.info("complete train_2nd")

            logger.info("start build")
            build_run = mlflow.run(
                uri="./build_model",
                run_name="build",
                entry_point="build",
                backend="local",
                parameters={
                    "mlflow_run_id": args.mlflow_run_id,
                    "preprocess_run_id": preprocess_run_id,
                    "train_1st_run_id": train_1st_run_id,
                    "train_2nd_run_id": train_2nd_run_id,
                },
            )
            build_run_id = build_run.run_id
            build_run = mlflow.tracking.MlflowClient().get_run(build_run_id)
            logger.info("complete build")

            logger.info("start evaluate")
            evaluate_run = mlflow.run(
                uri="./evaluate",
                run_name="evaluate",
                entry_point="evaluate",
                backend="local",
                parameters={
                    "mlflow_run_id": args.mlflow_run_id,
                    "preprocess_run_id": preprocess_run_id,
                    "train_1st_run_id": train_1st_run_id,
                    "train_2nd_run_id": train_2nd_run_id,
                    "build_run_id": build_run_id,
                },
            )
            evaluate_run_id = evaluate_run.run_id
            evaluate_run = mlflow.tracking.MlflowClient().get_run(evaluate_run_id)
            logger.info("complete evaluate")

    logger.info("complete pipeline")


if __name__ == "__main__":
    main()
