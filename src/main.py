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
                "backtest_params_jpy": 5000,  # 口座残高初期値
                "backtest_params_usdjpy": 160,  # ドル円
                "backtest_params_lot": 0.003,  # ポジションの最大サイズ
                "backtest_params_lot_min": 0.001,  # 最小取引単位
                "backtest_params_commision": 0.01,  # 手数料
                "backtest_params_use_ml": True,  # 2次モデルを使う/使わない
                "backtest_params_use_binary_label": True,  # 2次モデルを2クラス問題とする/多クラス問題とする
                "backtest_params_seed": 42,  # 乱数シード
            }
            optimize_params = {
                "optimize_params_pyramiding": 3,  # 最大ピラミッディング数
                "optimize_params_optimize_target": "pnl",  # 1次モデル評価指標 pnl:最終的な損益、sr:シャープレシオ、max_dd:最大ドローダウン
                "optimize_params_optimize_target_clf": "sr",  # 2次モデル評価指標
                "optimize_params_n_trials": 200,  # 1次モデル optunaでの試行回数
                "optimize_params_n_trials_clf": 100,  # 2次モデル optunaでの試行回数
                "optimize_params_class_weight": "balanced",  # "balanced" or "None"
                # ↓↓↓↓↓ TimeSeriesSplit用 ↓↓↓↓↓
                "optimize_params_n_splits": 5,  # 分割数
                "optimize_params_max_train_size": None,
                "optimize_params_test_size": None,
                "optimize_params_gap": 5,  # 学習データとテストデータの間何点あけるか
                # ↑↑↑↑↑ TimeSeriesSplit用 ↑↑↑↑↑
                "optimize_params_evaluate_ratio": 0.25,  # evaluate step で使用するデータの割合
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
