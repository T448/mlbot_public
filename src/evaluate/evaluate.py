import argparse
import datetime
from io import BytesIO
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from lightgbm import LGBMClassifier

from backtest.vector.vector_backtester import VectorBacktester
from config import logic
from entity.backtest_params import BackTestParams
from entity.optimize_params import OptimizeParams
from entity.s3 import S3Client
from common.optimize_utils import get_eval_index

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


SEED = 42
np.random.seed(SEED)

OFFSET = 10000

s3 = S3Client()


def get_pnl_img(df: pd.DataFrame):
    plt.clf()
    plt.figure()
    plt.plot(df.index, df["pnl"])
    plt.xlabel("date")
    plt.ylabel("pnl")

    img = BytesIO()
    plt.savefig(img)

    return Image.open(img)


def get_feature_importance_img(clf: LGBMClassifier) -> Image:
    plt.clf()
    plt.figure(figsize=(10, len(clf.feature_name_) / 5))
    plt.barh(y=clf.feature_name_, width=clf.feature_importances_)
    plt.axvline(0, color="black")
    plt.xlabel("feature_importance")
    plt.ylabel("feature_name")

    img = BytesIO()
    plt.savefig(img, bbox_inches="tight")

    return Image.open(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_run_id")
    parser.add_argument("--preprocess_run_id")
    parser.add_argument("--train_1st_run_id")
    parser.add_argument("--train_2nd_run_id")
    parser.add_argument("--build_run_id")
    args = parser.parse_args()

    if (
        args.mlflow_run_id
        and args.preprocess_run_id
        and args.train_1st_run_id
        and args.train_2nd_run_id
        and args.build_run_id
    ):
        # TODO python_model.pkl では前処理から予測まで通して実行できるため、活用する (本番に近づけるためにもそのように実装するのがよい)
        params = mlflow.get_run(args.mlflow_run_id).data.params
        preprocess_params = mlflow.get_run(args.preprocess_run_id).data.params

        # preprocess で加工したデータを取得
        df = s3.get_dataframe_with_datetime_index(preprocess_params["preprocessed_data_name"], BUCKET_NAME)

        # 2次モデルを取得
        # clf_buy = s3.get_pkl(args.mlflow_run_id, "artifacts/clf_buy/model.pkl")
        # if type(clf_buy) is not BaseEstimator:
        #     raise TypeError("clf is not BaseEstimator")
        # clf_sell = s3.get_pkl(args.mlflow_run_id, "artifacts/clf_sell/model.pkl")
        # if type(clf_sell) is not BaseEstimator:
        #     raise TypeError("clf is not BaseEstimator")
        clf = s3.get_pkl(args.mlflow_run_id, "artifacts/clf/python_model.pkl")
        img = get_feature_importance_img(clf.clf_buy)
        mlflow.log_image(img, "feature_importance_buy.png")
        img = get_feature_importance_img(clf.clf_sell)
        mlflow.log_image(img, "feature_importance_sell.png")

        # シグナル生成
        # 1次モデルのbest_paramsを使用し、シグナル生成
        df_signal = logic(df=df, mlflow_run_id=args.mlflow_run_id)

        # ラベリング
        # df_features_and_labels = get_label(df_signal)

        # テストデータの取得
        eval_idx = get_eval_index(args.preprocess_run_id)
        df_test = df_signal.loc[eval_idx["eval_start_index"] : eval_idx["eval_end_index"]]

        df_test_features = df_test[[i for i in df_test.columns if i not in EXCLUDED_COLUMNS]]

        # 実験条件の設定
        backtest_params = BackTestParams(params)
        np.random.seed(backtest_params.seed)

        optimize_params = OptimizeParams(params)

        # 予測
        buy_pred = clf.clf_buy.predict(df_test_features)
        sell_pred = clf.clf_sell.predict(df_test_features)

        df_test_with_2nd = df_test.copy()

        # 予測結果がFalseの場合、指値位置を極端にずらすことで取引しない扱いにする
        df_test_with_2nd["buy"] = df_test_with_2nd["buy"].where(buy_pred != 0, df_test_with_2nd["buy"] - OFFSET)
        df_test_with_2nd["sell"] = df_test_with_2nd["sell"].where(sell_pred != 0, df_test_with_2nd["sell"] + OFFSET)

        # バックテスト
        # 1次モデルのみ
        backtester = VectorBacktester(df_test, backtest_params, optimize_params)
        backtester.run()
        img = get_pnl_img(backtester.df[["pnl"]])
        mlflow.log_image(img, "pnl.png")

        index_timestamp = [1000 * int(datetime.datetime.timestamp(i)) for i in backtester.df.index.to_pydatetime()]
        Parallel(n_jobs=-1, verbose=1)(
            delayed(mlflow.log_metric)(key="pnl", value=k, step=i, timestamp=j, run_id=args.mlflow_run_id)
            for i, (j, k) in enumerate(zip(index_timestamp, backtester.df["pnl"].values))
        )

        # バックテスト
        # 2次モデル
        backtester_with_2nd = VectorBacktester(df_test_with_2nd, backtest_params, optimize_params)
        backtester_with_2nd.run()

        df_pnl_with_2nd = backtester_with_2nd.df[["pnl"]].dropna()
        img = get_pnl_img(df_pnl_with_2nd)
        mlflow.log_image(img, "pnl_with_2nd.png")

        index_timestamp = [1000 * int(datetime.datetime.timestamp(i)) for i in df_pnl_with_2nd.index.to_pydatetime()]
        Parallel(n_jobs=-1, verbose=1)(
            delayed(mlflow.log_metric)(key="pnl_with_2nd", value=k, step=i, timestamp=j, run_id=args.mlflow_run_id)
            for i, (j, k) in enumerate(zip(index_timestamp, df_pnl_with_2nd["pnl"].values))
        )


if __name__ == "__main__":
    main()
