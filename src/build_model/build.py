import argparse
import pickle
from typing import Callable, Dict
import mlflow
import numpy as np
from mlflow.models import infer_signature
import pandas as pd

from config import classifier, logic
from entity.s3 import S3Client
from entity.backtest_params import BackTestParams
from entity.optimize_params import OptimizeParams
from common.preprocess_utils import get_label
from common.optimize_utils import get_tscv_index_dict
from feature.feature_calculator import calc_feature, get_feature_names
from for_test.generate_data import ohlcv_with_datetime_index

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
        df = df.astype("float")

        # 実験条件の設定
        backtest_params = BackTestParams(params)
        np.random.seed(backtest_params.seed)

        optimize_params = OptimizeParams(params)

        # 1次モデルのbest_paramsを使用し、シグナル生成
        df_signal = logic(df=df, mlflow_run_id=args.mlflow_run_id)

        # ラベリング
        df_features_and_labels = get_label(df_signal, backtest_params.use_binary_label)
        tscv_idx_dict = get_tscv_index_dict(args.preprocess_run_id)
        tscv_idx = tscv_idx_dict[max(tscv_idx_dict.keys())]
        df_train = df_features_and_labels.loc[tscv_idx["train_start_index"] : tscv_idx["train_end_index"]]
        # df_test = df_features_and_labels.loc[tscv_idx["test_start_index"] : tscv_idx["test_end_index"]]
        df_train_features = df_train[[i for i in df_train.columns if i not in EXCLUDED_COLUMNS]]
        df_train_buy_label = df_train["y_buy"]
        df_train_sell_label = df_train["y_sell"]

        # df_test_features = df_test[[i for i in df_test.columns if i not in EXCLUDED_COLUMNS]]
        # df_test_buy_label = df_test["y_buy"]
        # df_test_sell_label = df_test["y_sell"]

        clf_buy = classifier(
            optimize_params=optimize_params,
            backtest_params=backtest_params,
            mlflow_run_id=args.mlflow_run_id,
            buy_sell_flag="buy",
        )
        clf_sell = classifier(
            optimize_params=optimize_params,
            backtest_params=backtest_params,
            mlflow_run_id=args.mlflow_run_id,
            buy_sell_flag="sell",
        )

        clf_buy.fit(df_train_features, df_train_buy_label)
        clf_sell.fit(df_train_features, df_train_sell_label)

        # signature_buy = infer_signature(df_train_features, df_train_buy_label)
        # signature_sell = infer_signature(df_train_features, df_train_sell_label)
        # input_example = df_train_features.sample(n=1)

        def generate_calc_feature_wrapper() -> Callable[[pd.DataFrame], pd.DataFrame]:

            def calc_feature_wrapper(_df: pd.DataFrame) -> pd.DataFrame:
                df = _df.copy()
                return calc_feature(df)

            return calc_feature_wrapper

        def generate_preprocess_func_wrapper() -> Callable[[pd.DataFrame], pd.DataFrame]:
            # feature_columns = get_feature_names()

            obj = s3.s3.get_object(Bucket=BUCKET_NAME, Key=f"0/{args.mlflow_run_id}/artifacts/ss/ss.pkl")
            s3_data = obj["Body"].read()
            ss = pickle.loads(s3_data)
            feature_columns = ss.feature_names_in_

            def preprocess_func_wrapper(_df: pd.DataFrame) -> pd.DataFrame:
                print("ss.feature_names_in_", ss.feature_names_in_)
                df = _df.copy()
                df_features = df[feature_columns]
                df_wo_features = df[[i for i in df.columns if i not in feature_columns]]
                df_features_normal = pd.DataFrame(
                    ss.transform(df_features), columns=feature_columns, index=df_features.index
                )

                return pd.concat(
                    [df_wo_features[df_features_normal.index[0] : df_features_normal.index[-1]], df_features_normal],
                    axis=1,
                )

            return preprocess_func_wrapper

        def generate_logic_func_wrapper() -> Callable[[pd.DataFrame], pd.DataFrame]:
            mlflow_run_id = args.mlflow_run_id

            def logic_func_wrapper(_df: pd.DataFrame) -> pd.DataFrame:
                df = _df.copy()
                return logic(df, mlflow_run_id=mlflow_run_id)

            return logic_func_wrapper

        class LGBMClassifierWrapper(mlflow.pyfunc.PythonModel):
            """
            calc_feature, preprocess_func, logic_func は pd.DataFrame のみを引数に取るようにする

            Args:
                mlflow (_type_): _description_
            """

            def __init__(self, logic, calc_feature, preprocess, clf_buy, clf_sell):
                self.logic = logic
                self.calc_feature = calc_feature
                self.preprocess = preprocess
                self.clf_buy = clf_buy
                self.clf_sell = clf_sell

            def predict(
                self, df: pd.DataFrame, context=None, params=None
            ) -> Dict[str, float]:  # TODO contextなしでビルドできるか確認 or contextの型を調べる
                df_signal = self.logic(df)
                df_features = self.calc_feature(df)
                df_preprocessed = self.preprocess(df_features)

                pred_buy = self.clf_buy.predict(df_preprocessed[self.clf_buy.feature_name_][-1:])
                pred_sell = self.clf_sell.predict(df_preprocessed[self.clf_sell.feature_name_][-1:])

                buy_limit_price = df_signal["buy"].values[-1] - (1 - pred_buy[-1]) * OFFSET
                sell_limit_price = df_signal["sell"].values[-1] + (1 - pred_sell[-1]) * OFFSET

                print("buy_limit_price", buy_limit_price)
                print("sell_limit_price", sell_limit_price)

                return {
                    "buy_limit_price": buy_limit_price,
                    "sell_limit_price": sell_limit_price,
                }

        preprocess_func_wrapper = generate_preprocess_func_wrapper()
        calc_feature_wrapper = generate_calc_feature_wrapper()
        logic_func_wrapper = generate_logic_func_wrapper()

        clf_wrapper = LGBMClassifierWrapper(
            logic=logic_func_wrapper,
            calc_feature=calc_feature_wrapper,
            preprocess=preprocess_func_wrapper,
            clf_buy=clf_buy,
            clf_sell=clf_sell,
        )

        model_input = ohlcv_with_datetime_index()
        model_output = {
            "buy_limit_price": 10000,
            "sell_limit_price": 20000,
        }

        signature = infer_signature(model_input, model_output)
        input_example = model_input.sample(n=1)

        with mlflow.start_run(run_id=args.mlflow_run_id):
            # mlflow.lightgbm.log_model(
            #     clf_buy, artifact_path="clf_buy", signature=signature_buy, input_example=input_example
            # )
            # mlflow.lightgbm.log_model(
            #     clf_sell, artifact_path="clf_sell", signature=signature_sell, input_example=input_example
            # )
            mlflow.pyfunc.log_model("clf", python_model=clf_wrapper, signature=signature, input_example=input_example)


def generate_schema():
    df = pd.DataFrame()


if __name__ == "__main__":
    main()
