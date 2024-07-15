import argparse
import datetime
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import mlflow
from tempfile import TemporaryDirectory
import os
import pickle
import pandas as pd
import boto3

from entity.optimize_params import OptimizeParams
from entity.s3 import S3Client
from common.git_utils import has_diff
from common.imported_modules import get_imported_modules
from common.influxdb_utils import get_ohlcv
from feature.feature_calculator import calc_feature

BUCKET_NAME = "mlflow"

EXCLUDED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

s3 = S3Client()
s3_resource = boto3.resource(
    "s3",
    endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def calc_standard_scaler(df: pd.DataFrame, ss: StandardScaler | None = None) -> tuple[StandardScaler, pd.DataFrame]:
    """
    標準化

    Args:
        df (pd.DataFrame): 標準化する特徴量DataFrame

    Returns:
        tuple[StandardScaler, pd.DataFrame]: 標準化器、変換後DataFrame
    """
    df_standard_columns = [i for i in df.columns if i not in EXCLUDED_COLUMNS]

    if "date" not in df.columns:
        df["date"] = df.index

    if ss is None:
        ss = StandardScaler()
        ss.fit(df[df_standard_columns])

    transformed_array = ss.transform(df[df_standard_columns])
    df_standard = pd.DataFrame(transformed_array, columns=df_standard_columns, index=df.index)
    del transformed_array

    df = df[EXCLUDED_COLUMNS]
    df = pd.concat([df, df_standard], axis=1).dropna()
    del df_standard

    return ss, df


def get_split_idx(df: pd.DataFrame, optimize_params: OptimizeParams) -> Dict[int, Dict[str, Any]]:
    """
    時系列CVのindexを取得

    Args:
        df (pd.DataFrame): _description_
        optimize_params (OptimizeParams): _description_

    Returns:
        Dict[str, Any]: _description_
    """

    n_splits = optimize_params.n_splits
    max_train_size = optimize_params.max_train_size
    test_size = optimize_params.test_size
    gap = optimize_params.gap

    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap)

    tscv_idx_dict = {}

    for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
        tscv_idx_dict[i] = {
            "train_start_index": int(df.index[train_idx[0]].to_pydatetime().timestamp()),
            "train_end_index": int(df.index[train_idx[-1]].to_pydatetime().timestamp()),
            "test_start_index": int(df.index[test_idx[0]].to_pydatetime().timestamp()),
            "test_end_index": int(df.index[test_idx[-1]].to_pydatetime().timestamp()),
        }

    return tscv_idx_dict


def get_latest_file_name(prefix: str) -> str:
    """
    前回の計算結果を格納したcsvのファイル名を取得する
    """
    s3_file_list = s3.s3.list_objects(Bucket=BUCKET_NAME, Prefix="data/")
    extracted_file_list = [i["Key"] for i in s3_file_list["Contents"] if prefix in i["Key"]]
    return max(extracted_file_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_run_id")
    parser.add_argument("--preprocess_run_id")
    args = parser.parse_args()

    if args.mlflow_run_id and args.preprocess_run_id:
        params = mlflow.get_run(args.mlflow_run_id).data.params

        # preprocess 関連で1つ前のcommitと差分がない場合は前回のデータを使用する
        skip = False
        if params["last_run_commit_hash"] != "":
            print("last_run_commit_hash", params["last_run_commit_hash"])
            file_list = get_imported_modules()
            file_list.append(__file__)
            print("[preprocess] file_list", file_list)
            skip = not has_diff(params["last_run_commit_hash"], file_list)
        print("[preprocess skip]", skip)

        now = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")

        # 特徴量の計算
        if params["re_calc_features"]:
            df_ohlcv = get_ohlcv("1y")  # influxdbからohlcvデータ取得
            df_features = calc_feature(df_ohlcv, "close")
            features_file_name = f"data/features_{now}.csv"
            s3_obj = s3_resource.Object(BUCKET_NAME, key=features_file_name)
            s3_obj.put(Body=df_features.to_csv(index=False).encode("utf_8"))
        else:
            features_file_name = get_latest_file_name("features_")
            df_features = s3.get_dataframe_with_datetime_index(features_file_name, BUCKET_NAME)

        print("features_file_name", features_file_name)
        print("df_features", df_features)

        # TimeSeriesCrossValidation時のtrain,test,evalのindexを取得
        optimize_params = OptimizeParams(params)

        train_size = int(df_features.index.size * (1 - optimize_params.evaluate_ratio))
        df_train = df_features[:train_size]
        df_eval = df_features[train_size:]
        tscv_idx_dict = get_split_idx(df_train, optimize_params)
        for i in tscv_idx_dict:
            mlflow.log_metrics(tscv_idx_dict[i], step=i)
        mlflow.log_metric("eval_start_index", datetime.datetime.timestamp(df_eval.index[optimize_params.gap]))
        mlflow.log_metric("eval_end_index", datetime.datetime.timestamp(df_eval.index[-1]))

        ss_start_idx = datetime.datetime.fromtimestamp(tscv_idx_dict[0]["train_start_index"])
        ss_end_idx = datetime.datetime.fromtimestamp(tscv_idx_dict[0]["train_end_index"])
        print(tscv_idx_dict)
        print(
            "df_features1",
            df_features[ss_start_idx:ss_end_idx].copy(),
        )

        # StandardScalerの計算
        # TODO build時にはより広範なデータを学習に使用するため、再計算するのがいいかもしれない。
        # (以下はTimeSeriesCrossValidationの1つ目の区間を使用している)
        ss, _ = calc_standard_scaler(df_features[ss_start_idx:ss_end_idx].copy())
        _, df_standard = calc_standard_scaler(df_features, ss)
        with TemporaryDirectory() as td:
            pkl_path = os.path.join(td, "ss.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(ss, f)
            mlflow.log_artifact(pkl_path, "ss", run_id=args.mlflow_run_id)

        if skip:
            preprocessed_file_name = get_latest_file_name("preprocess_")
        else:
            preprocessed_file_name = f"data/preprocess_{now}.csv"
            s3_obj = s3_resource.Object(BUCKET_NAME, key=preprocessed_file_name)
            s3_obj.put(Body=df_standard.to_csv(index=False).encode("utf_8"))

        mlflow.log_param("preprocessed_data_name", preprocessed_file_name)


if __name__ == "__main__":
    main()
