from typing import Any
import boto3
import os
import pandas as pd
import pickle

from sklearn.base import BaseEstimator


class S3Client:
    def __init__(self, bucket_name="mlflow"):
        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        self.bucket_name = bucket_name

    def get_pkl(self, run_id: str, file_path: str, bucket_name: str | None = None) -> BaseEstimator | Any:
        if bucket_name is None:
            bucket_name = self.bucket_name
        file_path = f"0/{run_id}/{file_path}"
        res = self.s3.get_object(Bucket=self.bucket_name, Key=file_path)
        obj = res["Body"].read()
        data: BaseEstimator | Any = pickle.loads(obj)
        return data

    def get_artifact_dataframe_with_datetime_index(self, run_id: str, file_path: str, bucket_name: str | None = None):
        """
        csvを取得
        MLflowのartifactの場合はこちらを使用する
        Args:
            run_id (str): _description_
            file_path (str): _description_
            bucket_name (str | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
        file_path = f"0/{run_id}/{file_path}"
        res = self.s3.get_object(Bucket=bucket_name, Key=file_path)
        df = pd.read_csv(res["Body"])
        df.index = pd.DatetimeIndex(df["date"].values)
        df = df[[i for i in df.columns if i != "date"]]

        return df

    def get_dataframe_with_datetime_index(self, file_path: str, bucket_name: str | None = None):
        """
        csvを取得
        MLflowのartifact以外の場合はこちらを使用する

        Args:
            run_id (str): _description_
            file_path (str): _description_
            bucket_name (str | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
        res = self.s3.get_object(Bucket=bucket_name, Key=file_path)
        df = pd.read_csv(res["Body"])
        df.index = pd.DatetimeIndex(df["date"].values)
        df = df[[i for i in df.columns if i != "date"]]

        return df
