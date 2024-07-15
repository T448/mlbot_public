from typing import Literal
import influxdb_client
import os
from flask.cli import load_dotenv
import pandas as pd
import datetime

# .envファイルの内容を読み込見込む
load_dotenv("/home/pyuser/project/docker/.env")
load_dotenv("/home/pyuser/project/docker/.env.dev")

print("influxdb_utils", os.environ)

INFLUXDB_URL = os.environ["INFLUXDB_URL"]
ORGANIZATION = os.environ["INFLUXDB_ORGANIZATION"]
INFLUXDB_TOKEN = os.environ["INFLUXDB_TOKEN"]

client = influxdb_client.InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=ORGANIZATION)


def get_ohlcv(start: Literal["15m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w", "1y"]) -> pd.DataFrame:
    """
    influxdbに保存中の生のohlcvデータを取得
    start分だけ遡って取得
    """
    query_api = client.query_api()

    query = f"""from(bucket: "ohlcv")
    |> range(start: -{start})
    |> filter(fn: (r) =>
        r._measurement == "bybit_btcusdt_15m" and
        (r._field == "close" or r._field == "open" or r._field == "high" or r._field == "low" or r._field == "volume")
    )"""

    tables = query_api.query(query=query, org=ORGANIZATION)

    columns = [i.label for i in tables[0].columns]
    df = pd.DataFrame(tables.to_values(), columns=columns)

    def get_selected_field_df(_df: pd.DataFrame, field: str) -> pd.DataFrame:
        if "_field" not in _df.columns:
            raise ValueError

        df = _df.copy()
        df = df[df["_field"] == field]
        # FIXME ほかの書き方があるはず
        idx = [datetime.datetime.strptime(i.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") for i in df["_time"]]
        df.index = pd.DatetimeIndex(idx, name="date")
        df = df[["_value"]]
        df.columns = [field]  # type:ignore
        return df

    df_open = get_selected_field_df(df, "open")
    df_high = get_selected_field_df(df, "high")
    df_low = get_selected_field_df(df, "low")
    df_close = get_selected_field_df(df, "close")
    df_volume = get_selected_field_df(df, "volume")

    return pd.concat([df_open, df_high, df_low, df_close, df_volume], axis=1)
