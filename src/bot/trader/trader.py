import pickle
from typing import Any
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from os import environ

# import yaml
import ccxt
import mlflow
import pandas as pd
import datetime

from dotenv import load_dotenv
from bot.recorder.recorder import fetch_ohlcv
from bot.recorder.save_log import save_log
from entity.s3 import S3Client

# .envファイルの内容を読み込見込む
load_dotenv("/home/pyuser/project/docker/.env")

BYBIT_API_KEY = environ["BYBIT_API_KEY"]
BYBIT_SECRET = environ["BYBIT_SECRET"]
INFLUXDB_URL = environ["INFLUXDB_URL"]
ORGANIZATION = environ["INFLUXDB_ORGANIZATION"]
INFLUXDB_TOKEN = environ["INFLUXDB_TOKEN"]

# with open("trader.yml", "r") as yml:
#     trading_params = yaml.safe_load(yml)

LOT = 0.001
PAIR = "BTCUSDT"
MAX_PYRAMIDING = 3
# LOT = float(trading_params["trader"]["lot"])
# PAIR = trading_params["trader"]["pair"]
# MAX_PYRAMIDING = int(trading_params["trader"]["max-pyramiding"])

INTERVAL = 15
CATEGORY = "linear"
CLASSIFIER_NAME = "LGBMClassifier"
ALIAS = "champion"

client = influxdb_client.InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=ORGANIZATION)
write_api = client.write_api(write_options=SYNCHRONOUS)

exchange = ccxt.bybit(
    {
        "apiKey": BYBIT_API_KEY,
        "secret": BYBIT_SECRET,
    }
)

mlflow_client = mlflow.MlflowClient()
s3_client = S3Client()


def get_model() -> tuple[str, Any]:
    model_source = mlflow_client.get_model_version_by_alias(CLASSIFIER_NAME, ALIAS).source
    model_path = model_source.replace("s3://mlflow/", "") + "/python_model.pkl"
    model = s3_client.s3.get_object(Bucket=s3_client.bucket_name, Key=model_path)
    model = model["Body"].read()
    model = pickle.loads(model)
    return model_source, model


def main() -> None:
    save_log("trader", "start")

    # 現在の注文をキャンセル
    exchange.cancel_all_orders("BTCUSDT")
    save_log("trader", "cancel_all_orders")

    # 現在のポジション取得
    current_positions = exchange.fetch_positions("BTCUSDT")
    df_positions = pd.DataFrame(
        [
            [
                int(i["info"]["createdTime"]),
                i["info"]["side"],
                float(i["info"]["size"]),
                float(i["info"]["avgPrice"]),
                i["info"]["symbol"],
            ]
            for i in current_positions
        ],
        columns=["createdTime", "side", "size", "avgPrice", "symbol"],
    )

    df_positions.index = pd.DatetimeIndex(
        [datetime.datetime.fromtimestamp(int(i) / 1000) for i in df_positions["createdTime"]]
    )
    df_positions = df_positions[["side", "size", "avgPrice", "symbol"]]

    # 現在のポジションの平均価格、サイズをDBに保存
    if not df_positions.empty:
        write_api.write(
            bucket="trade",
            org=ORGANIZATION,
            record=df_positions,
            data_frame_measurement_name=f"{PAIR}_positions",
        )

    # ohlcv取得
    df_ohlcv = fetch_ohlcv(CATEGORY, PAIR, INTERVAL, 400)

    # 指値位置計算, 特徴量計算, ノーマライズ, 予測
    model_source, model = get_model()
    res = model.predict(df=df_ohlcv, context=None)
    buy_limit_price = res["buy_limit_price"]
    sell_limit_price = res["sell_limit_price"]

    # 注文
    buy_limit_order = "----------no order----------"
    sell_limit_order = "----------no order----------"

    if len(df_positions) == 0:
        buy_limit_order = exchange.create_order(PAIR, "limit", "buy", LOT, buy_limit_price)
        sell_limit_order = exchange.create_order(PAIR, "limit", "sell", LOT, sell_limit_price)
    else:
        current_position_size = df_positions["size"].values[0]
        current_position_side = df_positions["side"].values[0]

        if current_position_size < LOT * MAX_PYRAMIDING:
            if current_position_side == "Buy":
                buy_limit_order = exchange.create_order(PAIR, "limit", "buy", LOT, buy_limit_price)
                sell_limit_order = exchange.create_order(
                    PAIR, "limit", "sell", current_position_size + LOT, sell_limit_price
                )
            elif current_position_side == "Sell":
                buy_limit_order = exchange.create_order(
                    PAIR, "limit", "buy", current_position_size + LOT, buy_limit_price
                )
                sell_limit_order = exchange.create_order(PAIR, "limit", "sell", LOT, sell_limit_price)
        else:
            if current_position_side == "Buy":
                sell_limit_order = exchange.create_order(
                    PAIR, "limit", "sell", LOT * (1 + MAX_PYRAMIDING), sell_limit_price
                )
            elif current_position_side == "Sell":
                buy_limit_order = exchange.create_order(
                    PAIR, "limit", "buy", LOT * (1 + MAX_PYRAMIDING), buy_limit_price
                )

    print("buy_limit_order", buy_limit_order)
    print("sell_limit_order", sell_limit_order)

    # 注文内容をDBに保存
    open_orders = exchange.fetchOpenOrders(PAIR)
    df_orders = pd.DataFrame(
        [
            [i["info"]["createdTime"], i["info"]["side"], float(i["info"]["price"]), i["info"]["symbol"], model_source]
            for i in open_orders
        ],
        columns=["createdTime", "side", "price", "symbol", "model_source"],
    )
    df_orders.index = pd.DatetimeIndex(
        [datetime.datetime.fromtimestamp(int(i) / 1000) for i in df_orders["createdTime"]]
    )
    df_orders = df_orders[["side", "price", "symbol", "model_source"]]

    if not df_orders.empty:
        write_api.write(
            bucket="trade",
            org=ORGANIZATION,
            record=df_orders,
            data_frame_measurement_name=f"{PAIR}_orders",
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        save_log("trader", str(e))
