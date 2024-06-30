import argparse
import requests
import pandas as pd
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from os import environ
from dotenv import load_dotenv
from bot.recorder.save_log import save_log

# .envファイルの内容を読み込見込む
load_dotenv("/home/pyuser/project/docker/.env")

BYBIT_API_KEY = environ["BYBIT_API_KEY"]
BYBIT_SECRET = environ["BYBIT_SECRET"]
INFLUXDB_URL = environ["INFLUXDB_URL"]
ORGANIZATION = environ["INFLUXDB_ORGANIZATION"]
INFLUXDB_TOKEN = environ["INFLUXDB_TOKEN"]

client = influxdb_client.InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=ORGANIZATION)
write_api = client.write_api(write_options=SYNCHRONOUS)


def fetch_ohlcv(category: str, symbol: str, interval: int, limit: int) -> pd.DataFrame:
    url = f"https://api.bybit.com/v5/market/kline?category={category}&symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url)

    column = ["time", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(res.json()["result"]["list"], columns=column)
    df.index = pd.DatetimeIndex(df["time"].astype("int") * 1e6)
    df = df[["open", "high", "low", "close", "volume"]]
    df = df.astype("float")
    df = df.sort_index()

    return df


def save_bybit_btcusdt_ohlcv(interval: int, measurement_name: str, limit: int = 50) -> None:
    category = "linear"
    symbol = "BTCUSDT"
    df = fetch_ohlcv(category=category, symbol=symbol, interval=interval, limit=limit)

    write_api.write(bucket="ohlcv", org=ORGANIZATION, record=df, data_frame_measurement_name=measurement_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval")
    parser.add_argument("--measurement_name")
    parser.add_argument("--limit")
    args = parser.parse_args()

    if args.interval is None:
        raise ValueError("interval がありません")

    if args.measurement_name is None:
        raise ValueError("measurement_name がありません")

    if args.limit is None:
        raise ValueError("limits がありません")

    interval = int(args.interval)
    measurement_name = str(args.measurement_name)
    limit = int(args.limit)

    try:
        save_bybit_btcusdt_ohlcv(interval=interval, measurement_name=measurement_name)
        save_log(f"OHLCV {args.interval}min", "OK")
    except Exception as e:
        save_log(f"OHLCV {args.interval}min", str(e))
