import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.wallet import get_bybit_wallet_balance
from dotenv import load_dotenv
from os import environ
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import datetime
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


def main():
    timestamp, wallet_balance = get_bybit_wallet_balance(BYBIT_API_KEY, BYBIT_SECRET)
    df = pd.DataFrame([float(wallet_balance)])
    df.columns = ["balance"]
    df.index = pd.DatetimeIndex([1e6 * timestamp])

    write_api.write(bucket="account", org=ORGANIZATION, record=df, data_frame_measurement_name="wallet_balance")


if __name__ == "__main__":
    try:
        main()
        print(datetime.datetime.now(), "wallet balance")
        save_log("Wallet Balance 15min", "OK")
    except Exception as e:
        save_log("Wallet Balance 15min", str(e))
