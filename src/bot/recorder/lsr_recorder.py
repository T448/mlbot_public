import requests
import pandas as pd
import datetime
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


def get_lsr() -> pd.DataFrame | None:
    """
    Longの比率を返す
    失敗した場合はNone

    Returns:
        pd.DataFrame | None: _description_
    """
    category = "linear"
    symbol = "BTCUSDT"
    period = "5min"
    limit = "10"
    query = f"?category={category}&symbol={symbol}&period={period}&limit={limit}"
    url = f"https://api.bybit.com/v5/market/account-ratio{query}"

    res = requests.get(url)

    if res.status_code != 200:
        return None

    df = pd.DataFrame(res.json()["result"]["list"])
    df.index = pd.DatetimeIndex(df["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(int(x) / 1000)))
    df = df[["buyRatio"]].astype("float")
    df.columns = pd.Index(["LSR"])
    return df


def main() -> None:
    data = get_lsr()
    if data is not None:
        write_api.write(bucket="market", org=ORGANIZATION, record=data, data_frame_measurement_name="BTCUSDT_LSR")


if __name__ == "__main__":
    try:
        main()
        save_log("LSR 5min", "OK")
    except Exception as e:
        save_log("LSR 5min", str(e))
