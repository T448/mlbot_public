import pandas as pd
import datetime
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from os import environ
from dotenv import load_dotenv

# .envファイルの内容を読み込見込む
load_dotenv("/home/pyuser/project/docker/.env")

BYBIT_API_KEY = environ["BYBIT_API_KEY"]
BYBIT_SECRET = environ["BYBIT_SECRET"]
INFLUXDB_URL = environ["INFLUXDB_URL"]
ORGANIZATION = environ["INFLUXDB_ORGANIZATION"]
INFLUXDB_TOKEN = environ["INFLUXDB_TOKEN"]

client = influxdb_client.InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=ORGANIZATION)
write_api = client.write_api(write_options=SYNCHRONOUS)


def save_log(name: str, message: str) -> None:
    now = datetime.datetime.now(tz=datetime.UTC)

    data = {"name": name, "message": message}
    df = pd.DataFrame(data, index=pd.DatetimeIndex([now]))

    write_api.write(bucket="log", org=ORGANIZATION, record=df, data_frame_measurement_name="log")
