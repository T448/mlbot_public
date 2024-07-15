from flask.cli import load_dotenv
import psycopg2
import pandas as pd
from os import environ

# .envファイルの内容を読み込見込む
load_dotenv("/home/pyuser/project/docker/.env")


def get_connection():
    POSTGRES_HOST = environ["POSTGRES_HOST"]
    POSTGRES_DB = environ["POSTGRES_DB"]
    POSTGRES_USER = environ["POSTGRES_USER"]
    POSTGRES_PASSWORD = environ["POSTGRES_PASSWORD"]
    POSTGRES_PORT = environ["POSTGRES_PORT"]
    dsn = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    print(dsn)
    return psycopg2.connect(dsn)


def select(query):
    result = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchall()

    df = pd.DataFrame(result)
    df.columns = [i.name for i in cur.description]
    return df


def get_last_run_commit_hash() -> str:
    # preprocess,train_1st,train_2md,build,evaluate以外 かつ 1つ前の親run
    query_runs = "select * from runs order by start_time desc limit 12"

    df_runs = select(query_runs)

    df_parent_runs = df_runs[
        (df_runs["name"] != "preprocess")
        & (df_runs["name"] != "train_1st")
        & (df_runs["name"] != "train_2nd")
        & (df_runs["name"] != "build")
        & (df_runs["name"] != "evaluate")
    ]

    if df_parent_runs.size == 0:
        return ''

    previous_parent_run_uuid = df_parent_runs["run_uuid"].values[1]

    query_tags = (
        f"select * from tags where run_uuid = '{previous_parent_run_uuid}' and key = 'mlflow.source.git.commit'"
    )
    df_tags = select(query_tags)
    if df_tags.size == 0:
        return ''

    return df_tags["value"].values[0]
