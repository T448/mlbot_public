FROM python:3.11.7-slim-bullseye

RUN apt update
RUN apt upgrade -y
RUN apt install build-essential -y
RUN apt install python3-dev -y
RUN apt install libpq-dev -y

RUN pip install -U pip
RUN pip install psycopg2 mlflow boto3
