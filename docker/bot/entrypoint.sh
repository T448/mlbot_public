#!/bin/sh

# cron 起動
service cron start

# 初回用
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/recorder.py --interval 1 --measurement_name bybit_btcusdt_1m --limit 10 >> /home/pyuser/log/recorder.log 2>&1
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/recorder.py --interval 5 --measurement_name bybit_btcusdt_5m --limit 10 >> /home/pyuser/log/recorder.log 2>&1
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/recorder.py --interval 15 --measurement_name bybit_btcusdt_15m --limit 10 >> /home/pyuser/log/recorder.log 2>&1

su pyuser
