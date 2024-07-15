#!/bin/sh

# 定期実行したい処理
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/recorder.py --interval 1 --measurement_name bybit_btcusdt_1m --limit 10 >> /home/pyuser/log/recorder.log 2>&1
