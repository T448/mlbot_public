#!/bin/sh

# 定期実行したい処理
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/recorder.py --interval 5 --measurement_name bybit_btcusdt_5m --limit 10 >> /home/pyuser/log/recorder.log 2>&1
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/lsr_recorder.py >> /home/pyuser/log/recorder.log 2>&1
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/oi_recorder.py >> /home/pyuser/log/recorder.log 2>&1
