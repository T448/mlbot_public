#!/bin/sh

# 定期実行したい処理
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/recorder.py --interval 15 --measurement_name bybit_btcusdt_15m --limit 10 >> /home/pyuser/log/recorder.log 2>&1
/usr/local/bin/python /home/pyuser/project/src/bot/recorder/wallet_balance_recorder.py >> /home/pyuser/log/recorder.log 2>&1
/usr/local/bin/python /home/pyuser/project/src/bot/trader/trader.py >> /home/pyuser/log/recorder.log 2>&1