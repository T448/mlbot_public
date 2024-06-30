# mlbot_public
## influxdbの設定
- パスワードの設定
- バケット作成
    - TODO 自動作成
    - 以下5つのバケットを手動で作成する
        - account : 口座残高等
        - ohlcv : ohlcvデータ
        - market : ohlcv以外のマーケットデータ
            - LS比
            - OpenInterest
        - trade : トレード履歴
            - 平均価格
            - 枚数
            - 指値価格
        - log : 定期実行の履歴

## Bot起動手順 (botコンテナ内で実行)
シンボリックリンクを設定

TODO DockerfileのCOPYを削除する、そうしないとシンボリックリンク張るときにエラーが出る

```bash
ln -s  /home/pyuser/project/docker/bot/cron1m.sh /root/script/cron1m.sh
ln -s  /home/pyuser/project/docker/bot/cron5m.sh /root/script/cron5m.sh
ln -s  /home/pyuser/project/docker/bot/cron15m.sh /root/script/cron15m.sh
```

crontabを設定 (PATHの設定がないと実行できない)

```bash
PYTHONPATH=/home/pyuser/project/src
PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

*/15 * * * * sh /root/script/cron15m.sh
*/5 * * * * sh /root/script/cron5m.sh
*/1 * * * * sh /root/script/cron1m.sh
```

最新のbot反映は`git pull origin main`で行う (github actions の deploy job も中身は同じ)

## その他
mlflow x nginx_proxy のためのホスト側での設定

https://zenn.dev/mjun0812/articles/192f4d4c5a14ab
```
cd nginx/htpasswd
htpasswd -c [domain名] [username]
```