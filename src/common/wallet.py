import time
from urllib.parse import urlencode, quote_plus
import urllib3
import hmac
import hashlib
import requests


def get_bybit_wallet_balance(access_key: str, secret_key: str) -> tuple[float, float]:
    """
    bybit の wallet balance を取得

    Args:
        access_key (str): BYBIT API ACCESS KEY
        secret_key (str): BYBIT API SECRET

    Returns:
        tuple[float, float]: timestamp, balance
    """
    ACCOUNT_TYPE = "UNIFIED"

    params = {
        "api_key": access_key,
        "timestamp": round(time.time() * 1000),
        "recv_window": 10000,
        "accountType": ACCOUNT_TYPE,
    }

    # Create the param str
    param_str = urlencode(sorted(params.items(), key=lambda tup: tup[0]))

    # Generate the signature
    hash = hmac.new(bytes(secret_key, "utf-8"), param_str.encode("utf-8"), hashlib.sha256)

    signature = hash.hexdigest()
    sign_real = {"sign": signature}

    param_str = quote_plus(param_str, safe="=&")
    full_param_str = f"{param_str}&sign={sign_real['sign']}"

    # Request information
    url = "https://api.bybit.com/v5/account/wallet-balance"
    headers = {"Content-Type": "application/json"}

    # body = dict(params, **sign_real)
    urllib3.disable_warnings()

    response = requests.get(f"{url}?{full_param_str}", headers=headers, verify=False)

    if response.status_code == 200:
        data = response.json()
        return data["time"], float(data["result"]["list"][0]["totalWalletBalance"])
    else:
        return response.json()["time"], -99999
