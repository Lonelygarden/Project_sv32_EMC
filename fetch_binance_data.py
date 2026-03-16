import requests
import pandas as pd
import time

def get_binance_options_data():
    base_url = "https://eapi.binance.com"
    
    # 1. 获取所有期权标的信息
    info = requests.get(f"{base_url}/eapi/v1/exchangeInfo").json()
    symbols = [s['symbol'] for s in info['timezone'] if '251011' in s['symbol']] # 过滤 251011 到期
    
    # 2. 获取标的指数价格 (BTCUSDT)
    index_price = float(requests.get(f"{base_url}/eapi/v1/index", params={"underlying": "BTCUSDT"}).json()['indexPrice'])
    
    # 3. 获取深度/盘口数据来计算 Mid Price (比 Last Price 更准)
    option_chain = []
    for symbol in symbols:
        ticker = requests.get(f"{base_url}/eapi/v1/ticker", params={"symbol": symbol}).json()[0]
        # 解析 symbol: BTC-251011-60000-C
        parts = symbol.split('-')
        strike = float(parts[2])
        opt_type = parts[3]
        
        mid_price = (float(ticker['bidPrice']) + float(ticker['askPrice'])) / 2
        if mid_price > 0:
            option_chain.append({
                'symbol': symbol,
                'strike': strike,
                'type': opt_type,
                'mid_price': mid_price,
                'iv': float(ticker['markVolatility']) # 币安自带的 IV 可做参考
            })
            
    return pd.DataFrame(option_chain), index_price