import requests
import pandas as pd
import time
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.styles import Alignment

def fetch_deribit_trades_robust(start_time_str, end_time_str, currency="BTC"):
    """带有自动重试和防断流机制的稳定抓取器"""
    start_ts = int(pd.Timestamp(start_time_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_time_str).timestamp() * 1000)
    url = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
    
    trades = []
    current_start = start_ts
    
    while current_start < end_ts:
        params = {
            "currency": currency,
            "kind": "option",
            "start_timestamp": current_start,
            "end_timestamp": end_ts,
            "count": 1000,
        }
        
        # 增加最多 5 次的重试机制
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                break # 请求成功，跳出重试循环
            except Exception as e:
                print(f"  [网络波动] 抓取失败 ({e}). 第 {attempt+1}/{max_retries} 次重试中...")
                time.sleep(3) # 遇到阻击，休眠3秒后重试
        else:
            print("  [严重错误] 连续5次重试失败，保存已抓取数据并退出本时间段。")
            break # 放弃当前批次，保留已经拿到手的数据
            
        if 'result' not in data or not data['result']['trades']:
            break
            
        batch_trades = data['result']['trades']
        trades.extend(batch_trades)
        
        # 推进时间戳，防止死循环
        current_start = batch_trades[-1]['timestamp'] + 1
        print(f"  已成功安全抓取 {len(trades)} 条记录...", end='\r') 
        time.sleep(0.3) # 保护性限速
            
    if not trades:
        return pd.DataFrame()
        
    df = pd.DataFrame(trades)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    cols = ['datetime', 'instrument_name', 'price', 'amount', 'direction', 'index_price', 'iv']
    return df[[c for c in cols if c in df.columns]]

def process_and_filter_otm_dynamic(df):
    """基于每一笔交易瞬时 Moneyness 的动态 OTM 过滤"""
    if df.empty:
        return pd.DataFrame()
        
    parsed = df['instrument_name'].str.split('-', expand=True)
    df['Underlying'] = parsed[0]
    df['ExpiryStr'] = parsed[1]
    df['Strike'] = parsed[2].astype(float)
    df['Type'] = parsed[3]
    
    df['Expiry'] = pd.to_datetime(df['ExpiryStr'], format='%d%b%y')
    df['ExpiryTime'] = df['Expiry'] + pd.Timedelta(hours=8)
    df['T'] = (df['ExpiryTime'] - df['datetime']).dt.total_seconds() / (365 * 24 * 3600)
    
    df_valid = df[df['T'] > 0]
    if df_valid.empty:
        return pd.DataFrame()
        
    min_expiry = df_valid['Expiry'].min()
    df_shortest = df_valid[df_valid['Expiry'] == min_expiry].copy()
    
    # 1. 直接计算每笔交易瞬间的 Moneyness
    df_shortest['Moneyness'] = df_shortest['Strike'] / df_shortest['index_price']
    
    # 2. 动态分离 OTM
    # 只要交易发生时 K < S (Moneyness < 1.0)，就是价外 Put
    puts = df_shortest[(df_shortest['Type'] == 'P') & (df_shortest['Moneyness'] < 1.0)]
    # 只要交易发生时 K >= S (Moneyness >= 1.0)，就是价外 Call
    calls = df_shortest[(df_shortest['Type'] == 'C') & (df_shortest['Moneyness'] >= 1.0)]
    
    # 合并
    otm_df = pd.concat([puts, calls])
    
    # 3. 按 Moneyness 去重：同一时间段内，同一方向和差不多的 Moneyness，取最后一笔
    # 为了避免相同 Strike 在不同时刻成交导致 Moneyness 微小差异，我们直接用 Strike 和 Type 去重
    otm_df = otm_df.sort_values('datetime').groupby(['Strike', 'Type']).tail(1).reset_index(drop=True)
    
    # 按 Moneyness 排序，方便后续画出平滑的 Volatility Smile
    otm_df = otm_df.sort_values(by='Moneyness').reset_index(drop=True)
    
    return otm_df


def extract_long_term_otm(raw_csv_path, min_days=30, max_days=90):
    """
    从全量 Raw 数据中，提取到期时间在 min_days 到 max_days 之间的 OTM 期权
    """
    print(f"正在读取原始数据: {raw_csv_path} ...")
    df = pd.read_csv(raw_csv_path)
    
    if df.empty:
        print("数据为空！")
        return pd.DataFrame()
        
    # 1. 重新解析合约信息和时间 T
    df['datetime'] = pd.to_datetime(df['datetime'])
    parsed = df['instrument_name'].str.split('-', expand=True)
    df['Strike'] = parsed[2].astype(float)
    df['Type'] = parsed[3]
    df['Expiry'] = pd.to_datetime(parsed[1], format='%d%b%y')
    df['ExpiryTime'] = df['Expiry'] + pd.Timedelta(hours=8)
    
    # 计算年化到期时间 T
    df['T'] = (df['ExpiryTime'] - df['datetime']).dt.total_seconds() / (365 * 24 * 3600)
    
    # 2. 筛选远期期权 (例如 30天 到 90天)
    min_T = min_days / 365.0
    max_T = max_days / 365.0
    df_far = df[(df['T'] >= min_T) & (df['T'] <= max_T)].copy()
    
    if df_far.empty:
        print(f"警告：在这 5 分钟内，没有找到 {min_days}-{max_days} 天到期的期权成交记录。")
        print("建议放宽天数限制，比如尝试 min_days=14, max_days=180")
        return pd.DataFrame()
        
    # 看看选出了哪些到期日，选其中成交最活跃（数据量最大）的那个到期日
    most_liquid_expiry = df_far['Expiry'].value_counts().idxmax()
    print(f"选定最活跃的远期到期日: {most_liquid_expiry.strftime('%Y-%m-%d')}")
    
    df_target = df_far[df_far['Expiry'] == most_liquid_expiry].copy()
    
    # 3. 动态 Moneyness 截面清洗
    df_target['Moneyness'] = df_target['Strike'] / df_target['index_price']
    
    # 价外判定
    puts = df_target[(df_target['Type'] == 'P') & (df_target['Moneyness'] < 1.0)]
    calls = df_target[(df_target['Type'] == 'C') & (df_target['Moneyness'] >= 1.0)]
    otm_df = pd.concat([puts, calls])
    
    # 去重：按行权价和方向，保留最后一次成交
    otm_df = otm_df.sort_values('datetime').groupby(['Strike', 'Type']).tail(1).reset_index(drop=True)
    otm_df = otm_df.sort_values(by='Moneyness').reset_index(drop=True)
    
    print(f"成功提取远端 OTM 样本: {len(otm_df)} 个")
    return otm_df