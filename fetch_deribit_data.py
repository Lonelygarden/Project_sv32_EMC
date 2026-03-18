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


import requests
import pandas as pd
import numpy as np
from scipy.stats import linregress
import datetime

def fetch_dvol_and_estimate_parameters(end_date_str, days_lookback=180):
    """
    通过 Deribit API 获取 DVOL 数据并估计 3/2 模型的 kappa 和 theta
    end_date_str: 闪崩发生的日期，例如 '2024-10-11'。我们会取这之前的数据，避免未来函数。
    days_lookback: 回溯天数，默认半年(180天)
    """
    print(f"📥 正在从 Deribit 抓取闪崩前 {days_lookback} 天的 BTC DVOL 数据...")
    
    # 1. 计算时间戳
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    start_date = end_date - datetime.timedelta(days=days_lookback)
    
    end_ts = int(end_date.timestamp() * 1000)
    start_ts = int(start_date.timestamp() * 1000)
    
    # 2. 调用 Deribit DVOL 历史接口 (1D resolution)
    url = f"https://deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp={start_ts}&end_timestamp={end_ts}&resolution=1D"
    
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ API 请求失败")
        return None, None
        
    data = response.json().get('result', {}).get('data', [])
    if not data:
        print("❌ 未获取到数据")
        return None, None
        
    # 3. 解析数据为 DataFrame
    # Deribit DVOL 返回格式: [timestamp, open, high, low, close]
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 4. 计算每日方差 V_t
    # close 是 DVOL 的收盘值 (百分比，例如 55.4 代表 55.4%)
    df['V_t'] = (df['close'] / 100.0) ** 2
    
    V = df['V_t'].values
    df.to_excel("DVOL_data.xlsx", index=False)  # 保存原始数据到 Excel
    if len(V) < 10:
        print("❌ 数据样本太少，无法回归")
        return None, None
        
    # ==========================================
    # 5. 核心计算：计算 Theta
    # ==========================================
    theta_estimate = np.mean(V)
    
    # ==========================================
    # 6. 核心计算：AR(1) 回归计算 Kappa
    # ==========================================
    # 构建 V_t 和 V_{t-1} 序列
    Y = V[1:]   # V_t
    X = V[:-1]  # V_{t-1}
    
    # 进行一元线性回归 Y = alpha + beta * X
    beta, alpha, r_value, p_value, std_err = linregress(X, Y)
    
    # 计算 dt (日度数据年化)
    dt = 1.0 / 365.0
    
    # 根据 beta 反推 kappa
    if beta > 0 and beta < 1:
        kappa_estimate = -np.log(beta) / dt
    else:
        print(f"⚠️ 警告: 回归得到的 beta={beta:.4f} 不在 (0,1) 区间，均值回归特性可能不显著。")
        # 如果出现非稳态，使用近似公式 (1 - beta) / dt
        kappa_estimate = (1.0 - beta) / dt

    # ==========================================
    # 7. 打印报告
    # ==========================================
    print(f"✅ 数据抓取与计量回归完成！(样本数: {len(V)} 天)")
    print("-" * 40)
    print(f"统计区间: {df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"DVOL 均值:  {df['close'].mean():.2f}%")
    print("-" * 40)
    print(f"🎯 估计的长期方差 (Theta):    {theta_estimate:.6f}")
    print(f"🎯 估计的均值回归速度 (Kappa): {kappa_estimate:.4f}")
    print("-" * 40)
    print(f"[附] 回归 R^2: {r_value**2:.4f}, p-value: {p_value:.2e}")
    
    return kappa_estimate, theta_estimate