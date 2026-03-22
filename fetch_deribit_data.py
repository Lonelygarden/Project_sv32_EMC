import requests
import pandas as pd
import numpy as np
import time
import datetime
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from scipy.stats import linregress

from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_deribit_trades_fast(start_time_str, end_time_str, currency="BTC"):
    """多线程 + 指数退避机制"""
    start_ts = int(pd.Timestamp(start_time_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_time_str).timestamp() * 1000)
    url = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
    
    session = requests.Session()
    
    def fetch_chunk(chunk_start, chunk_end):
        chunk_trades = []
        curr_start = chunk_start
        
        while curr_start < chunk_end:
            params = {
                "currency": currency,
                "kind": "option",
                "start_timestamp": curr_start,
                "end_timestamp": chunk_end,
                "count": 1000,
                "sorting": "asc"  # 💥 强制正序滚动，防止游标错乱
            }
            
            success = False
            # 最多重试 5 次
            for attempt in range(5):
                try:
                    resp = session.get(url, params=params, timeout=15)
                    
                    # 拦截 429 频率限制报错
                    if resp.status_code == 429:
                        time.sleep(2 * (attempt + 1)) # 被限制了就多睡一会儿
                        continue
                        
                    resp.raise_for_status() # 拦截 502/504 等服务器网关报错
                    data = resp.json()
                    
                    # 拦截 API 内部的报错体
                    if 'error' in data:
                        time.sleep(2 * (attempt + 1))
                        continue
                        
                    success = True
                    break # 成功拿到数据，跳出重试循环
                    
                except Exception as e:
                    time.sleep(2) # 遇到网络抖动，休眠 2 秒
            
            if not success:
                print(f"\n  [警告] 时间块 {chunk_start} 连续 5 次请求失败，请检查网络！")
                break # 该块彻底绝望，只能退出

            batch = data.get('result', {}).get('trades', [])
            if not batch:
                break
                
            chunk_trades.extend(batch)
            
            # 如果拿到的不够 1000 条，说明这块彻底扫空了
            if len(batch) < 1000:
                break 
                
            # 否则沿着最后一条的时间戳继续往下推
            curr_start = batch[-1]['timestamp'] + 1
            
        return chunk_trades

    # 将块大小定为 5 分钟 (300,000 毫秒)，减少 HTTP 连接建立次数
    chunk_size_ms = 300 * 1000 
    intervals = []
    c_start = start_ts
    while c_start < end_ts:
        c_end = min(c_start + chunk_size_ms, end_ts)
        intervals.append((c_start, c_end))
        c_start = c_end
        
    all_trades = []
    
    print(f"  [引擎启动] 划分为 {len(intervals)} 个并发块，安全全速抓取...")
    
    # 将 max_workers 降为 4，达到极限效率与不被封 IP 的完美平衡
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_chunk, s, e): (s, e) for s, e in intervals}
        
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            result = future.result()
            all_trades.extend(result)
            print(f"  [并发进度] 已完成 {completed_count}/{len(intervals)} 块 (累计获取 {len(all_trades)} 条成交记录)...", end='\r')
            
    print("\n  所有任务结束，正在重组与去重...")
    
    if not all_trades:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_trades)
    
    # 依靠底层 trade_seq 去重，绝对严谨
    df = df.drop_duplicates(subset=['trade_seq']) 
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    cols = ['datetime', 'instrument_name', 'price', 'amount', 'direction', 'index_price', 'iv']
    return df[[c for c in cols if c in df.columns]]


def fetch_dvol_data(end_date_str, days_lookback=180):
    """
    通过 Deribit API 获取 DVOL 数据，
    并使用严谨的 3/2 模型离散化回归估算 kappa 和 theta
    """
    print(f"正在从 Deribit 抓取 {end_date_str} 前 {days_lookback} 天的 BTC DVOL 数据...")
    
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    start_date = end_date - datetime.timedelta(days=days_lookback)
    
    end_ts = int(end_date.timestamp() * 1000)
    start_ts = int(start_date.timestamp() * 1000)
    
    url = f"https://deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp={start_ts}&end_timestamp={end_ts}&resolution=1D"
    
    response = requests.get(url)
    if response.status_code != 200:
        print("❌ API 请求失败")
        return None, None
        
    data = response.json().get('result', {}).get('data', [])
    if not data:
        print("❌ 未获取到数据")
        return None, None
        
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['V_t'] = (df['close'] / 100.0) ** 2
    df.to_excel("DVOL_data.xlsx", index=False)  # 保存原始数据到 Excel
    return df
    
    

def estimate_32_parameters():
    # 进行分析
    df = pd.read_excel("DVOL_data.xlsx")
    V = df['V_t'].values
    if len(V) < 10:
        print("❌ 数据样本太少，无法回归")
        return None, None
    # ==========================================
    # 构建 3/2 模型的回归变量
    # Y = (V_t - V_{t-1}) / V_{t-1}
    # X = V_{t-1}
    # ==========================================
    Y = (V[1:] - V[:-1]) / V[:-1]
    X = V[:-1]
    
    # 运行一元线性回归
    beta, alpha, r_value, p_value, std_err = linregress(X, Y)
    
    dt = 1.0 / 365.0
    
    # 根据数学推导反解物理参数
    kappa_estimate = -beta / dt
    
    if beta != 0:
        theta_estimate = -alpha / beta
    else:
        theta_estimate = np.mean(V) # 兜底防除零
        
    # ==========================================
    # 打印学术级报告
    # ==========================================
    print(f"✅ 3/2 模型专属离散化回归完成！(样本数: {len(X)} 天)")
    print("-" * 45)
    print(f"统计区间: {df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"回归方程: dV/V = ({alpha:.4f}) + ({beta:.4f}) * V_prev")
    print("-" * 45)
    print(f"🎯 估计的长期方差 (Theta):    {theta_estimate:.6f} (约 {np.sqrt(theta_estimate)*100:.1f}% IV)")
    print(f"🎯 估计的均值回归速度 (Kappa): {kappa_estimate:.4f}")
    print("-" * 45)
    print(f"[附] 回归 R^2: {r_value**2:.4f}, p-value(斜率): {p_value:.2e}")
    
    if kappa_estimate <= 0:
        print("⚠️ 警告: Kappa 为负，说明该时间段内方差表现为发散而非均值回归！")
        
    return kappa_estimate, theta_estimate