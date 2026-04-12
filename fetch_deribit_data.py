import requests
import pandas as pd
import numpy as np
import time
import datetime
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import os
import matplotlib.pyplot as plt
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
RAW_FILES = {
    "1_Pre_Shock": "Deribit_RAW_1_Pre_Shock.csv",
    "2_Crash":     "Deribit_RAW_2_Crash.csv",
    "3_Recovery":  "Deribit_RAW_3_Recovery.csv"
}

# 选取流动性最好的20天期限作为核心分析对象
TARGET_MATURITIES = {
    "20D": 20.3 / 365.25,
    "6D": 6 / 365.26
}

def fetch_deribit_trades_fast(start_time_str, end_time_str, currency="BTC"):
    """多线程 + 指数退避机制"""
    start_ts = int(pd.Timestamp(start_time_str, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_time_str, tz="UTC").timestamp() * 1000)
    url = (
        "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
    )

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
                "sorting": "asc",  # 💥 强制正序滚动，防止游标错乱
            }

            success = False
            # 最多重试 5 次
            for attempt in range(5):
                try:
                    resp = session.get(url, params=params, timeout=15)

                    # 拦截 429 频率限制报错
                    if resp.status_code == 429:
                        time.sleep(2 * (attempt + 1))  # 被限制了就多睡一会儿
                        continue

                    resp.raise_for_status()  # 拦截 502/504 等服务器网关报错
                    data = resp.json()

                    # 拦截 API 内部的报错体
                    if "error" in data:
                        time.sleep(2 * (attempt + 1))
                        continue

                    success = True
                    break  # 成功拿到数据，跳出重试循环

                except Exception as e:
                    time.sleep(2)  # 遇到网络抖动，休眠 2 秒

            if not success:
                print(
                    f"\n  [警告] 时间块 {chunk_start} 连续 5 次请求失败，请检查网络！"
                )
                break  # 该块彻底绝望，只能退出

            batch = data.get("result", {}).get("trades", [])
            if not batch:
                break

            chunk_trades.extend(batch)

            # 如果拿到的不够 1000 条，说明这块彻底扫空了
            if len(batch) < 1000:
                break

            # 否则沿着最后一条的时间戳继续往下推
            curr_start = batch[-1]["timestamp"] + 1

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
            print(
                f"  [并发进度] 已完成 {completed_count}/{len(intervals)} 块 (累计获取 {len(all_trades)} 条成交记录)...",
                end="\r",
            )

    print("\n  所有任务结束，正在重组与去重...")

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)

    # 依靠底层 trade_seq 去重，绝对严谨
    df = df.drop_duplicates(subset=["trade_seq"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    cols = [
        "datetime",
        "instrument_name",
        "price",
        "amount",
        "direction",
        "index_price",
        "iv",
    ]
    return df[[c for c in cols if c in df.columns]]


def fetch_dvol_data(end_date_str, days_lookback=180):
    """
    通过 Deribit API 获取 DVOL 数据 (全链路严格 UTC 时间)，
    并使用严谨的 3/2 模型离散化回归估算 kappa 和 theta
    """
    print(
        f"正在从 Deribit 抓取 {end_date_str} 前 {days_lookback} 天的 BTC DVOL 数据 (UTC)..."
    )

    end_date = pd.Timestamp(end_date_str, tz="UTC")
    start_date = end_date - pd.Timedelta(days=days_lookback)

    end_ts = int(end_date.timestamp() * 1000)
    start_ts = int(start_date.timestamp() * 1000)

    url = f"https://deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp={start_ts}&end_timestamp={end_ts}&resolution=1D"

    response = requests.get(url)
    if response.status_code != 200:
        print("❌ API 请求失败")
        return None

    data = response.json().get("result", {}).get("data", [])
    if not data:
        print("❌ 未获取到数据")
        return None

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    
    # 🌟 核心修复：转化为 UTC 后，去掉时区尾巴，骗过 Excel
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    
    df["V_t"] = (df["close"] / 100.0) ** 2
    
    cols = ["date", "timestamp", "open", "high", "low", "close", "V_t"]
    df = df[cols]
    
    df.to_excel("DVOL_data.xlsx", index=False)
    print(f"✅ 成功获取 {len(df)} 天的 DVOL 数据并保存。")
    return df


def clean_and_split_smart():
    for stage_name, raw_file in RAW_FILES.items():
        print(f"\n📂 正在读取并智能提纯: {raw_file}")
        
        if not os.path.exists(raw_file):
            print(f"  ❌ 文件不存在，跳过。")
            continue
            
        df = pd.read_csv(raw_file)
        
        # 1. 拆解参数
        parts = df['instrument_name'].str.split('-', expand=True)
        df['Expiry_Str'] = parts[1]
        df['K'] = parts[2].astype(float)
        df['Type'] = parts[3]
        
        # 2. 时间处理
        # 明确告诉 Pandas，Trade_Time 解析为 UTC 时间
        df['Trade_Time'] = pd.to_datetime(df['datetime'], format='mixed', utc=True)
        df['Expiry_Time'] = pd.to_datetime(df['Expiry_Str'], format='%d%b%y').dt.tz_localize('UTC') + pd.Timedelta(hours=8)
        df['T'] = (df['Expiry_Time'] - df['Trade_Time']).dt.total_seconds() / (365.25 * 24 * 3600)
        
        # 3. 聚合去重 (注意这里加入了 Expiry_Str 作为保留字段)
        df_agg = df.groupby(['instrument_name', 'Expiry_Str']).agg({
            'index_price': 'mean',
            'iv': 'mean',
            'K': 'first',
            'Type': 'first',
            'T': 'mean'
        }).reset_index()
        
        # 4. 提取 OTM 虚值期权
        df_agg['Moneyness'] = df_agg['K'] / df_agg['index_price']
        cond_call = (df_agg['Type'] == 'C') & (df_agg['Moneyness'] >= 1.0)
        cond_put = (df_agg['Type'] == 'P') & (df_agg['Moneyness'] <= 1.0)
        df_otm = df_agg[cond_call | cond_put].copy()
        
        if df_otm.empty:
            print(f"  ⚠️ 没有有效的虚值期权！")
            continue

        # 💥 核心修复：计算每个【交割日】的平均 T，而不是把微小差异的 T 当作独立的期限
        expiry_T_mapping = df_otm.groupby('Expiry_Str')['T'].mean()
        
        for maturity_label, target_T in TARGET_MATURITIES.items():
            best_expiry = None
            min_diff = float('inf')
            
            # 遍历每一个交割日 (比如 '31OCT25')
            for exp_str, t_mean in expiry_T_mapping.items():
                
                # 统一放宽截断区间，保留核心交战区 (M 在 0.5 到 2.0 之间)
                subset = df_otm[(df_otm['Expiry_Str'] == exp_str) & 
                                (df_otm['Moneyness'] >= 0.5) & 
                                (df_otm['Moneyness'] <= 2.0)]
                
                # 只有样本数 >= 3，我们才认为它能构成一条微笑曲线
                if len(subset) >= 3:
                    diff = np.abs(t_mean - target_T)
                    if diff < min_diff:
                        min_diff = diff
                        best_expiry = exp_str
            
            if best_expiry is None:
                print(f"  ⚠️ [{maturity_label}] 失败！没有任何一个期限包含 >= 3 个虚值期权。")
                pd.DataFrame(columns=['Moneyness', 'Type', 'iv', 'T']).to_csv(f"Deribit_{maturity_label}_{stage_name}.csv", index=False)
                continue
                
            # 提取最终的最佳数据集
            df_slice = df_otm[(df_otm['Expiry_Str'] == best_expiry) & 
                              (df_otm['Moneyness'] >= 0.5) & 
                              (df_otm['Moneyness'] <= 2.0)].copy()
                
            df_final = df_slice[['Moneyness', 'Type', 'iv', 'T']].copy()
            df_final = df_final.sort_values('Moneyness').reset_index(drop=True)
            
            output_filename = f"Deribit_{maturity_label}_{stage_name}.csv"
            df_final.to_csv(output_filename, index=False)
            
            actual_days = expiry_T_mapping[best_expiry] * 365.25
            print(f"  ✅ [{maturity_label}] 导出至: {output_filename}")
            print(f"     -> 锚定交割日: {best_expiry} | 匹配天数: {actual_days:.2f}天 | 有效样本数: {len(df_final)}")


def estimate_32_parameters():
    # 数据读取与预处理（保留你的逻辑）
    df = pd.read_excel("DVOL_data.xlsx")
    V = df["V_t"].values
    V = np.where(V <= 0, 1e-8, V)
    if len(V) < 10:
        print("❌ 数据样本太少，无法回归")
        return None, None
    
    # 构建回归变量
    dt = 1.0 / 365.25
    Y = (V[1:] - V[:-1]) / (V[:-1] ** 1.5)
    X1 = 1 / (V[:-1] ** 0.5) * dt
    X2 = V[:-1] ** 0.5 * dt
    X = pd.DataFrame({"X1 (κθ)": X1, "X2 (κ)": X2})
    
    # 运行回归
    model = sm.OLS(Y, X)
    results = model.fit()

    # ========== 核心：生成Stata风格表格 ==========
    # 1. 基础版（单模型，完整统计信息）
    print("\n 基础Stata风格回归表（statsmodels原生）")
    print("="*80)
    print(results.summary(xname=["X1 (κθ)", "X2 (κ)"]))  # 自定义变量名
    
    # 2. 精简版（仅核心指标，更贴近Stata简洁风格）
    print("\n 精简版回归表")
    print("="*80)
    summary = summary_col(
        [results],
        stars=True,  # 显示显著性星号（* p<0.1, ** p<0.05, *** p<0.01）
        float_format="%.6f",  # 数值格式
        info_dict={
            'N': lambda x: f"{int(x.nobs)}",
            'R²': lambda x: f"{x.rsquared:.4f}",
            'Adj. R²': lambda x: f"{x.rsquared_adj:.4f}"
        }
    )
    print(summary)

    # 反解参数（保留你的逻辑）
    kappa = -results.params["X2 (κ)"]
    kappa_theta = results.params["X1 (κθ)"]
    theta = kappa_theta / kappa if kappa != 0 else np.nan
    
    return kappa, theta

def liquidity_xray(raw_csv, description=""):
    print(f"正在透视底层数据: {raw_csv}")
    df = pd.read_csv(raw_csv)
    
    # 拆解参数
    parts = df['instrument_name'].str.split('-', expand=True)
    df['Expiry_Str'] = parts[1]
    df['K'] = parts[2].astype(float)
    df['Type'] = parts[3]
    
    # 明确读取为 UTC 时间
    df['Trade_Time'] = pd.to_datetime(df['datetime'], format='mixed', utc=True)
    
    # 解析交割日，打上 UTC 标签，再加上早上 8 点的固定交割时间
    df['Expiry_Time'] = pd.to_datetime(df['Expiry_Str'], format='%d%b%y').dt.tz_localize('UTC') + pd.Timedelta(hours=8)
    
    # 计算精确的“距离到期天数 (Days to Expiry)”
    df['T_days'] = (df['Expiry_Time'] - df['Trade_Time']).dt.total_seconds() / (24 * 3600)
    df['Moneyness'] = df['K'] / df['index_price']
    
    # 提取虚值期权 (OTM)
    cond_call = (df['Type'] == 'C') & (df['Moneyness'] >= 1.0)
    cond_put = (df['Type'] == 'P') & (df['Moneyness'] <= 1.0)
    df_otm = df[cond_call | cond_put].copy()
    
    # ==========================================
    # 打印流动性最集中的真实期限
    # ==========================================
    print("\n 按有效虚值成交笔数排名:")
    # 将天数四舍五入为整数
    top_expiries = df_otm['T_days'].round(0).value_counts().head(5)
    for t_days, count in top_expiries.items():
        print(f"   -> 距离到期 {t_days:>5.1f} 天 | 虚值成交量: {count} 笔")

    # ==========================================
    # 绘制 X-Ray 透视图
    # ==========================================
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 6), dpi=120)
    
    # 画出所有的 OTM 散点
    plt.scatter(df_otm['T_days'], df_otm['Moneyness'], 
                alpha=0.3, s=15, c='#1f77b4', edgecolor='none')
    
    plt.title(f'Deribit OTM Option Trades Distribution, {description}', fontsize=14, fontweight='bold')
    plt.xlabel('Days to Expiration (T)', fontsize=12)
    plt.ylabel('Moneyness (K/S)', fontsize=12)
    plt.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='ATM (M=1.0)')
    plt.savefig(f'Deribit_OTM_Option_Trades_Distribution_{description}', bbox_inches='tight')
    # 限制 Y 轴只看核心区，防止极端脏数据拉坏比例
    plt.ylim(0.5, 2.0)
    plt.legend()
    plt.show()