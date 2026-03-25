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

from concurrent.futures import ThreadPoolExecutor, as_completed


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
    X = pd.DataFrame({"X1 (κθ·dt)": X1, "X2 (κ·dt)": X2})
    
    # 运行回归
    model = sm.OLS(Y, X)
    results = model.fit()

    # ========== 核心：生成Stata风格表格 ==========
    # 1. 基础版（单模型，完整统计信息）
    print("\n 基础Stata风格回归表（statsmodels原生）")
    print("="*80)
    print(results.summary(xname=["X1 (κθ·dt)", "X2 (κ·dt)"]))  # 自定义变量名
    
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
    kappa = -results.params["X2 (κ·dt)"]
    kappa_theta = results.params["X1 (κθ·dt)"]
    theta = kappa_theta / kappa if kappa != 0 else np.nan
    
    return kappa, theta


def estimate_heston_parameters():
    # 1. 数据读取与预处理
    # 假设 Excel 结构一致，V_t 代表波动率或方差数据
    df = pd.read_excel("DVOL_data.xlsx")
    V = df["V_t"].values
    
    # CIR过程要求 V > 0，处理极小值避免除以0
    V = np.where(V <= 0, 1e-8, V)
    
    if len(V) < 10:
        print("❌ 数据样本太少，无法回归")
        return None, None
    
    # 2. 构建 Heston (CIR) 回归变量
    dt = 1.0 / 365.25
    
    # 因变量: (V_{t+1} - V_t) / sqrt(V_t)
    Y = (V[1:] - V[:-1]) / np.sqrt(V[:-1])
    
    # 自变量 X1: dt / sqrt(V_t) -> 对应参数 kappa * theta
    X1 = (1.0 / np.sqrt(V[:-1])) * dt
    
    # 自变量 X2: sqrt(V_t) * dt -> 对应参数 -kappa
    X2 = np.sqrt(V[:-1]) * dt
    
    X = pd.DataFrame({
        "X1 (κθ·dt)": X1, 
        "X2 (-κ·dt)": X2
    })
    
    # 3. 运行回归 (无截距项，因为模型已完全参数化)
    model = sm.OLS(Y, X)
    results = model.fit()

    # 4. 生成 Stata 风格表格
    print("\n [Heston/CIR] 基础Stata风格回归表")
    print("="*80)
    print(results.summary(xname=["X1 (κθ·dt)", "X2 (-κ·dt)"])) 
    
    print("\n [Heston/CIR] 精简版回归表")
    print("="*80)
    summary = summary_col(
        [results],
        stars=True,
        float_format="%.6f",
        info_dict={
            'N': lambda x: f"{int(x.nobs)}",
            'R²': lambda x: f"{x.rsquared:.4f}",
            'Adj. R²': lambda x: f"{x.rsquared_adj:.4f}"
        }
    )
    print(summary)

    # 5. 反解参数
    # 系数 X2 对应 -kappa
    kappa = -results.params["X2 (-κ·dt)"]
    # 系数 X1 对应 kappa * theta
    kappa_theta = results.params["X1 (κθ·dt)"]
    
    theta = kappa_theta / kappa if kappa != 0 else np.nan
    
    # 额外：估算波动率系数 sigma (残差的标准差)
    # sigma = np.sqrt(results.mse_resid / dt)
    
    print("-" * 30)
    print(f"📈 估算结果:")
    print(f"均值回复速度 (kappa): {kappa:.4f}")
    print(f"长期均值水平 (theta): {theta:.4f}")
    
    return kappa, theta