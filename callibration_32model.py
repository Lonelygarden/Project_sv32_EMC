import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

def medvedev_scaillet_jump_iv(X_array, atm_iv, vov, rho, jump_vol, T):
    """
    完全版 Medvedev & Scaillet (2007) 公式 (含跳跃曲率项)
    jump_vol = \lambda * E[|\Delta J|] (年化跳跃绝对强度)
    """
    # 1. 扩散倾斜项 (Diffusion Skew)
    coef_X = (vov * rho / 4.0) * atm_iv
    
    # 2. 扩散曲率项 (Diffusion Curvature)
    diff_curv = (vov**2 * atm_iv / 48.0) * (2.0 - rho**2)
    
    # 3. 💥 跳跃曲率项 (Jump Curvature - 论文 Eq 24 精华)
    # 当 T 很小时，这一项主导了整个微笑曲线的陡峭程度
    jump_curv = (np.sqrt(2 * np.pi) / 4.0) * (jump_vol / (atm_iv**2 * np.sqrt(T)))
    
    # 总曲率
    coef_X2 = diff_curv + jump_curv
    
    return atm_iv - coef_X * X_array + coef_X2 * (X_array**2)

def objective_ms_jump(params, X_array, market_ivs, atm_iv, T):
    vov, rho, jump_vol = params
    
    # 物理边界保护
    # jump_vol 必须 >= 0
    if vov <= 0.01 or vov > 15.0 or rho < -0.999 or rho > 0.999 or jump_vol < 0.0:
        return 1e10
        
    model_ivs = medvedev_scaillet_jump_iv(X_array, atm_iv, vov, rho, jump_vol, T)
    
    # IV 空间 MSE
    return np.mean((model_ivs - market_ivs)**2)

def run_ms_jump_calibration(stage_name, csv_filepath):
    print(f"\n[{stage_name}] 启动 Jump-Diffusion 渐近校准...")
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_filepath}")
        return

    r = 0.0
    T = df['T'].iloc[0]
    df['X'] = -np.log(df['Moneyness']) + r * T
    
    atm_idx = np.abs(df['X']).argmin()
    atm_iv = df['iv'].iloc[atm_idx] / 100.0 if df['iv'].mean() > 5 else df['iv'].iloc[atm_idx]
    
    # 扩大一点 X 的范围，让跳跃产生的尾部特征更明显
    df_filtered = df[(df['X'] >= -0.08) & (df['X'] <= 0.08)].copy()
    
    X_array = df_filtered['X'].values
    market_ivs = df_filtered['iv'].values / 100.0 if df_filtered['iv'].mean() > 5 else df_filtered['iv'].values
    
    # 拟合 3 个核心自由度: [vov, rho, jump_vol]
    initial_guess = [2.0, -0.5, 0.5]
    bounds = [(0.1, 20.0), (-0.95, 0.95), (0.001, 20.0)]
    
    start = time.time()
    res = minimize(
        objective_ms_jump, 
        initial_guess, 
        args=(X_array, market_ivs, atm_iv, T), 
        method='L-BFGS-B',
        bounds=bounds
    )
    
    if res.success:
        vov, rho, jump_vol = res.x
        print(f"✅ 校准成功！耗时: {time.time() - start:.5f} 秒")
        print(f"🔒 锚定 -> ATM IV: {atm_iv*100:.2f}%")
        print(f"🎯 扩散项 -> VoV: {vov:.4f}, Rho: {rho:.4f}")
        print(f"🎯 跳跃项 -> Jump Vol: {jump_vol:.4f}")
    else:
        print("❌ 优化失败")

def medvedev_scaillet_32_iv(X_array, atm_iv, vov, rho):
    """
    基于 Medvedev & Scaillet (2007) 论文的纯 3/2 模型隐含波动率二次渐近展开。
    极大提升短端期权校准速度。
    """
    # 计算一阶项和二阶项系数
    coef_X = (vov * rho / 4.0) * atm_iv
    coef_X2 = (vov**2 * atm_iv / 48.0) * (2.0 - rho**2)
    
    # 直接输出模型拟合的 IV 微笑曲线
    model_ivs = atm_iv - coef_X * X_array + coef_X2 * (X_array**2)
    return model_ivs

def objective_ms_paper(params, X_array, market_ivs, atm_iv):
    vov, rho = params
    
    # 物理边界保护
    if vov <= 0.01 or vov > 15.0 or rho < -0.999 or rho > 0.999:
        return 1e10
        
    model_ivs = medvedev_scaillet_32_iv(X_array, atm_iv, vov, rho)
    
    # 💥 直接在 IV 空间计算均方误差 (MSE)，告别价格空间的数值噪音！
    mse = np.mean((model_ivs - market_ivs)**2)
    return mse

def run_ms_paper_calibration(stage_name, csv_filepath):
    print(f"\n[{stage_name}] 启动 Medvedev & Scaillet (2007) 渐近校准...")
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_filepath}")
        return

    # 1. 转换 Moneyness 为论文中的对数货币度 X
    # 论文中 X = ln(S/K) + r*tau。因为我们的 M = K/S，所以 X = -ln(M) + r*tau
    r = 0.0
    T = df['T'].iloc[0]
    df['X'] = -np.log(df['Moneyness']) + r * T
    
    # 2. 提取 ATM IV (X 最接近 0 的点)
    atm_idx = np.abs(df['X']).argmin()
    atm_iv = df['iv'].iloc[atm_idx] / 100.0 if df['iv'].mean() > 5 else df['iv'].iloc[atm_idx]
    
    # 3. 💥 论文硬性约束：只取近平值区域 |X| <= 0.05 保证渐近展开的精度
    df_filtered = df[(df['X'] >= -0.05) & (df['X'] <= 0.05)].copy()
    
    if len(df_filtered) < 3:
        print("⚠️ 过滤后数据太少，放宽 X 限制到 0.1")
        df_filtered = df[(df['X'] >= -0.1) & (df['X'] <= 0.1)].copy()

    X_array = df_filtered['X'].values
    market_ivs = df_filtered['iv'].values / 100.0 if df_filtered['iv'].mean() > 5 else df_filtered['iv'].values
    
    # 4. 执行极速拟合
    initial_guess = [1.5, -0.5]
    bounds = [(0.1, 20.0), (-0.99, 0.99)]
    
    start = time.time()
    # 因为函数极度平滑，L-BFGS-B 现在可以完美运行，瞬间收敛
    res = minimize(
        objective_ms_paper, 
        initial_guess, 
        args=(X_array, market_ivs, atm_iv), 
        method='L-BFGS-B',
        bounds=bounds
    )
    
    if res.success:
        vov, rho = res.x
        print(f"✅ 校准成功！耗时: {time.time() - start:.5f} 秒")
        print(f"🔒 锚定 -> ATM IV (\hat{{\sigma}}): {atm_iv*100:.2f}%")
        print(f"🎯 拟合 -> VoV: {vov:.4f}, Rho: {rho:.4f}")
    else:
        print("❌ 优化失败")