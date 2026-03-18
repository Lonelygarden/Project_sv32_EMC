import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import pyfeng as pf
import time

# ==========================================
# 🎯 填入你刚刚跑出的真实世界物理参数
# ==========================================
FIXED_KAPPA = 29.6002
FIXED_THETA = 0.166832

# ==========================================
# 1. 基础定价器与 Vega 加权残差函数
# ==========================================
def bs_price_normalized(M, T, r, iv, opt_type):
    d1 = (np.log(1.0 / M) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    if opt_type == 'C':
        return norm.cdf(d1) - M * np.exp(-r * T) * norm.cdf(d2)
    else:
        return M * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1)

def objective_2_params(params, fixed_V0, M_array, type_array, T, r, market_prices, market_vegas):
    vov, rho = params
    
    # 物理边界保护
    if vov <= 0.01 or vov > 10.0 or rho < -0.999 or rho > 0.999:
        return 1e10
    
    try:
        # 实例化 3/2 模型
        m = pf.sv_fft.Sv32FourierCos(
            sigma=np.sqrt(fixed_V0), vov=vov, rho=rho, 
            mr=FIXED_KAPPA, theta=FIXED_THETA, intr=r
        )
        
        # 向量化计算归一化价格
        m_prices = m.price(strike=M_array, spot=1.0, texp=T)
        m_prices = np.where(type_array == 'C', m_prices, m_prices - 1.0 + M_array * np.exp(-r*T))
        
        # Vega 加权残差 (速度极快，精度媲美 IV 拟合)
        residuals = (m_prices - market_prices) / (market_vegas + 1e-5)
        mse = np.sum(residuals**2)
        print(f"测试参数 -> VoV: {vov:.4f}, Rho: {rho:.4f}, MSE: {mse:.6f}")
        return mse
    except:
        return 1e10

# ==========================================
# 2. 阶段执行主函数
# ==========================================
def run_stage_calibration(stage_name, csv_filepath):
    print(f"\n[{stage_name}] 开始极速校准...")
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {csv_filepath}")
        return
        
    M_array = df['Moneyness'].values
    type_array = df['Type'].values
    iv_array = df['iv'].values / 100.0 if df['iv'].mean() > 5 else df['iv'].values
    T, r = df['T'].iloc[0], 0.0
    
    # 1. 提取 ATM IV 锁定瞬时方差 V0
    atm_idx = np.abs(M_array - 1.0).argmin()
    atm_iv = iv_array[atm_idx]
    fixed_V0 = atm_iv**2
    
    # 2. 计算市场目标 (价格与 Vega)
    d1 = (np.log(1.0/M_array) + (r + 0.5*iv_array**2)*T) / (iv_array*np.sqrt(T))
    market_vegas = norm.pdf(d1) * np.sqrt(T)
    market_prices = np.array([bs_price_normalized(M, T, r, iv, t) for M, iv, t in zip(M_array, iv_array, type_array)])
    
    # 3. 执行 2 维极速优化
    initial_guess = [1.5, -0.5] # [vov, rho]
    bounds = [(0.1, 5.0), (-0.99, 0.99)]
    
    start = time.time()
    res = minimize(
        objective_2_params, 
        initial_guess, 
        args=(fixed_V0, M_array, type_array, T, r, market_prices, market_vegas),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    # 4. 打印报告
    if res.success:
        vov, rho = res.x
        print(f"✅ 校准成功！耗时: {time.time() - start:.3f} 秒")
        print(f"🔒 锚定参数 -> V0: {fixed_V0:.4f} (ATM IV: {atm_iv*100:.1f}%), Kappa: {FIXED_KAPPA:.2f}, Theta: {FIXED_THETA:.4f}")
        print(f"🎯 拟合结果 -> VoV: {vov:.4f}, Rho: {rho:.4f}")
    else:
        print("❌ 优化失败")