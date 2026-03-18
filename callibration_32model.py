import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import time

# 锁定物理参数
FIXED_KAPPA = 127.0590
FIXED_THETA = 0.163960

# ==========================================
# 1. 稳定的向量化蒙特卡洛定价器 (专门克制末日期权)
# ==========================================
def bs_price_normalized(M, T, r, iv, opt_type):
    d1 = (np.log(1.0 / M) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    if opt_type == 'C':
        return norm.cdf(d1) - M * np.exp(-r * T) * norm.cdf(d2)
    else:
        return M * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1)

def three_halves_mc_pricer_normalized(M_array, type_array, T, r, V0, kappa, theta, sigma, rho, Z_v_mat, Z_x_mat):
    num_paths, num_steps = Z_v_mat.shape
    dt = T / num_steps
    
    S = np.ones(num_paths) 
    V = np.full(num_paths, V0)
    
    for i in range(num_steps):
        Z_v = Z_v_mat[:, i]
        Z_x = Z_x_mat[:, i]
        Z_s = rho * Z_v + np.sqrt(1.0 - rho**2) * Z_x
        
        V_pos = np.maximum(V, 1e-8) 
        
        S = S * np.exp((r - 0.5 * V_pos) * dt + np.sqrt(V_pos * dt) * Z_s)
        V = V + kappa * V_pos * (theta - V_pos) * dt + sigma * (V_pos**1.5) * np.sqrt(dt) * Z_v
        
    discount_factor = np.exp(-r * T)
    model_prices = np.zeros(len(M_array))
    
    for idx, (M, opt_type) in enumerate(zip(M_array, type_array)):
        if opt_type == 'C':
            payoff = np.maximum(S - M, 0.0)
        else:
            payoff = np.maximum(M - S, 0.0)
        model_prices[idx] = np.mean(payoff) * discount_factor
        
    return model_prices

# ==========================================
# 2. 目标函数 (相对价格误差)
# ==========================================
def objective_2_params_mc(params, fixed_V0, M_array, type_array, T, r, market_prices, Z_v_mat, Z_x_mat):
    vov, rho = params
    
    # 物理边界保护
    if vov <= 0.01 or vov > 20.0 or rho < -1.0 or rho > 1.0:
        return 1e10
    
    # 调用稳定的 MC 定价器
    m_prices = three_halves_mc_pricer_normalized(
        M_array, type_array, T, r, fixed_V0, FIXED_KAPPA, FIXED_THETA, vov, rho, Z_v_mat, Z_x_mat
    )
    
    # 相对价格误差
    residuals = (m_prices - market_prices) / (market_prices + 1e-5)
    mse = np.sum(residuals**2)
    
    print(f"试探: VoV={vov:.4f}, Rho={rho:.4f} => MSE: {mse:.4f}")
    return mse

# ==========================================
# 3. 执行校准
# ==========================================
def run_stage_calibration_mc(stage_name, csv_filepath):
    print(f"\n[{stage_name}] 开始极速 MC 校准...")
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
    
    # 2. 获取市场归一化基准价格
    market_prices = np.array([bs_price_normalized(M, T, r, iv, t) for M, iv, t in zip(M_array, iv_array, type_array)])
    
    # 3. 预先生成随机矩阵 (锁定随机种子保证平滑梯度)
    # 因为只有 2 个参数，我们可以放大 paths 保证精度，而不会太慢
    np.random.seed(123456)
    num_paths = 30000 
    num_steps = 500    # 末日期权 T 极小，20 步足够了
    Z_v_mat = np.random.standard_normal((num_paths, num_steps))
    Z_x_mat = np.random.standard_normal((num_paths, num_steps))
    
    # 4. 执行 2 维寻优
    initial_guess = [1.5, -0.5]
    
    start = time.time()
    res = minimize(
        objective_2_params_mc, 
        initial_guess, 
        args=(fixed_V0, M_array, type_array, T, r, market_prices, Z_v_mat, Z_x_mat), 
        method='Nelder-Mead', # 核心修改：换成无梯度单纯形法
        options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000} # 调整单纯形的收敛精度
    )
    
    if res.success:
        vov, rho = res.x
        print(f"✅ 校准成功！耗时: {time.time() - start:.3f} 秒")
        print(f"🔒 锚定 -> V0: {fixed_V0:.4f} (IV:{atm_iv*100:.1f}%), Kappa: {FIXED_KAPPA:.2f}, Theta: {FIXED_THETA:.4f}")
        print(f"🎯 拟合 -> VoV: {vov:.4f}, Rho: {rho:.4f}")
    else:
        print("❌ 优化未完全收敛，但最后参数为:")
        print(f"🎯 拟合 -> VoV: {res.x[0]:.4f}, Rho: {res.x[1]:.4f}")