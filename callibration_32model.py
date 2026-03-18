import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import time
import pyfeng as pf
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares

# ==========================================
# 1. Black-Scholes 归一化定价器
# ==========================================
def bs_price_normalized(moneyness, T, r, iv, opt_type):
    """计算 S0=1 时的标准 BS 绝对价格"""
    d1 = (np.log(1.0 / moneyness) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    if opt_type == 'C':
        return norm.cdf(d1) - moneyness * np.exp(-r * T) * norm.cdf(d2)
    else: # Put
        return moneyness * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1)

# ==========================================
# 2. 傅里叶 COS 定价器 (Moneyness 封装版)
# ==========================================
def fourier_pricer_wrapper_moneyness(M_array, type_array, texp, r, V0, kappa, theta, vov, rho):
    """
    使用 Moneyness 的 Fourier COS 定价器
    将现价强行设为 S=1.0，行权价 K=M
    """
    import pyfeng as pf
    
    # 强制将输入转换为纯净的 float64 numpy 数组，防止 pandas 格式干扰底层 C++ 代码
    M_arr = np.array(M_array, dtype=np.float64)
    t_arr = np.array(type_array)
    
    try:
        # PyFeng 中 sigma 通常代表初始状态，我们暂且按原样传入 V0
        m = pf.sv_fft.Sv32FourierCos(
            sigma=V0,  
            vov=vov, 
            rho=rho,
            mr=kappa, 
            theta=theta, 
            intr=r
        )
        
        # 尝试进行向量化定价
        call_prices = m.price(strike=M_arr, spot=1.0, texp=texp)
        
        # 处理返回值为标量的情况 (万一数组只传进了一个数字)
        if isinstance(call_prices, (float, int)):
            call_prices = np.array([call_prices])
            
        # Put-Call Parity 换算
        prices = np.where(
            t_arr == 'C',
            call_prices,
            call_prices - 1.0 + M_arr * np.exp(-r * texp)
        )
        
        # 检查有没有算出非数字 (NaN)
        if np.any(np.isnan(prices)):
            print(f"⚠️ 警告: PyFeng 算出了 NaN! 参数组合: V0={V0:.3f}, vov={vov:.3f}, rho={rho:.3f}")
            return np.full_like(M_arr, 1e6)
            
        return prices
        
    except Exception as e:
        # 【极其重要】把被掩盖的真实报错打印出来！
        print(f"\n❌ [DEBUG 拦截] PyFeng 底层报错: {e}")
        print(f"出问题的参数组合: V0={V0}, kappa={kappa}, theta={theta}, vov={vov}, rho={rho}")
        
        # 如果向量化失败，我们退化回最安全的 for 循环来计算 (安全网)
        print("尝试退化为 for 循环逐个计算...")
        prices = np.zeros_like(M_arr)
        try:
            for i, (M_val, type_val) in enumerate(zip(M_arr, t_arr)):
                cp = m.price(strike=M_val, spot=1.0, texp=texp)
                if type_val == 'C':
                    prices[i] = cp
                else:
                    prices[i] = cp - 1.0 + M_val * np.exp(-r * texp)
            print("for 循环计算成功！")
            return prices
        except Exception as e2:
            print(f"❌ for 循环也彻底失败: {e2}")
            return np.full_like(M_arr, 1e6)

# ==========================================
# 3. 目标函数 (5参数 & 3参数)
# ==========================================
def objective_function_5_params(params, M_array, type_array, T, r, market_prices):
    V0, kappa, theta, vov, rho = params
    
    if V0 <= 0 or kappa <= 0 or theta <= 0 or vov <= 0 or rho < -0.999 or rho > 0.999:
        return 1e10
        
    model_prices = fourier_pricer_wrapper_moneyness(
        M_array, type_array, T, r, V0, kappa, theta, vov, rho
    )
    
    error = (model_prices - market_prices) / (market_prices + 1e-5)
    mse = np.sum(error**2)
    
    print(f"[监控] 尝试参数: V0={V0:.4f}, kappa={kappa:.4f}, theta={theta:.4f}, vov={vov:.4f}, rho={rho:.4f} => MSE误差: {mse:.6f}")
    
    # 如果算出 NaN，强制返回惩罚值，逼迫优化器回头
    if np.isnan(mse) or np.isinf(mse):
        return 1e10
        
    return mse

def objective_function_3_params(params_3, fixed_kappa, fixed_theta, M_array, type_array, T, r, market_prices):
    V0, vov, rho = params_3 
    
    # 边界惩罚
    if V0 <= 0 or vov <= 0 or rho < -0.999 or rho > 0.999:
        return 1e10
        
    model_prices = fourier_pricer_wrapper_moneyness(
        M_array, type_array, T, r, V0, fixed_kappa, fixed_theta, vov, rho
    )
    error = (model_prices - market_prices) / (market_prices + 1e-5)
    mse = np.sum(error**2)
    print(f"[监控] 尝试参数: V0={V0:.4f}, vov={vov:.4f}, rho={rho:.4f} => MSE误差: {mse:.6f}")
    
    return np.sum(mse)

# ==========================================
# 4. 校准执行函数：全参数 (用于提取远端 kappa, theta)
# ==========================================
def run_calibration(csv_filepath, r=0.0):
    print(f"\n[{csv_filepath}] 开始 5 参数模型校准 (寻找远端基准)...")
    
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty: return
    except FileNotFoundError:
        return
        
    M_array = df['Moneyness'].values
    type_array = df['Type'].values
    iv_array = df['iv'].values / 100.0 if df['iv'].mean() > 5 else df['iv'].values
    T = df['T'].iloc[0] 
    
    market_prices = np.array([
        bs_price_normalized(M, T, r, iv, opt_type) 
        for M, iv, opt_type in zip(M_array, iv_array, type_array)
    ])
    
    # 初始猜测
    atm_iv = df.iloc[(df['Moneyness'] - 1.0).abs().argsort()[:1]]['iv'].values[0]
    atm_iv = atm_iv / 100.0 if atm_iv > 5 else atm_iv
    v0_guess = atm_iv**2
    
    initial_guess = [v0_guess, 2.0, v0_guess, 1.0, -0.5] 
    initial_guess = [0.8, 5.0, 0.5, 2.0, -0.8]
    initial_guess = [0.1, 1.0, 0.1, 0.5, -0.3]
    bounds = ((1e-4, 5.0), (0.1, 20.0), (1e-4, 5.0), (0.1, 20.0), (-0.999, 0.999))
    
    start_time = time.time()
    start_time = time.time()
    bounds_de = [(1e-4, 2.0), (0.1, 10.0), (1e-4, 2.0), (0.1, 5.0), (-0.999, 0.999)]

    # 效率极高：因为它不需要计算梯度，对 COS 方法这种有网格感的函数极度友好
    result = differential_evolution(
        objective_function_5_params, 
        bounds=bounds_de, 
        args=(M_array, type_array, T, r, market_prices),
        strategy='best1bin', 
        popsize=15, 
        tol=0.01, 
        mutation=(0.5, 1), 
        recombination=0.7
    )
    
    if result.success:
        V0, kappa, theta, vov, rho = result.x
        print(f"✅ 5参数校准成功！耗时: {time.time() - start_time:.2f} 秒")
        print(f"拟合结果: V0={V0:.4f}, kappa={kappa:.4f}, theta={theta:.4f}, vov={vov:.4f}, rho={rho:.4f}")
        return result.x
    else:
        print("❌ 校准失败:", result.message)
        return result.x

# ==========================================
# 5. 校准执行函数：降维 3参数 (用于异动短端期权)
# ==========================================
def calibrate_crash_phase(csv_filepath, fixed_kappa, fixed_theta, r=0.0):
    print(f"\n[{csv_filepath}] 开始 3 参数降维校准 (锁定 kappa={fixed_kappa}, theta={fixed_theta})...")
    
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty: return
    except FileNotFoundError:
        return
        
    M_array = df['Moneyness'].values
    type_array = df['Type'].values
    iv_array = df['iv'].values / 100.0 if df['iv'].mean() > 5 else df['iv'].values
    T = df['T'].iloc[0]
    
    market_prices = np.array([
        bs_price_normalized(M, T, r, iv, opt_type) 
        for M, iv, opt_type in zip(M_array, iv_array, type_array)
    ])
    
    atm_iv = df.iloc[(df['Moneyness'] - 1.0).abs().argsort()[:1]]['iv'].values[0]
    atm_iv = atm_iv / 100.0 if atm_iv > 5 else atm_iv
    v0_guess = atm_iv**2
    
    initial_guess_3 = [v0_guess, 1.0, -0.5] 
    bounds_3 = ((1e-4, 5.0), (0.1, 20.0), (-0.999, 0.999))
    
    start_time = time.time()
    result = minimize(
        objective_function_3_params, 
        initial_guess_3, 
        args=(fixed_kappa, fixed_theta, M_array, type_array, T, r, market_prices), 
        method='L-BFGS-B', 
        bounds=bounds_3,
        options={'maxiter': 50, 'ftol': 1e-5, 'eps': 1e-3, 'disp': True}
    )
    
    if result.success:
        V0_opt, vov_opt, rho_opt = result.x
        print(f"✅ 降维校准成功！耗时: {time.time() - start_time:.2f} 秒")
        print(f"拟合的 V0:  {V0_opt:.4f}")
        print(f"拟合的 vov: {vov_opt:.4f}")
        print(f"拟合的 rho: {rho_opt:.4f}")
        return result.x
    else:
        print("❌ 校准失败:", result.message)
        return result.x