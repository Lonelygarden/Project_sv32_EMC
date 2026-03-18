import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, least_squares
from scipy.stats import norm
import time
import pyfeng as pf  # 只导入一次，避免重复开销

# ==========================================
# 1. Black-Scholes 归一化定价器
# ==========================================
def bs_price_normalized(moneyness, T, r, iv, opt_type):
    """计算 S0=1 时的标准 BS 绝对价格"""
    if iv * np.sqrt(T) == 0:  # 避免除零错误
        return 0.0 if (opt_type == 'C' and moneyness >= 1) else max(moneyness * np.exp(-r*T) - 1, 0)
    
    d1 = (np.log(1.0 / moneyness) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    
    if opt_type == 'C':
        return norm.cdf(d1) - moneyness * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put
        return moneyness * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1)

# ==========================================
# 2. 傅里叶 COS 定价器 (优化版)
# ==========================================
def fourier_pricer_wrapper_moneyness(M_array, type_array, texp, r, V0, kappa, theta, vov, rho):
    """
    优化版 Fourier COS 定价器
    - 移除重复import
    - 简化异常处理
    - 提前参数校验
    - 减少冗余计算
    """
    # 提前参数校验，避免无效计算
    if V0 <= 0 or kappa <= 0 or theta <= 0 or vov <= 0 or abs(rho) >= 1:
        return np.full_like(M_array, 1e6)
    
    # 强制类型转换（只做一次）
    M_arr = np.array(M_array, dtype=np.float64)
    t_arr = np.array(type_array)
    
    try:
        # 初始化定价模型（只初始化一次）
        m = pf.sv_fft.Sv32FourierCos(
            sigma=np.sqrt(V0), vov=vov, rho=rho, mr=kappa, theta=theta, intr=r
        )
        
        # 向量化定价（核心逻辑）
        call_prices = m.price(strike=M_arr, spot=1.0, texp=texp)
        
        # 统一为数组格式
        if np.isscalar(call_prices):
            call_prices = np.array([call_prices])
        
        # Put-Call 平价转换
        prices = np.where(
            t_arr == 'C',
            call_prices,
            call_prices - 1.0 + M_arr * np.exp(-r * texp)
        )
        
        # NaN 处理
        if np.any(np.isnan(prices)):
            return np.full_like(M_arr, 1e6)
        
        return prices
        
    except Exception as e:
        # 简化错误输出，只在严重错误时打印
        if np.random.random() < 0.01:  # 1%概率打印，减少I/O
            print(f"定价错误（采样输出）: {e} | 参数: V0={V0:.3f}, rho={rho:.3f}")
        return np.full_like(M_arr, 1e6)



def objective_function_2_params(params_2, fixed_kappa, fixed_theta, fixed_V0,
                               M_array, type_array, T, r, market_prices):
    """2参数目标函数（优化版）"""
    vov, rho = params_2 
    
    # 计算模型价格
    model_prices = fourier_pricer_wrapper_moneyness(
        M_array, type_array, T, r, fixed_V0, fixed_kappa, fixed_theta, vov, rho
    )
    
    # 误差计算
    error = (model_prices - market_prices) / (market_prices + 1e-6)
    mse = np.sum(error**2)
    print(f"当前参数: vov={vov:.4f}, rho={rho:.4f}, MSE={mse:.6f}")
    
    return mse

# ==========================================
# 5. 降维校准函数（优化版）
# ==========================================
def calibrate_two_params(csv_filepath, fixed_kappa, fixed_theta, r=0.0):
    """
    3参数降维校准
    - 优化L-BFGS-B参数，提升收敛速度
    - 增加提前终止
    """
    print(f"\n[{csv_filepath}] 开始 2 参数降维校准 (锁定 kappa={fixed_kappa:.4f}, theta={fixed_theta:.4f})...")
    
    # 数据加载
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            print("空数据文件")
            return None
    except FileNotFoundError:
        print("文件未找到")
        return None
    
    # 数据预处理
    M_array = df['Moneyness'].values
    type_array = df['Type'].values
    iv_array = df['iv'].values / 100.0 if df['iv'].mean() > 5 else df['iv'].values
    T = df['T'].iloc[0]
    
    # 向量化计算市场价格
    market_prices = np.vectorize(bs_price_normalized)(
        M_array, T, r, iv_array, type_array
    )
    
    # 初始猜测
    atm_iv = df.iloc[(df['Moneyness'] - 1.0).abs().argsort()[:1]]['iv'].values[0]
    atm_iv = atm_iv / 100.0 if atm_iv > 5 else atm_iv
    initial_guess_2 = [1.0, -0.5]
    fixed_V0 = atm_iv**2
    
    # 更紧凑的边界
    bounds_2 = [(0.1, 5.0), (-0.999, 0.999)]
    
    start_time = time.time()
    
    # 优化L-BFGS-B参数
    result = minimize(
        objective_function_2_params, 
        initial_guess_2, 
        args=(fixed_kappa, fixed_theta, fixed_V0, M_array, type_array, T, r, market_prices), 
        method='L-BFGS-B', 
        bounds=bounds_2,
        options={
            'maxiter': 30,    # 减少最大迭代次数
            'ftol': 1e-4,     # 放宽容差
            'eps': 1e-3,      
            'disp': False     # 关闭详细输出
        }
    )
    
    # 结果输出
    if result.success:
        V0_opt, vov_opt, rho_opt = result.x
        print(f"✅ 降维校准成功！耗时: {time.time() - start_time:.2f} 秒")
        print(f"拟合的 V0:  {V0_opt:.4f}")
        print(f"拟合的 vov: {vov_opt:.4f}")
        print(f"拟合的 rho: {rho_opt:.4f}")
        return result.x
    else:
        print(f"❌ 降维校准失败: {result.message}")
        return None