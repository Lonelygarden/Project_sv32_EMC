import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import time

# ==========================================
# 1. Black-Scholes 归一化定价器 (将 IV 转为相对价格)
# ==========================================
def bs_price_normalized(M, T, r, iv, opt_type):
    """
    计算 S0=1 时的标准 Black-Scholes 期权价格
    M: Moneyness (相当于归一化后的 K)
    """
    d1 = (np.log(1.0 / M) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    
    if opt_type == 'C':
        return norm.cdf(d1) - M * np.exp(-r * T) * norm.cdf(d2)
    else: # Put
        return M * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1)

# ==========================================
# 2. 3/2 模型蒙特卡洛定价器 (归一化版本)
# ==========================================
def three_halves_mc_pricer_normalized(M_array, type_array, T, r, V0, kappa, theta, sigma, rho, Z_v_mat, Z_x_mat):
    """
    S0 = 1.0 的 3/2 模型向量化蒙特卡洛定价
    使用固定的随机矩阵以保证优化器计算梯度的平滑性
    """
    num_paths, num_steps = Z_v_mat.shape
    dt = T / num_steps
    
    S = np.ones(num_paths) # S0 = 1.0
    V = np.full(num_paths, V0)
    
    for i in range(num_steps):
        Z_v = Z_v_mat[:, i]
        Z_x = Z_x_mat[:, i]
        # 构建相关的布朗运动
        Z_s = rho * Z_v + np.sqrt(1.0 - rho**2) * Z_x
        
        # 反射壁处理，防止方差变为负数
        V_pos = np.maximum(V, 1e-8) 
        
        # Euler-Maruyama 离散化演化
        S = S * np.exp((r - 0.5 * V_pos) * dt + np.sqrt(V_pos * dt) * Z_s)
        # 3/2 模型的方差过程: dV = kappa*V*(theta-V)dt + sigma*V^(3/2)dW
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
# 3. 目标函数 (损失函数)
# ==========================================
def objective_function(params, M_array, type_array, T, r, market_prices, Z_v_mat, Z_x_mat):
    V0, kappa, theta, sigma, rho = params
    
    # 严格的边界惩罚，防止参数跑飞
    if V0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or rho < -0.999 or rho > 0.999:
        return 1e10
        
    model_prices = three_halves_mc_pricer_normalized(
        M_array, type_array, T, r, V0, kappa, theta, sigma, rho, Z_v_mat, Z_x_mat
    )
    
    # 异动期间期权价格差异极大，使用加权相对误差 (Weighted Relative MSE)
    # 给远离平值的深度虚值期权更高的容忍度，或者统一化误差尺度
    error = (model_prices - market_prices) / (market_prices + 1e-5)
    mse = np.sum(error**2)
    
    return mse

# ==========================================
# 4. 执行校准的主流程
# ==========================================
def run_calibration(csv_filepath, r=0.0):
    print(f"\n[{csv_filepath}] 开始加载数据并进行 3/2 模型校准...")
    
    # 读取数据
    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            print("数据为空，跳过。")
            return
    except FileNotFoundError:
        print(f"文件 {csv_filepath} 不存在，跳过。")
        return
        
    # 提取需要的数组
    M_array = df['Moneyness'].values
    type_array = df['Type'].values
    iv_array = df['iv'].values / 100.0 if df['iv'].mean() > 5 else df['iv'].values # 确保 IV 是小数形式 (例如 0.5 而不是 50)
    T = df['T'].iloc[0] # 该截面的到期时间都是一样的
    
    # 1. 把市场 IV 转为归一化(S0=1)的市场绝对价格
    market_prices = np.array([
        bs_price_normalized(M, T, r, iv, opt_type) 
        for M, iv, opt_type in zip(M_array, iv_array, type_array)
    ])
    
    # 2. 预先生成随机矩阵 (锁定随机种子，保证优化器平滑)
    # 异动期短端期权 T 极小，增加 paths 减少方差，降低 steps 提升速度
    num_paths = 20000 
    num_steps = 30    
    np.random.seed(42)
    Z_v_mat = np.random.standard_normal((num_paths, num_steps))
    Z_x_mat = np.random.standard_normal((num_paths, num_steps))
    
    # 3. 设定初始参数与边界
    # 初始 V0 根据平值附近(M~1.0)的 IV 猜测
    atm_iv = df.iloc[(df['Moneyness'] - 1.0).abs().argsort()[:1]]['iv'].values[0]
    atm_iv = atm_iv / 100.0 if atm_iv > 5 else atm_iv
    v0_guess = atm_iv**2
    
    # params = [V0, kappa, theta, sigma, rho]
    initial_guess = [v0_guess, 2.0, v0_guess, 1.0, -0.5] 
    
    # 边界设置: 异动时 sigma 可能极大，rho 可能极度贴近 -1
    bounds = (
        (1e-4, 5.0),    # V0
        (0.1,  20.0),   # kappa (均值回归速度)
        (1e-4, 5.0),    # theta (长期方差)
        (0.1,  20.0),   # sigma (Vol-of-vol，3/2模型允许它很大)
        (-0.999, 0.999) # rho
    )
    
    print(f"期权数量: {len(M_array)}, 到期时间 T: {T:.5f}, 初始猜测 V0: {v0_guess:.4f}")
    print("优化器启动中，请耐心等待 (约需 1~3 分钟)...")
    
    start_time = time.time()
    
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(M_array, type_array, T, r, market_prices, Z_v_mat, Z_x_mat), 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 50, 'ftol': 1e-4, 'disp': False} # disp=True 可以看每一步迭代
    )
    
    end_time = time.time()
    
    if result.success:
        V0, kappa, theta, sigma, rho = result.x
        print(f"✅ 校准成功！耗时: {end_time - start_time:.1f} 秒")
        print("-" * 30)
        print(f"V0    (瞬时方差) = {V0:.4f}  (相当于瞬时 IV = {np.sqrt(V0)*100:.2f}%)")
        print(f"kappa (回归速度) = {kappa:.4f}")
        print(f"theta (长期方差) = {theta:.4f}")
        print(f"sigma (Vol-of-Vol) = {sigma:.4f}")
        print(f"rho   (相关系数) = {rho:.4f}")
        print("-" * 30)
        
        # 将拟合的价格写回 dataframe 供对比
        fitted_prices = three_halves_mc_pricer_normalized(
            M_array, type_array, T, r, V0, kappa, theta, sigma, rho, Z_v_mat, Z_x_mat
        )
        df['Market_Norm_Price'] = np.round(market_prices, 6)
        df['Model_Norm_Price']  = np.round(fitted_prices, 6)
        df['Error_%'] = np.round((fitted_prices - market_prices) / market_prices * 100, 2)
        
        print(df[['Moneyness', 'Type', 'iv', 'Market_Norm_Price', 'Model_Norm_Price', 'Error_%']].to_string(index=False))
        return result.x
    else:
        print("❌ 校准失败或未完全收敛，原因:", result.message)
        # 即使未成功，返回最后的参数看看是否由于达到迭代次数上限
        return result.x
    
    
# ==========================================
# 降维版目标函数：只拟合 V0, sigma, rho
# (kappa 和 theta 作为已知常数传入)
# ==========================================
def objective_function_3_params(params_3, fixed_kappa, fixed_theta, M_array, type_array, T, r, market_prices, Z_v_mat, Z_x_mat):
    # 优化器现在只吐出 3 个正在寻找的参数
    V0, sigma, rho = params_3 
    
    # 锁定 kappa 和 theta
    kappa = fixed_kappa
    theta = fixed_theta
    
    # 边界惩罚 (仅针对这3个参数)
    if V0 <= 0 or sigma <= 0 or rho < -0.999 or rho > 0.999:
        return 1e10
        
    # 调用依然是那个完整的 5 参数定价器
    model_prices = three_halves_mc_pricer_normalized(
        M_array, type_array, T, r, V0, kappa, theta, sigma, rho, Z_v_mat, Z_x_mat
    )
    
    # 计算加权相对误差
    error = (model_prices - market_prices) / (market_prices + 1e-5)
    mse = np.sum(error**2)
    
    return mse


# ==========================================
# Crash 阶段的专用 Calibration 调用示例
# ==========================================
def calibrate_crash_phase(csv_filepath, fixed_kappa, fixed_theta, r=0.0):
    # ... (前面的读取数据和生成随机数矩阵 Z_v_mat, Z_x_mat 的代码保持不变) ...
    
    # 初始猜测现在只剩 3 个：[V0, sigma, rho]
    v0_guess = 0.5 # 可以用 ATM IV 的平方
    initial_guess_3 = [v0_guess, 1.0, -0.5] 
    
    # 边界也只剩 3 个
    bounds_3 = (
        (1e-4, 5.0),    # V0
        (0.1,  20.0),   # sigma (允许它极大)
        (-0.999, 0.999) # rho
    )
    
    # 使用 L-BFGS-B 进行 3 维寻优
    result = minimize(
        objective_function_3_params, 
        initial_guess_3, 
        args=(fixed_kappa, fixed_theta, M_array, type_array, T, r, market_prices, Z_v_mat, Z_x_mat), 
        method='L-BFGS-B', 
        bounds=bounds_3,
        options={'maxiter': 50, 'ftol': 1e-5}
    )
    
    if result.success:
        V0_opt, sigma_opt, rho_opt = result.x
        print(f"✅ Crash阶段降维校准成功！")
        print(f"锁定的 kappa: {fixed_kappa}, 锁定的 theta: {fixed_theta}")
        print(f"拟合的 V0: {V0_opt:.4f}")
        print(f"拟合的 sigma: {sigma_opt:.4f}")
        print(f"拟合的 rho: {rho_opt:.4f}")
        return result.x