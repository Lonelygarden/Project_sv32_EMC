import numpy as np
import time
from contextlib import ContextDecorator
import pyfeng as pf
import pyfeng.ex as pfex

case_dict = {
    "Case I": {
        "sigma": 1.0,
        "vov": 0.2,
        "rho": -0.5,
        "mr": 2.0,
        "theta": 1.5,
        "intr": 0.05,
        "texp": 1.0,
        "strike": 1.0,
        "spot": 1.0,
        "p_exact": 0.443059,
        "description": "Eq. (4.2) in Baldeaux (2012)",
    },
    "Case II": {
        "sigma": 0.06,
        "vov": 8.56,
        "rho": -0.99,
        "mr": 22.84,
        "theta": 0.218,
        "intr": 0.0,
        "texp": 0.5,
        "strike": np.array([95, 100, 105]),
        "spot": 100.0,
        "p_exact": np.array([10.364, 7.386, 4.938]),
        "description": "Set 2 in Kouarfate et al. (2021)",
    },
    "Case III": {
        "sigma": 0.06,
        "vov": 3.20,
        "rho": -0.99,
        "mr": 20.48,
        "theta": 0.218,
        "intr": 0.0,
        "texp": 0.5,
        "strike": np.array([95, 100, 105]),
        "spot": 100.0,
        "p_exact": np.array([11.724, 8.999, 6.710]),
        "description": "in Kouarfate et al. (2021)",
    },
    "Case IV": {
        "sigma": 1,
        "vov": 0.3,
        "rho": 0.5,
        "mr": 0.7,
        "theta": 0.6,
        "intr": 0.0,
        "texp": 0.1,
        "strike": np.array([47, 48, 49, 50, 51]),
        "spot": 49,
        "p_exact": np.array([7.04, 6.5, 6.121, 5.7006, 5.305]),
        "description": "Near expiration and atm",
    },
    "Case V": {
        "sigma": 1,
        "vov": 0.3,
        "rho": 0.5,
        "mr": 0.7,
        "theta": 0.6,
        "intr": 0.0,
        "texp": 0.1,
        "strike": np.array([10, 20, 40]),
        "spot": 50,
        "p_exact": np.array([39.9985, 30.002, 11.9266]),
        "description": "Near expiration only",
    },
    "Case VI": {
        "sigma": 1,
        "vov": 3.3,
        "rho": 0.5,
        "mr": 0.7,
        "theta": 0.6,
        "intr": 0.0,
        "texp": 1,
        "strike": np.array([47, 48, 49, 50, 51]),
        "spot": 49,
        "p_exact": np.array([12.6802, 12.333, 11.999, 11.678, 11.37]),
        "description": "atm only",
    },
    "Case VII": {
        "sigma": 0.06,
        "vov": 3.20,
        "rho": -0.99,
        "mr": 20.48,
        "theta": 0.218,
        "intr": 0.0,
        "texp": 0.5,
        "strike": np.array([95, 100, 105]),
        "spot": 100,
        "p_exact": np.array([11.7235, 8.9978, 6.7091]),
        "description": "Lewis AL (2000) Option valuation under stochastic volatility: with Mathematica code. Finance Press",
    },
    "Case VIII": {
        "sigma": 1,
        "vov": 3.3,
        "rho": 0.9,
        "mr": 12.0,
        "theta": 0.6,
        "intr": 0.0,
        "texp": 1,
        "strike": np.array([47, 48, 49, 50, 51]),
        "spot": 49,
        "p_exact": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "description": "atm only",
    }
}

class CaseTimer:
    def __init__(self, case_name, case_dict):
        self.case_name = case_name
        self.case_dict = case_dict
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"[{self.case_name}]: {self.case_dict[self.case_name]['description']} \n 运行耗时: {self.interval:.6f} 秒")
        

def run_valuation(model_class, model_scheme, case_names, case_dict):
    print(f"\n{'='*10} 正在运行模型: {model_class.__name__} {'='*10}")
    
    for case_name in case_names:
        case = case_dict[case_name]
        
        # 1. 统一初始化模型
        m = model_class(
            sigma=case["sigma"],
            vov=case["vov"],
            rho=case["rho"],
            mr=case["mr"],
            theta=case["theta"],
            intr=case["intr"],
        )
        
        # 2. 差异化设置 dt
        # 只有 Sv32McTimeStep 需要根据 Case 设置具体的 dt
        if model_class == pfex.Sv32McTimeStep:
            if case_name == "Case II":
                dt = 1 / 50000
            elif case_name == "Case VI":
                dt = 1 / 50000
            else:
                dt = 1 / 500
        else:
            # 其他模型（如解析解模型或其它蒙特卡洛模型）dt 设为 None
            dt = None
            
        # 3. 配置参数
        if model_class == pf.sv_fft.Sv32Fft:
            pass
        else:
            m.set_num_params(n_path=1.6e5, dt=dt, rn_seed=123456)
            m.scheme = model_scheme
            m.correct_fwd = False
        
        # 4. 使用之前定义的 CaseTimer 计时
        with CaseTimer(case_name=case_name, case_dict=case_dict):
            strike, spot, texp, p_exact = case['strike'], case['spot'], case['texp'], case['p_exact']
            bias = m.price(strike, spot, texp) - p_exact
            print(f"Bias: {bias} | dt: {dt}")
        
        print("-" * 30)