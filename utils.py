import os
import random
import numpy as np
import pandas as pd
import time
from contextlib import ContextDecorator
from openpyxl import load_workbook
from openpyxl.styles import Alignment
import pyfeng as pf
import pyfeng.ex as pfex

# Feller条件 2*mr*theta > *vov^2
cases_dict = {
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
        "description": "Eq. (4.2) in Baldeaux (2012), ATM",
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
        "description": "Set 2 in Kouarfate et al. (2021), ATM",
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
        "description": "in Kouarfate et al. (2021), ATM",
    },
    "Case IV": {
        "sigma": 1.0,
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
        "sigma": 1.0,
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
        "sigma": 1.0,
        "vov": 3.3,
        "rho": 0.5,
        "mr": 0.7,
        "theta": 0.6,
        "intr": 0.0,
        "texp": 1,
        "strike": np.array([47, 48, 49, 50, 51]),
        "spot": 49,
        "p_exact": np.array([12.7376, 12.3901, 12.0562, 11.7356, 11.4273]),
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
}

situation_dict = {
    "6D pre-shock": {
        "sigma": 0.419,
        "vov": 19.70,
        "rho": -0.24,
        "mr": 126.15,
        "theta": 0.164,
        "intr": 0.00,
        "texp": 6.0/365.25,
        "strike": np.array([96.0, 98.0, 100.0, 102.0, 104.0]),
        "spot": 100.0,
        "p_exact": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "description": "",
    },
    "6D during crash":{
        "sigma": 0.554,
        "vov": 20.0,
        "rho": -0.11,
        "mr": 126.15,
        "theta": 0.164,
        "intr": 0.00,
        "texp": 6.0/365.25,
        "strike": np.array([96.0, 98.0, 100.0, 102.0, 104.0]),
        "spot": 100.0,
        "p_exact": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "description": "",
    },
    "6D recovery":{
        "sigma": 0.471,
        "vov": 17.14,
        "rho": -0.22,
        "mr": 126.15,
        "theta": 0.164,
        "intr": 0.00,
        "texp": 6.0/365.25,
        "strike": np.array([96.0, 98.0, 100.0, 102.0, 104.0]),
        "spot": 100.0,
        "p_exact": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "description": "",
    },
    "20D pre-shock":{
        "sigma": 0.40,
        "vov": 13.26,
        "rho": -0.24,
        "mr": 126.15,
        "theta": 0.164,
        "intr": 0.00,
        "texp": 20.3/365.25,
        "strike": np.array([96.0, 98.0, 100.0, 102.0, 104.0]),
        "spot": 100.0,
        "p_exact": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "description": "",
    },
    "20D during crash":{
        "sigma": 0.487,
        "vov": 17.6,
        "rho": -0.25,
        "mr": 126.15,
        "theta": 0.164,
        "intr": 0.00,
        "texp": 20.3/365.25,
        "strike": np.array([96.0, 98.0, 100.0, 102.0, 104.0]),
        "spot": 100.0,
        "p_exact": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "description": "",
    },
    "20D recovery":{
        "sigma": 0.422,
        "vov": 11.97,
        "rho": -0.19,
        "mr": 126.15,
        "theta": 0.164,
        "intr": 0.00,
        "texp": 20.3/365.25,
        "strike": np.array([96.0, 98.0, 100.0, 102.0, 104.0]),
        "spot": 100.0,
        "p_exact": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "description": "",
    },
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
        

def run_valuation_for_single_model(model_class, model_scheme, case_names, case_dict, dt=None):
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
            dt = dt if dt is not None else 1/500
        else:
            # 其他模型（如解析解模型或其它蒙特卡洛模型）dt 设为 None
            dt = None
            
        # 3. 配置参数
        if model_class == pf.sv_fft.Sv32Fft or model_class == pf.sv_fft.Sv32FourierCos:
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


def run_valuation(model_class, model_scheme, scheme_name, case_names, case_dict, results_dict, dt=None, add_dt_to_index=False):
    print(f"\n{'='*10} 正在运行模型: {model_class.__name__} - {scheme_name} {'='*10}")
    
    for case_name in case_names:
        case = case_dict[case_name]
        np.random.seed(123456)
        # 1. 统一初始化模型
        m = model_class(
            sigma=case["sigma"], vov=case["vov"], rho=case["rho"],
            mr=case["mr"], theta=case["theta"], intr=case["intr"],
        )
        
        # 2. 差异化设置 dt
        if model_class == pfex.Sv32McTimeStep:
            # 优先使用外部传入的 dt，未传入则默认给一个值（或根据需求报错）
            current_dt = dt if dt is not None else 1/5000 
        else:
            current_dt = None
            
        # 3. 配置参数
        if model_class not in (pf.sv_fft.Sv32Fft, pf.sv_fft.Sv32FourierCos):
            m.set_num_params(n_path=1.6e5, dt=current_dt, rn_seed=123456)
            m.scheme = model_scheme
            m.correct_fwd = False
        
        # 4. 计时与计算
        start_time = time.perf_counter()
        
        strike = case['strike']
        spot = case['spot']
        texp = case['texp']
        p_exact = case['p_exact']
        
        bias = m.price(strike, spot, texp) - p_exact
        
        end_time = time.perf_counter()
        interval = end_time - start_time
        
        # 将 dt 转换为分数形式的字符串以美化表格（例如 "1/500"）
        dt_str = f"1/{int(1/current_dt)}" if current_dt else "N/A"
        print(f"[{case_name}]: 运行耗时: {interval:.6f} 秒 | dt: {dt_str}")
        
        # ================== 5. 核心排版逻辑 ==================
        
        strikes_arr = np.atleast_1d(strike)
        p_exact_arr = np.atleast_1d(p_exact)
        bias_arr = np.atleast_1d(bias)
        
        for i, (k, p, b) in enumerate(zip(strikes_arr, p_exact_arr, bias_arr)):
            # 关键修改：根据需求决定是否将 dt 纳入行的索引
            if add_dt_to_index:
                row_key = (case_name, dt_str, k)
            else:
                row_key = (case_name, k) 
                
            if row_key not in results_dict:
                results_dict[row_key] = {}
                
            # (1) 写入时间: 只在每个 block 的第一个 Strike 行写入，其余留空
            results_dict[row_key][(scheme_name, 'Time(s)')] = interval if i == 0 else ""
            
            # (2) 写入 Bias 并计算百分比
            rel_err = b / p if p != 0 else 0
            formatted_bias = f"{b:.4g} ({rel_err:.2%})" 
            
            results_dict[row_key][(scheme_name, 'Option Bias')] = formatted_bias

    print("-" * 30)


def format_excel_time_cells(excel_path):
    print(f"正在处理排版: {excel_path} ...")
    wb = load_workbook(excel_path)
    ws = wb.active
    
    # 1. 动态寻找 "Time(s)" 所在的列号和数据起始行
    time_cols = []
    header_row = None
    
    # 遍历前 5 行，寻找表头位置 (兼容 pandas 的各种多级索引输出格式)
    for r in range(1, 6):
        for c in range(1, ws.max_column + 1):
            if ws.cell(row=r, column=c).value == "Time(s)":
                time_cols.append(c)
                header_row = r
        if time_cols: 
            break
            
    if not time_cols:
        print(f"  [警告] 在 {excel_path} 中未找到 'Time(s)' 列，跳过合并。")
        return

    # Pandas 导出的多级表头，数据通常在 header_row 的下方 2 行处开始 (中间隔着 Index 名字)
    data_start_row = header_row + 2 

    # 2. 对每一个 Time(s) 列进行垂直合并
    for col in time_cols:
        start_row = None
        
        # 遍历数据行，直到最后一行 + 1（为了处理最后一组收尾）
        for row in range(data_start_row, ws.max_row + 2):
            cell = ws.cell(row=row, column=col) if row <= ws.max_row else None
            cell_val = cell.value if cell else None
            
            # 判断当前单元格是否有实际的时间数据 (非空且非空字符串)
            has_data = cell_val is not None and str(cell_val).strip() != ""
            
            # 如果遇到新数据，或者到达了表格最底部
            if has_data or row > ws.max_row:
                # 结算上一组的合并
                if start_row is not None:
                    end_row = row - 1
                    if end_row > start_row:
                        ws.merge_cells(start_row=start_row, start_column=col, end_row=end_row, end_column=col)
                    
                    # 无论是否跨越多行，都给组的起始单元格（也就是显示数据的单元格）设置居中对齐
                    ws.cell(row=start_row, column=col).alignment = Alignment(horizontal='center', vertical='center')
                
                # 开启新一组的起点
                if row <= ws.max_row:
                    start_row = row

    wb.save(excel_path)
    print(f"  [成功] 已完成合并并保存！\n")
    
    
def seed_everything(seed=123456):
    # 1. 固定 Python 内置 random 模块
    random.seed(seed)
    
    # 2. 固定 Numpy 的随机种子 (最常见的原因)
    np.random.seed(seed)
    
    # 3. 固定 Python hash 随机化 (防止字典/集合遍历顺序随机导致浮点数截断误差)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"全局随机数种子已锁定为: {seed}")