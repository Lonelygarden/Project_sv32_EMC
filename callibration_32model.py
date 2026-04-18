import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



def analyze_and_plot_pie_chart(phase_name, csv_filename):
    """
    Reads CSV, calculates Call/Put Buy/Sell ratios, and plots English pie charts.
    """
    print(f"\n>>> Processing Phase: {phase_name} ({csv_filename})")
    # Set a clean plot style
    plt.style.use('ggplot')
    if not os.path.exists(csv_filename):
        print(f"[Error] File {csv_filename} not found.")
        return

    df = pd.read_csv(csv_filename)
    if df.empty:
        print(f"[Warning] {csv_filename} is empty.")
        return

    # 1. Feature Extraction
    # Identify Call/Put from instrument name
    df['option_type'] = df['instrument_name'].apply(
        lambda x: 'Call' if x.endswith('-C') else ('Put' if x.endswith('-P') else 'Other')
    )
    
    # Format Direction
    df['trade_side'] = df['direction'].str.capitalize()
    
    # Create Category Label
    df['category'] = df['option_type'] + " " + df['trade_side']
    
    # 2. Statistical Aggregation (By Trade Amount/Volume)
    # Using 'amount' is more representative of market impact than 'count'
    stats = df.groupby('category')['amount'].sum()

    # Ensure all four quadrants exist even if 0
    required_labels = ['Call Buy', 'Call Sell', 'Put Buy', 'Put Sell']
    stats = stats.reindex(required_labels).fillna(0)

    # 3. Plotting Configuration
    labels = stats.index
    sizes = stats.values
    
    if sizes.sum() == 0:
        print(f"[Skip] Total volume is zero for {phase_name}.")
        return

    # Professional Color Palette:
    # Calls: Blues | Puts: Oranges/Reds
    colors = ['#3498db', '#2980b9', '#e67e22', '#d35400']
    explode = (0.05, 0, 0.05, 0) # Slightly separate Buys from Sells

    fig, ax = plt.subplots(figsize=(10, 7))
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        autopct='%1.1f%%',
        shadow=False, 
        startangle=140, 
        colors=colors,
        pctdistance=0.85,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    # Styling the percentage text
    plt.setp(autotexts, size=10, color="white")

    # Add a center circle to make it a Donut Chart (optional, cleaner look)
    # centre_circle = plt.Circle((0,0), 0.70, fc='white')
    # fig.gca().add_artist(centre_circle)

    # 4. English Titles and Legends
    plt.title(f"Option Trade Composition: {phase_name.replace('_', ' ')}", fontsize=16, pad=20)
    ax.legend(
        wedges, 
        labels,
        title="Trade Categories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=11
    )

    # Save Output
    output_name = f"Chart_{phase_name}.png"
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f"[Success] Saved chart as {output_name}")
    plt.close(fig)


def medvedev_scaillet_32_no_jump_iv(X_array, atm_iv, vov, rho):
    """
    剥离跳跃后的纯扩散曲率。
    在 T 极小时，它能提供的曲率上限被死死限制住了。
    """
    # 扩散倾斜项
    coef_X = (vov * rho / 4.0) * atm_iv
    # 扩散曲率项
    diff_curv = (vov**2 * atm_iv / 48.0) * (2.0 - rho**2)
    
    # 纯扩散的总曲率
    coef_X2 = diff_curv
    
    return atm_iv - coef_X * X_array + coef_X2 * (X_array**2)


def objective_ms_no_jump(params, X_array, market_ivs, atm_iv):
    vov, rho = params
    # 物理边界保护
    if vov <= 0.01 or vov > 20.0 or rho < -0.999 or rho > 0.999:
        return 1e10
    model_ivs = medvedev_scaillet_32_no_jump_iv(X_array, atm_iv, vov, rho)
    return np.mean((model_ivs - market_ivs)**2)


def calibrate_single_dataset_no_jump(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
    except Exception:
        return None, None, f"❌ 读取失败: {csv_filepath}"

    T = df['T'].iloc[0]
    r = 0.0
    df['X'] = -np.log(df['Moneyness']) + r * T
    
    # 提取 ATM IV
    atm_idx = np.abs(df['X']).argmin()
    atm_iv = df['iv'].iloc[atm_idx] / 100.0 if df['iv'].mean() > 5 else df['iv'].iloc[atm_idx]
    
    # 截取核心交战区
    df_filtered = df[(df['X'] >= -0.15) & (df['X'] <= 0.15)].copy()
    if len(df_filtered) < 3:
        return None, None, "❌ 过滤后样本不足"
        
    X_array = df_filtered['X'].values
    market_ivs = df_filtered['iv'].values / 100.0 if df_filtered['iv'].mean() > 5 else df_filtered['iv'].values
    
    # 仅需拟合 2 个参数: VoV 和 Rho
    initial_guess = [2.0, -0.5]
    bounds = [(0.1, 30.0), (-0.99, 0.99)]
    
    res = minimize(
        objective_ms_no_jump, initial_guess, 
        args=(X_array, market_ivs, atm_iv), 
        method='L-BFGS-B', bounds=bounds
    )
    
    if res.success:
        vov, rho = res.x
        return df_filtered, {"atm_iv": atm_iv, "vov": vov, "rho": rho}, "✅ 成功"
    else:
        return None, None, "❌ 优化失败"


def run_32_model_no_jump_analysis(days: int):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    files = {
        "1. Pre-Shock": f"Deribit_{days}D_1_Pre_Shock.csv",
        "2. Crash":     f"Deribit_{days}D_2_Crash.csv",
        "3. Recovery":  f"Deribit_{days}D_3_Recovery.csv"
    }
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    fig.suptitle(f'Volatility Smile Evolution ({days}-Day Maturity)\nFitted with Pure Diffusion Asymptotics', 
                 fontsize=16, fontweight='bold', y=1.05)
    
    for idx, (stage, filepath) in enumerate(files.items()):
        ax = axes[idx]
        print(f"\n正在处理 {stage} (纯扩散组)...")
        
        df_plot, params, status = calibrate_single_dataset_no_jump(filepath)
        
        if not params:
            ax.text(0.5, 0.5, f"Data Error:\n{status}", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"{stage}")
            continue
            
        print(f"  -> ATM IV: {params['atm_iv']*100:.1f}%, VoV: {params['vov']:.2f}, Rho: {params['rho']:.2f}")
        
        market_X = df_plot['X'].values
        market_ivs = df_plot['iv'].values / 100.0 if df_plot['iv'].mean() > 5 else df_plot['iv'].values
        
        smooth_X = np.linspace(-0.15, 0.15, 300)
        model_ivs = medvedev_scaillet_32_no_jump_iv(
            smooth_X, params["atm_iv"], params["vov"], params["rho"]
        )
        
        ax.scatter(market_X, market_ivs * 100, color='#1f77b4', alpha=0.7, edgecolor='black', s=40, label='Market Quotes')
        ax.plot(smooth_X, model_ivs * 100, color='#ff7f0e', linewidth=3, linestyle='--', label='Pure Diffusion Fit')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_ylim([min(market_ivs*100)*0.8, max(market_ivs*100)*1.2])
        
        title_color = 'black'
        ax.set_title(f"{stage}\nATM IV={params['atm_iv']*100:.1f}%\nVoV={params['vov']:.2f}, Rho={params['rho']:.2f}", 
                     fontsize=14, color=title_color,
                     family='Times New Roman')
        
        ax.set_xlabel('Log-Moneyness (X)', fontsize=12, family='Times New Roman')
        if idx == 0:
            ax.set_ylabel('Implied Volatility (%)', fontsize=12, family='Times New Roman')
            
        ax.legend(loc='best', fontsize=10, prop={'family': 'Times New Roman'})

    plt.tight_layout()
    plt.savefig(f'Flash_Crash_{days}D_Evolution_NoJump.png', bbox_inches='tight')
    plt.show()
    print(f"\n 对照组绘图完成！图片已保存为 Flash_Crash_{days}D_Evolution_NoJump.png")


def medvedev_scaillet_32_jump_iv(X_array, atm_iv, vov, rho, jump_vol, T):
    coef_X = (vov * rho / 4.0) * atm_iv
    diff_curv = (vov**2 * atm_iv / 48.0) * (2.0 - rho**2)
    jump_curv = (np.sqrt(2 * np.pi) / 4.0) * (jump_vol / (atm_iv**2 * np.sqrt(T)))
    coef_X2 = diff_curv + jump_curv
    return atm_iv - coef_X * X_array + coef_X2 * (X_array**2)


def objective_ms_jump(params, X_array, market_ivs, atm_iv, T):
    vov, rho, jump_vol = params
    # 物理边界保护
    if vov <= 0.01 or vov > 20.0 or rho < -0.999 or rho > 0.999 or jump_vol < 0.0:
        return 1e10
    model_ivs = medvedev_scaillet_32_jump_iv(X_array, atm_iv, vov, rho, jump_vol, T)
    return np.mean((model_ivs - market_ivs)**2)


def calibrate_single_dataset(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
    except Exception:
        return None, None, f"❌ 读取失败: {csv_filepath}"

    T = df['T'].iloc[0]
    r = 0.0
    df['X'] = -np.log(df['Moneyness']) + r * T
    
    # 提取 ATM IV
    atm_idx = np.abs(df['X']).argmin()
    atm_iv = df['iv'].iloc[atm_idx] / 100.0 if df['iv'].mean() > 5 else df['iv'].iloc[atm_idx]
    
    # 放宽 X 轴提取区间，让 OTM 散点尽可能多参与拟合 (-0.15 到 0.15 约等于 M 0.85 到 1.15)
    df_filtered = df[(df['X'] >= -0.15) & (df['X'] <= 0.15)].copy()
    if len(df_filtered) < 3:
        return None, None, "❌ 过滤后有效虚值样本不足 3 个"
        
    X_array = df_filtered['X'].values
    market_ivs = df_filtered['iv'].values / 100.0 if df_filtered['iv'].mean() > 5 else df_filtered['iv'].values
    
    # 寻优初始值与边界 (放开 jump_vol 上限捕捉极值)
    initial_guess = [2.0, -0.5, 0.5]
    bounds = [(0.1, 15.0), (-0.99, 0.99), (0.001, 30.0)]
    
    res = minimize(
        objective_ms_jump, initial_guess, 
        args=(X_array, market_ivs, atm_iv, T), 
        method='L-BFGS-B', bounds=bounds
    )
    
    if res.success:
        vov, rho, jump_vol = res.x
        return df_filtered, {"atm_iv": atm_iv, "vov": vov, "rho": rho, "jump_vol": jump_vol, "T": T}, "✅ 成功"
    else:
        return None, None, "❌ 优化失败"


def run_32_model_jump_analysis(days: int):
    # 填入你清洗出的文件名
    files = {
        "1. Pre-Shock": f"Deribit_{days}D_1_Pre_Shock.csv",
        "2. Crash":     f"Deribit_{days}D_2_Crash.csv",
        "3. Recovery":  f"Deribit_{days}D_3_Recovery.csv"
    }
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    fig.suptitle(f'Volatility Smile Evolution ({days}-Day Maturity)\nFitted with Jump-Diffusion Asymptotics', 
                 fontsize=16, fontweight='bold', y=1.05)
    
    for idx, (stage, filepath) in enumerate(files.items()):
        ax = axes[idx]
        print(f"\n正在处理 {stage}...")
        
        df_plot, params, status = calibrate_single_dataset(filepath)
        
        if not params:
            ax.text(0.5, 0.5, f"Data Error:\n{status}", ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"{stage}")
            continue
            
        # 打印控制台报告
        print(f"  -> ATM IV: {params['atm_iv']*100:.1f}%, VoV: {params['vov']:.2f}, Rho: {params['rho']:.2f}, Jump Vol: {params['jump_vol']:.2f}")
        
        market_X = df_plot['X'].values
        market_ivs = df_plot['iv'].values / 100.0 if df_plot['iv'].mean() > 5 else df_plot['iv'].values
        
        # 绘制拟合平滑曲线
        smooth_X = np.linspace(-0.15, 0.15, 300)
        model_ivs = medvedev_scaillet_32_jump_iv(
            smooth_X, params["atm_iv"], params["vov"], params["rho"], params["jump_vol"], params["T"]
        )
        
        # 散点与折线
        ax.scatter(market_X, market_ivs * 100, color='#1f77b4', alpha=0.7, edgecolor='black', s=40, label='Market Quotes')
        ax.plot(smooth_X, model_ivs * 100, color='#d62728', linewidth=3, label='Jump-Diffusion Fit')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # 统一 Y 轴，凸显 Crash 时的高度
        ax.set_ylim([min(market_ivs*100)*0.8, max(market_ivs*100)*1.2])
        
        title_color = 'black'
        ax.set_title(f"{stage}\nATM IV={params['atm_iv']*100:.1f}%, , VoV: {params['vov']:.2f}, Rho: {params['rho']:.2f}, Jump Vol={params['jump_vol']:.2f}", 
                     fontsize=14, color=title_color)
        
        ax.set_xlabel('Log-Moneyness (X)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Implied Volatility (%)', fontsize=12)
            
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'Flash_Crash_{days}D_Evolution.png', bbox_inches='tight')
    plt.show()
    print(f"\n✅ 绘图完成！图片已保存为 Flash_Crash_{days}D_Evolution.png")