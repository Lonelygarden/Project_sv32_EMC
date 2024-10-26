# Case II: Set 2 in Kouarfate et al. (2021)
sigma, mr, theta, vov, rho = 0.06, 22.84, 0.218, 8.56, -0.99
intr = 0
strike, spot, texp = np.array([95, 100, 105]), 100, 0.5
p_exact = np.array([10.364, 7.386, 4.938])
iv_exact = pf.Bsm(1).impvol(p_exact, strike, spot, texp)
iv_exact * 100