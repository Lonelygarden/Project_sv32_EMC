import numpy as np
import time
import scipy.stats as spst
import scipy.special as spsp
import copy

# import mp math as mp

import sys

sys.path.insert(0, "")
sys.path.insert(
    sys.path.index("") + 1,
    "D:\GitHUb\Project_sv32_EMC",
)
import pyfeng as pf
import pyfeng.ex as pfex

np.set_printoptions(precision=4)


# # Case I: Eq. (4.2) in Baldeaux (2012)
# mr, theta, vov, rho, sigma, intr = 2.0, 1.5, 0.2, -0.5, 1.0, 0.05
# strike, spot, texp = 1, 1, 1
# p_exact = 0.443059

# Case II: Set 2 in Kouarfate et al. (2021)
sigma, mr, theta, vov, rho = 0.06, 22.84, 0.218, 8.56, -0.99
intr = 0
strike, spot, texp = np.array([95, 100, 105]), 100, 0.5
p_exact = np.array([10.364, 7.386, 4.938])
iv_exact = pf.Bsm(1).impvol(p_exact, strike, spot, texp)
iv_exact * 100


# # Case III:in Kouarfate et al. (2021)
# sigma, mr, theta, vov, rho = 0.06, 20.48, 0.218, 3.20, -0.99
# intr = 0
# strike, spot, texp = np.array([95, 100, 105]), 100, 0.5
# p_exact = np.array([11.724, 8.999, 6.710])
# iv_exact = pf.Bsm(1).impvol(p_exact, strike, spot, texp)
# iv_exact * 100


# # Case IV:
# vov = 0.3
# mr = 0.7
# spot = 49  # asset price
# texp = 0.1
# intr = 0
# theta = 0.6
# strike = np.array([47, 48, 49, 50, 51])  # strike price
# sigma = 1  # sigma
# rho = 0.5
# p_exact = 0


# # Pricing with time discretezation using Euler/Milstein scheme, Exact Stepping, Almost Exact Stepping
# m = pfex.Sv32McTimeStep(sigma, vov, rho, mr, theta, intr)
# m.set_num_params(n_path=1.6e5, dt=1 / 500, rn_seed=123456)
# m.scheme = 1  # Euler/Milstein scheme, here dt should be small enough (dt=1/500 for Case I,III; not work well in Case II)
# m.correct_fwd = False
# bias = m.price(strike, spot, texp) - p_exact
# print(bias)


# # # Exact Stepping with 1 / NCX2
# m.set_num_params(n_path=1.6e5, dt=1 / 50, rn_seed=123456)
# m.scheme = 2
# bias = m.price(strike, spot, texp) - p_exact
# print(bias)


# # Pricing with Exact Simulation
# m1 = pfex.Sv32McBaldeaux2012Exact(sigma, vov, rho, mr, theta, intr)
# m1.set_num_params(n_path=10000, rn_seed=123456, dt=None)
# m1.correct_fwd = False
# bias = m1.price(strike, spot, texp) - p_exact
# print(bias)


# Pricing with IG approximation (Almost Exact Simulation)
m2 = pfex.Sv32McChoiKwok2023Ig(sigma, vov, rho, mr, theta, intr)
m2.set_num_params(n_path=100000, rn_seed=123456, dt=None)
m2.correct_fwd = False
bias = m2.price(strike, spot, texp) - p_exact
print(bias)  # Sometimes the deviation can touch 0.17


# # Pricing with Gamma-Poi approximation (Almost Exact Simulation)
# m2.dist = "ga"
# bias = m2.price(strike, spot, texp) - p_exact
# print(bias)
