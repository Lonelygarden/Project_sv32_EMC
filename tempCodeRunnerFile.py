m2 = pfex.Sv32McChoiKwok2023Ig(sigma, vov, rho, mr, theta, intr)
m2.set_num_params(n_path=100000, rn_seed=123456, dt=None)
m2.correct_fwd = False
bias = m2.price(strike, spot, texp) - p_exact
print(bias)  # Sometimes the deviation can touch 0.17
