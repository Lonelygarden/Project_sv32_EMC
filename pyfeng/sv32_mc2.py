import abc
import math
import numpy as np
from . import sv_abc as sv
from . import heston_mc
import scipy.optimize as spop
import scipy.special as spsp
import scipy.stats as spst
from scipy.misc import derivative
from scipy.special import gammaln, digamma, loggamma, polygamma


class Sv32McABC(sv.SvABC, sv.CondMcBsmABC, abc.ABC):
    model_type = "3/2"
    var_process = True
    scheme = None
    _m_heston = None

    def set_num_params(self, n_path=10000, dt=None, rn_seed=None, antithetic=True):
        super().set_num_params(n_path, dt, rn_seed, antithetic)

        mr = self.mr * self.theta
        theta = (self.mr + self.vov**2) / mr
        self._m_heston = heston_mc.HestonMcAndersen2008(
            1 / self.sigma, self.vov, self.rho, mr, theta
        )
        self._m_heston.set_num_params(n_path, dt, rn_seed, antithetic)

    @abc.abstractmethod
    def cond_states_step(self, dt, var_0):
        """
        Final variance and integrated variance over dt given var_0
        The int_var is normalized by dt

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            (var_t, avgvar)
        """
        return NotImplementedError

    @staticmethod
    def iv_complex(nu, zz):
        """
        Modified Bessel function of the first kind with complex argument

        Args:
            nu: index
            zz: value

        Returns:

        """
        p0 = np.power(0.5 * zz, nu) / spsp.gamma(nu + 1)
        iv = p0.copy()
        zzh2 = (zz / 2) ** 2
        for kk in np.arange(1, 64):
            p0 *= zzh2 / (kk * (kk + nu))
            iv += p0
        return iv

    @staticmethod
    def iv_d12(nu, zz):
        """
        The 1st and 2nd derivative of modified Bessel function of the first kind w.r.t. the index nu

        Args:
            nu: index
            zz: value

        Returns:

        References:
            * https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/20/ShowAll.html
        """
        p0 = np.power(zz / 2, nu) / spsp.gamma(nu + 1)
        # psi_1 = np.full_like(zz, spsp.polygamma(1, nu + 1), dtype=float)
        psi_1 = spsp.polygamma(1, nu + 1)
        log_m_psi0 = np.log(zz / 2) - spsp.digamma(nu + 1)
        iv1 = log_m_psi0 * p0
        iv2 = (log_m_psi0**2 - psi_1) * p0

        kk_max = max(64, int((np.mean(zz) + 2 * np.std(zz)) * 20))
        zzh2 = (zz / 2) ** 2
        # print(f'kk: {kk_max}')
        for kk in np.arange(1, kk_max):
            p0 *= zzh2 / (kk * (kk + nu))
            log_m_psi0 -= 1 / (nu + kk)
            psi_1 -= 1 / (nu + kk) ** 2
            iv1 += log_m_psi0 * p0
            iv2 += (log_m_psi0**2 - psi_1) * p0
        return iv1, iv2

    def cond_avgvar_mv(self, dt, var_0, var_t, eta=None):
        """
        Mean and variance of the integrated variance conditional on initial var, final var, and eta

        Args:
            var_0: initial variance
            var_t: final variance
            eta: Poisson RV
            dt: time step

        Returns:
            (integarted variance / dt)
        """
        phi, _ = self._m_heston.phi_exp(dt)
        nu = self._m_heston.chi_dim() / 2 - 1

        vov2dt = self.vov**2 * dt
        d1_nu_bb = 4 / vov2dt / nu
        d2_nu_bb = -((4 / vov2dt) ** 2) / nu**3
        zz = phi / np.sqrt(var_0 * var_t)

        # print(f'phi: {phi}, nu: {nu}, zz: {zz.mean()}')
        if eta is None:
            iv = spsp.iv(nu, zz)
            iv_d1, iv_d2 = self.iv_d12(nu, zz)
            d1 = -(iv_d1 * d1_nu_bb) / iv
            var = (iv_d1 * d2_nu_bb + iv_d2 * d1_nu_bb**2) / iv - d1**2
        else:
            d1 = -spsp.digamma(eta + nu + 1) + np.log(zz / 2)
            var = d2_nu_bb * d1 - d1_nu_bb**2 * spsp.polygamma(1, eta + nu + 1)
            d1 *= -d1_nu_bb

        var[var < 0] = 1e-16
        return d1, var

    def cond_spot_sigma(self, texp, var_0):
        """
        from 01_SV_Simulation p13
        """
        tobs = self.tobs(texp)
        dt = np.diff(tobs, prepend=0)
        n_dt = len(dt)

        var_t = np.full(self.n_path, var_0)
        avgvar = np.zeros(self.n_path)

        for i in range(n_dt):
            var_t, avgvar_inc = self.cond_states_step(dt[i], var_t)
            avgvar += avgvar_inc * dt[i]

        avgvar /= texp
        spot_cond = (
            np.log(var_t / var_0)
            - texp * (self.mr * self.theta - (self.mr + self.vov**2 / 2) * avgvar)
        ) / self.vov - self.rho * avgvar * texp * 0.5
        np.exp(self.rho * spot_cond, out=spot_cond)
        sigma_cond = np.sqrt(
            (1.0 - self.rho**2) * avgvar / var_0
        )  # normalize by initial variance

        # return normalized forward and volatility
        return spot_cond, sigma_cond

    def return_var_realized(self, texp, cond):
        return None


class Sv32McTimeStep(Sv32McABC):
    """
    CONDITIONAL SIMULATION OF THE 3/2 MODEL

    """

    scheme = 1  # Milstein

    def var_step_euler(self, var_0, dt, milstein=True):
        """
        Simulate final variance with Euler/Milstein schemes (scheme = 0, 1)

        Args:
            var_0: initial variance
            dt: time step
            milstein: True or False (default)

        Returns:
            final variance (at t=T)
        """
        zz = self.rv_normal(spawn=0)

        # Euler scheme
        var_t = (
            1.0
            + self.mr * (self.theta - var_0) * dt
            + self.vov * np.sqrt(var_0 * dt) * zz
        )
        # Extra-term for Milstein scheme
        if milstein:
            var_t += 0.75 * self.vov**2 * var_0 * (zz**2 - 1.0) * dt

        var_t *= var_0
        var_t[var_t < 0] = 1e-16  # variance should be larger than zero

        return var_t

    def cond_states_step(self, dt, var_0):
        if self.scheme < 2:
            milstein = (self.scheme == 1)
            # Euler (or Milstein) scheme
            var_t = self.var_step_euler(var_0, dt, milstein=milstein)
        elif self.scheme == 2:
            # Exact method, but simulate steps in dt
            # Draw final variance after dt from NCX2 distribution
            var_t = self._m_heston.var_step_ncx2(dt, 1 / var_0)
            np.divide(1.0, var_t, out=var_t)
        elif self.scheme == 3:
            # Exact method, but simulate steps in dt
            # Draw final variance after dt from Poison-Gamma distribution
            var_t, _ = self._m_heston.var_step_pois_gamma(dt, 1 / var_0)
            np.divide(1.0, var_t, out=var_t)
        elif self.scheme == 4:
            # QE method
            var_t, _ = self._m_heston.var_step_qe(dt, 1/var_0)
            np.divide(1.0, var_t, out=var_t)
        else:
            raise ValueError(f"Invalid scheme: {self.scheme}")

        # Trapezoidal rule
        avgvar = (var_0 + var_t) / 2

        return var_t, avgvar


class Sv32McBaldeaux2012Exact(Sv32McABC):
    """
    EXACT SIMULATION OF THE 3/2 MODEL

    Parameters:
        sigma: float, initial volatility
        vov, mr, rho, theta: float, parameters of the 3/2 model, similar to Heston model where
            vov is the volatility of the variance process
            mr is the rate at which the variance reverts toward its long-term mean
            rho is correlation between asset price and volatility
            theta is the mean long-term variance
        intr, divr: float, interest rate and dividend yield
        is_fwd: Bool, true if asset price is forward
    """

    def cond_avgvar_laplace(self, bb, dt, var_0, var_t, eta=None):
        """
        Laplace transform of reciprocal of X_t, corresponding to Baldeaux2012 Eq(3.4)

        Parameters:
            bb: b^2/2
            dt: time step
            var_0, var_t: squared volatility at time 0, time t
        """

        vov2dt = self.vov**2 * dt
        phi, _ = self._m_heston.phi_exp(dt)
        nu = self._m_heston.chi_dim() / 2 - 1
        nu_bb = np.sqrt(nu**2 + 8 * bb / vov2dt)
        zz = phi / np.sqrt(var_0 * var_t)

        if eta is None:
            # nu_bb = np.array(nu_bb, dtype=np.complex128)
            # zz = np.array(zz, dtype=np.complex128)
            # ret = spsp.iv(nu_bb, zz) / spsp.iv(nu, zz)
            ret = self.iv_complex(nu_bb, zz) / spsp.iv(nu, zz)
        else:
            nu_diff = 8 * bb / self.vov**2 / (nu_bb + nu)
            ret = (
                spsp.gamma(eta + nu + 1)
                / spsp.gamma(eta + nu_bb + 1)
                * np.power(zz / 2, nu_diff)
            )

        return ret
    
    def find_truncation_N(self, h: float, epsilon: float, Phi_func: callable)-> int:
        """
        计算截断项数 N
        
        参数:
        h (float): 积分步长
        epsilon (float): 允许的误差容忍度 (例如 1e-4)
        Phi_func (callable): 特征函数 Φ(u)，输入实数 u，返回复数结果
        
        返回:
        int: 满足条件的 N 值
        """
        # 计算右侧的阈值
        threshold = (math.pi * epsilon) / 2.0
        
        j = 1
        max_iter = 100  # 设置安全上限，防止特征函数收敛过慢导致死循环
        
        while j <= max_iter:
            u = h * j
            
            # 计算特征函数 Φ(h*j)
            phi_val = Phi_func(u)
            
            # 计算当前项的绝对值 |Φ(h*j)| / j
            # 注意：phi_val 是复数，使用 abs() 计算其模长
            term_value = abs(phi_val) / j
            
            # 判断是否满足论文中的截断条件
            if term_value.all() < threshold:
                return j
            j += 1
        return max_iter
    
    def cond_avgvar_mv_analytic(self, dt, var_0, var_t):
        """
        Warn: not exact still
        Mean and variance of the average variance conditional on initial var, final var.
        It is computed from the analytical method in Low-bias simulation scheme for the Heston model by IG approximation(2013).

        Args:
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            mean, variance

        See Also:
            cond_avgvar_mv
        """
        phi, _ = self._m_heston.phi_exp(dt)
        nu = self._m_heston.chi_dim() / 2 - 1
        nu_bb = np.sqrt(nu**2)
        zz = phi / np.sqrt(var_0 * var_t)
        derivative1_of_numerator, derivative2_of_numerator = self.iv_d12(nu = nu_bb, zz = zz)
        denominator = self.iv_complex(nu, zz)
        f_prime = derivative1_of_numerator / denominator
        f_double_prime = derivative2_of_numerator / denominator
        p_prime = 4.0 / (self.vov**2 * nu_bb)
        p_double_prime = -16.0 / (self.vov**4 * (nu_bb**3))
        derivative1 = -(f_prime * p_prime) 

        # E[I^2] = L''(0) = f''(p) * (p'(0))^2 + f'(p) * p''(0)
        derivative2 = f_double_prime * (p_prime**2) + f_prime * p_double_prime
        m1 = derivative1
        var = derivative2 - m1**2
        return m1, var
    

    def draw_cond_avgvar(self, dt, var_0, var_t):
        """
        Draw condtional reciprocal of X_t, which is condtional average quared volatility

        Parameters:
            dt: time step
            var_0, var_t: squared volatility at time 0, time t
        """

        def laplace_cond(bb):
            return self.cond_avgvar_laplace(bb, dt, var_0, var_t)
        
        # Using the numeric derivatives
        # m1 = -derivative(laplace_cond, 0, n=1, dx=1e-5, order=5)
        # var = derivative(laplace_cond, 0, n=2, dx=1e-5, order=5) - m1**2
        
        # Using the analytic derivatives
        m1, var = self.cond_avgvar_mv_analytic(dt, var_0, var_t)
        
        ## Exclude the negative variances
        # idx = (var > np.finfo(float).eps)
        # avgvar = np.zeros_like(mean)
        # mean = mean[idx]
        # var = var[idx]
        # skew = skew[idx]
        ln_sig = np.sqrt(np.log(1 + var / m1**2))

        u_error = m1 + 12 * np.sqrt(np.fmax(var, 0))
        h = np.pi / (2*u_error)

        # N = np.ones(self.n_path)
        # for i in range(self.n_path):
        #    Nfun = lambda _N: mp.fabs(
        #        besseli_ufun(np.sqrt(nu**2 - 8j * h[i] * _N / self.vov**2), z[i]) / base_val[i]) \
        #                      - np.pi * self.error * _N / 2
        #    N[i] = int(spop.brentq(Nfun, 0, 1000)) + 1
        # N = N.max()
        # print(N)
        N = self.find_truncation_N(h, 1e-4, laplace_cond)

        # Store the value of characteristic function for each term in the summation when approximating the CDF
        jj = np.arange(1, N + 1)[:, None]
        phimat = laplace_cond(-1j * jj * h).real
        phimat = laplace_cond(-1j * jj * h).real

        # Sample the conditional integrated variance by inverse transform sampling
        zz = self.rv_normal(spawn=0)
        uu = spst.norm.cdf(zz)

        def root(xx):
            h_xx = h * xx
            rv = h_xx + 2 * (phimat * np.sin(h_xx * jj) / jj).sum(axis=0) - uu * np.pi
            return rv

        guess = m1 * np.exp(ln_sig * (zz - ln_sig / 2))
        avgvar = spop.newton(root, guess)

        return avgvar
    
    
    def cond_states_step(self, dt, var_0):
        """
        Sample variance at maturity and conditional integrated variance

        Args:
            dt: float, time to maturity
        Returns:
            tuple, variance at maturity and conditional integrated variance
        """

        var_t = self._m_heston.var_step_ncx2(dt, 1 / var_0)
        np.divide(1.0, var_t, out=var_t)

        avgvar = self.draw_cond_avgvar(dt, var_0, var_t)

        return var_t, avgvar


class Sv32McChoiKwok2023Ig(Sv32McBaldeaux2012Exact):

    dist = "ig"

    def draw_from_mean_var(self, mean, var, dist, skew=None,ifskew=None):
        """
        Draw RNs from distributions with mean and variance matched
        Args:
            mean: mean (1d array)
            var: variance (1d array)
            dist: distribution. 'ig' for IG, 'ga' for Gamma, 'ln' for log-normal

        Returns:
            RNs with size of mean/variance
        """
        # Inverse Gaussian Distribution needs mean and variance are all positive
        idx = np.logical_and(mean > np.finfo(float).eps, var > np.finfo(float).eps)
        # idx = (mean > np.finfo(float).eps)
        # idx = (var > np.finfo(float).eps)
        avgvar = np.zeros_like(mean)
        mean = mean[idx]
        var = var[idx]
        if ifskew:
            skew = skew[idx]
            # Set skew to eps if skew==0, avoiding 3/skew is inf or -inf
            skew[(skew == 0)] = np.finfo(float).eps
        if dist.lower() == "ig":
            # mu and lambda defined in https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
            # RNG.wald takes the same parameters
            if ifskew:
                lam = (3 / skew)**2
                eta = 3 * np.sqrt(var) / (mean*skew)
                avgvar[idx] = (1-eta)*mean + eta*mean*self.rng_spawn[1].wald(mean=1, scale=lam)
                avgvar[avgvar<0] = 0
            else:
                lam = mean**3 / var
                avgvar[idx] = self.rng_spawn[1].wald(mean=mean, scale=lam)

        elif dist.lower() == "ga":
            scale = var / mean
            shape = mean / scale
            avgvar[idx] = scale * self.rng_spawn[1].standard_gamma(shape=shape)
        elif dist.lower() == "ln":
            scale = np.sqrt(np.log(1 + var / mean**2))
            avgvar[idx] = mean * np.exp(scale * (self.rv_normal(spawn=1) - scale / 2))
        else:
            raise ValueError(f"Incorrect distribution: {dist}.")

        return avgvar

    def cond_avgvar_mv_numeric(self, dt, var_0, var_t):
        """
        Mean and variance of the average variance conditional on initial var, final var.
        It is computed from the numerical derivatives of the conditional Laplace transform.

        Args:
            var_0: initial variance
            var_t: final variance
            dt: time step

        Returns:
            mean, variance

        See Also:
            cond_avgvar_mv
        """

        # conditional Cumulant Generating Fuction
        def cumgenfunc_cond(bb):
            # return np.log(self.cond_avgvar_laplace(-bb, dt, var_0, var_t))
            return self.cond_avgvar_laplace(bb, dt, var_0, var_t)

        # m1 = derivative(cumgenfunc_cond, 0, n=1, dx=1e-5, order=5)
        # var = derivative(cumgenfunc_cond, 0, n=2, dx=1e-5d, order=5)
        m1 = -derivative(cumgenfunc_cond, 0, n=1, dx=1e-5, order=5)
        var = derivative(cumgenfunc_cond, 0, n=2, dx=1e-5, order=5)-m1**2
        skew = (-derivative(cumgenfunc_cond, 0, n=3, dx=1e-5, order=5)-3*m1*var+2*m1**3)/np.sqrt(var)**3
    
        return m1, var, skew

    def cond_states_step(self, dt, var_0):
        """
        Final variance and integrated variance over dt given var_0
        The int_var is normalized by dt

        Args:
            var_0: initial variance
            dt: time step

        Returns:
            (var_t, avgvar)
        """
        if self.dist == "ig" or self.dist == "ln":
            var_t = self._m_heston.var_step_ncx2(dt, 1 / var_0)
        elif self.dist == "ga":
            var_t, _ = self._m_heston.var_step_pois_gamma(1 / var_0, dt)
            
        np.divide(1.0, var_t, out=var_t)
        np.divide(1.0, var_0, out=var_0)
        ## The different ways to calculate var_t and avgvar
        m1_analytic, var_analytic = self.cond_avgvar_mv_analytic(dt, var_0, var_t)
        # m1_numeric, var_numeric, skew_numeric = self.cond_avgvar_mv_numeric(dt, var_0, var_t)
        avgvar = self.draw_from_mean_var(m1_analytic, var_analytic, self.dist, skew=None, ifskew=False)
        # var_t, avgvar = self.cond_states_step_invlap(var_0, dt)

        return var_t, avgvar
    
    
class Sv32McBrignoneJunike2026ConditionalCos(Sv32McABC):
    use_cos = True
    def __init__(self, intr, mr, theta, vov, rho, sigma):
        """
        初始化 3/2 模型参数 (Grasselli 2017 框架下 a=0, b=1)
        mr: 均值回归速度 (kappa)
        theta: 长期均值
        vov: 波动率的波动率 (sigma/epsilon)
        rho: 相关系数
        """
        self.intr = intr
        self.mr = mr
        self.theta = theta
        self.vov = vov
        self.rho = rho
        self.sigma = sigma
        
    def _bessel_ratio_complex(self, p, nu, z):
        """
        稳健计算复数阶贝塞尔函数比值: I_p(z) / I_nu(z)
        使用对数平移法彻底解决复数域下的下溢问题。
        """
        N_terms = 150
        k = np.arange(N_terms)[:, np.newaxis]  # [N_terms, 1]
        p_ext = np.asarray(p)[np.newaxis, :]   # [1, len(p)]
        
        log_z_half = np.log(z / 2.0)
        
        # 1. 计算分母 (实数项)
        log_wk_den = 2 * k * log_z_half - gammaln(k + 1) - gammaln(nu + k + 1)
        max_log_den = np.max(log_wk_den, axis=0)
        wk_den = np.exp(log_wk_den - max_log_den)
        sum_den = np.sum(wk_den, axis=0)
        
        # 2. 计算分子 (复数项，支持 u 为复数数组)
        log_wk_num = 2 * k * log_z_half - gammaln(k + 1) - loggamma(p_ext + k + 1)
        max_log_num = np.max(np.real(log_wk_num), axis=0)
        wk_num = np.exp(log_wk_num - max_log_num)
        sum_num = np.sum(wk_num, axis=0)
        
        # 3. 组合对数比值
        log_ratio = (p - nu) * log_z_half + np.log(sum_num) - np.log(sum_den) + max_log_num - max_log_den
        return np.exp(log_ratio)

    def _conditional_cf(self, u, v0, vt, T):
        """
        计算 3/2 模型对数收益率 X_T | V_T 的条件特征函数
        """
        # 提取 3/2 模型映射参数，将原本的 self.v0 全部替换为传入的动态 v0
        xi_1 = (self.intr + self.rho * self.mr / self.vov) * T - (self.rho / self.vov) * np.log(v0)
        xi_3 = self.rho / self.vov
        xi_5 = (self.rho / self.vov) * (self.vov**2 / 2.0 - self.mr * self.theta) - 0.5
        xi_8 = 1.0 - self.rho**2

        u2 = -1j * u * xi_5 + 0.5 * (u**2) * xi_8
        
        nu = (2.0 * self.mr * self.theta) / (self.vov**2) - 1.0
        # 同样，这里的 self.v0 也替换成了 v0
        z = (2.0 * self.mr * np.sqrt(v0 * vt)) / (self.vov**2 * np.sinh(self.mr * T / 2.0))
        
        p_u = np.sqrt(nu**2 + 8.0 * u2 / self.vov**2)
        ratio = self._bessel_ratio_complex(p_u, nu, z)
        
        ccf_val = np.exp(1j * u * (xi_1 + xi_3 * np.log(vt))) * ratio
        return ccf_val

    def get_32_moments_conditional(self, v0, vt, T):
        """
        修正：严格对齐 Grasselli (2017) 的 a=0, b=1 映射参数
        """
        nu = (2.0 * self.mr * self.theta) / (self.vov**2) - 1.0
        k_cir = self.mr
        
        # 核心修正：z 不取倒数，且系数仅为 mr
        z = (2.0 * k_cir * np.sqrt(v0 * vt)) / (self.vov**2 * np.sinh(k_cir * T / 2.0))
        
        ks = np.arange(100)
        z_arr = np.atleast_1d(z)
        log_z_half = np.log(z_arr / 2.0)
        
        log_wk = (nu + 2 * ks[:, np.newaxis]) * log_z_half[np.newaxis, :] \
                 - gammaln(ks[:, np.newaxis] + 1) \
                 - gammaln(nu + ks[:, np.newaxis] + 1)
        
        max_log_w = np.max(log_wk, axis=0)
        wk = np.exp(log_wk - max_log_w[np.newaxis, :])
        
        psi_k = digamma(nu + ks + 1)[:, np.newaxis]
        tri_k = polygamma(1, nu + ks + 1)[:, np.newaxis]
        diff_term = log_z_half[np.newaxis, :] - psi_k
        
        sum_wk = np.sum(wk, axis=0)
        dlnI_dp = np.sum(wk * diff_term, axis=0) / sum_wk
        d2I_over_I = np.sum(wk * (diff_term**2 - tri_k), axis=0) / sum_wk
        
        p_prime = 4.0 / (self.vov**2 * nu)
        p_double_prime = -16.0 / (self.vov**4 * nu**3)
        
        exp_I = -(dlnI_dp * p_prime)
        moment2_I = d2I_over_I * (p_prime**2) + dlnI_dp * p_double_prime
        var_I = np.maximum(0.0, moment2_I - exp_I**2)

        # 核心修正：严格按照 a=0, b=1 的公式，不混杂多余映射
        xi_1 = (self.intr + self.rho * self.mr / self.vov) * T - (self.rho / self.vov) * np.log(v0)
        xi_3 = self.rho / self.vov
        xi_5 = (self.rho / self.vov) * (self.vov**2 / 2.0 - self.mr * self.theta) - 0.5
        xi_8 = 1.0 - self.rho**2
        
        mean_X = xi_1 + xi_3 * np.log(vt) + xi_5 * exp_I
        var_X = xi_8 * exp_I + (xi_5**2) * var_I
        
        return np.squeeze(exp_I), np.squeeze(var_I), np.squeeze(mean_X), np.squeeze(var_X)

    def _cos_cdf(self, y, v0, vt, T, L, N, mu):
        """
        基于 Fourier-Cosine 级数计算累积分布函数 G(y)
        """
        k = np.arange(N)
        u_k = k * np.pi / (2.0 * L)
        
        # 注意：把 v0 传给条件特征函数
        ccf_vals = self._conditional_cf(u_k, v0, vt, T)
        
        f_hat = ccf_vals * np.exp(-1j * u_k * mu)
        c_k = (1.0 / L) * np.real(f_hat * np.exp(1j * k * np.pi / 2.0))
        c_k[0] /= 2.0  
        
        v_k = np.zeros(N)
        diff = min(y - mu, L) + L
        v_k[0] = diff
        if N > 1:
            k_pos = k[1:]
            v_k[1:] = (2.0 * L / (k_pos * np.pi)) * np.sin(k_pos * np.pi * diff / (2.0 * L))
            
        return np.sum(c_k * v_k)

    def simulate_log_return(self, v0, vt, T):
        # 修改处：提取 mean_X, var_X
        _, _, mean_X, var_X = self.get_32_moments_conditional(v0, vt, T)
        std_X = np.sqrt(var_X)
        
        L = 12.0 * std_X  
        N = 256           
        mu = mean_X
        
        U = np.random.uniform(0, 1)
        
        target_func = lambda y: self._cos_cdf(y, v0, vt, T, L, N, mu) - U
        try:
            y_sim = spop.brentq(target_func, mu - 0.95 * L, mu + 0.95 * L, xtol=1e-5)
        except ValueError:
            y_sim = np.random.normal(mean_X, std_X)
            
        return y_sim
    
    def _simulate_vT(self, v0, texp, n_paths):
        """
        修正：V_t 直接就是 CIR 过程，不需要取倒数！
        """
        k_cir = self.mr
        theta_cir = self.theta
        eps_x = self.vov
        
        c = 2.0 * k_cir / ((1.0 - np.exp(-k_cir * texp)) * eps_x**2)
        df = 4.0 * k_cir * theta_cir / (eps_x**2)
        nc_param = 2.0 * c * v0 * np.exp(-k_cir * texp)
        
        chi2_samples = np.random.noncentral_chisquare(df, nc_param, n_paths)
        vt_samples = chi2_samples / (2.0 * c)
        return vt_samples

    def cond_spot_sigma(self, texp, var_0):
        v0 = var_0
        n_paths = getattr(self, 'n_paths', 10000)
        vt_samples = self._simulate_vT(v0, texp, n_paths)
        
        if not self.use_cos:
            _, _, mean_X, var_X = self.get_32_moments_conditional(v0, vt_samples, texp)
            sigma_bs = np.sqrt(var_X / texp)
            sigma_base = np.sqrt(var_0) if getattr(self, 'var_process', True) else var_0
            sigma_cond = sigma_bs / sigma_base
            fwd_cond = np.exp(mean_X + 0.5 * var_X)
            
        else:
            # === 提速核心：获取所有的期望和方差，并预计算 C_k ===
            _, _, mean_X, var_X = self.get_32_moments_conditional(v0, vt_samples, texp)
            std_X = np.sqrt(var_X)
            L_array = 12.0 * std_X
            N_cos = 128  # COS 级数项，128 足以兼顾高精度与极速
            
            # (128, n_paths) 的系数矩阵，0.1秒内完成
            c_k_matrix = self.precompute_cos_ck(v0, vt_samples, texp, L_array, mean_X, N_cos)
            
            xt_samples = np.zeros(n_paths)
            U_samples = np.random.uniform(0, 1, n_paths)
            k_pos = np.arange(1, N_cos)  # 缓存常量数组
            
            for i in range(n_paths):
                c_k_i = c_k_matrix[:, i]
                L_i = L_array[i]
                mu_i = mean_X[i]
                U_i = U_samples[i]
                
                # 现在的 target_func 极其轻量，不再有任何复数运算
                def target_func(y):
                    diff = min(y - mu_i, L_i) + L_i
                    v_k = np.zeros(N_cos)
                    v_k[0] = diff
                    v_k[1:] = (2.0 * L_i / (k_pos * np.pi)) * np.sin(k_pos * np.pi * diff / (2.0 * L_i))
                    return np.sum(c_k_i * v_k) - U_i
                    
                try:
                    xt_samples[i] = spop.brentq(target_func, mu_i - 0.95 * L_i, mu_i + 0.95 * L_i, xtol=1e-4)
                except ValueError:
                    xt_samples[i] = np.random.normal(mu_i, std_X[i])
            
            fwd_cond = np.exp(xt_samples)
            sigma_cond = np.full(n_paths, 1e-8)
            
        return fwd_cond, sigma_cond
    
    def precompute_cos_ck(self, v0, vt_array, T, L_array, mu_array, N=128):
        """
        修正：同步更新底层映射的特征函数，保障 COS 方法输出真实分布
        """
        n_paths = len(vt_array)
        k = np.arange(N)[:, np.newaxis]
        u = k * np.pi / (2.0 * L_array[np.newaxis, :]) 
        
        # 核心修正
        xi_1 = (self.intr + self.rho * self.mr / self.vov) * T - (self.rho / self.vov) * np.log(v0)
        xi_3 = self.rho / self.vov
        xi_5 = (self.rho / self.vov) * (self.vov**2 / 2.0 - self.mr * self.theta) - 0.5
        xi_8 = 1.0 - self.rho**2

        u2 = -1j * u * xi_5 + 0.5 * (u**2) * xi_8
        
        nu = (2.0 * self.mr * self.theta) / (self.vov**2) - 1.0
        k_cir = self.mr
        # 核心修正：z 的内部不再使用倒数
        z = (2.0 * k_cir * np.sqrt(v0 * vt_array)) / (self.vov**2 * np.sinh(k_cir * T / 2.0))
        
        p_u = np.sqrt(nu**2 + 8.0 * u2 / self.vov**2)
        
        N_terms = 100 
        k_b = np.arange(N_terms)[:, np.newaxis, np.newaxis]     
        p_ext = p_u[np.newaxis, :, :]                           
        log_z_half = np.log(z / 2.0)[np.newaxis, np.newaxis, :] 
        
        log_wk_den = 2 * k_b * log_z_half - gammaln(k_b + 1) - gammaln(nu + k_b + 1)
        max_log_den = np.max(log_wk_den, axis=0)
        sum_den = np.sum(np.exp(log_wk_den - max_log_den), axis=0)
        
        log_wk_num = 2 * k_b * log_z_half - gammaln(k_b + 1) - loggamma(p_ext + k_b + 1)
        max_log_num = np.max(np.real(log_wk_num), axis=0)
        sum_num = np.sum(np.exp(log_wk_num - max_log_num), axis=0)
        
        log_ratio = (p_ext[0] - nu) * log_z_half[0] + np.log(sum_num) - np.log(sum_den) + max_log_num - max_log_den
        ratio = np.exp(log_ratio)
        
        ccf_vals = np.exp(1j * u * (xi_1 + xi_3 * np.log(vt_array[np.newaxis, :]))) * ratio
        
        f_hat = ccf_vals * np.exp(-1j * u * mu_array[np.newaxis, :])
        c_k = (1.0 / L_array[np.newaxis, :]) * np.real(f_hat * np.exp(1j * k * np.pi / 2.0))
        c_k[0, :] /= 2.0
        
        return c_k
    
    def cond_states_step(self, dt, var_0):
        """
        修正：同步移除步进过程中的所有倒数和倒数参数映射
        """
        v0_array = np.atleast_1d(var_0)
        n_paths = len(v0_array)
        if n_paths == 1 and hasattr(self, 'n_paths'):
            n_paths = self.n_paths
            v0_array = np.full(n_paths, v0_array[0])

        k_cir = self.mr
        theta_cir = self.theta
        eps_x = self.vov
        
        c = 2.0 * k_cir / ((1.0 - np.exp(-k_cir * dt)) * eps_x**2)
        df = 4.0 * k_cir * theta_cir / (eps_x**2)
        nc_param = 2.0 * c * v0_array * np.exp(-k_cir * dt)
        
        chi2_samples = np.random.noncentral_chisquare(df, nc_param, n_paths)
        var_t = chi2_samples / (2.0 * c)

        exp_I, var_I, _, _ = self.get_32_moments_conditional(v0_array, var_t, dt)

        exp_I = np.maximum(exp_I, 1e-12)
        var_I = np.maximum(var_I, 1e-12)
        
        s2 = np.log(1.0 + var_I / (exp_I**2))
        s = np.sqrt(s2)
        m = np.log(exp_I) - 0.5 * s2
        
        Z = np.random.standard_normal(n_paths)
        I_sample = np.exp(m + s * Z)
        
        avgvar = I_sample / dt
        
        if np.isscalar(var_0) and not hasattr(self, 'n_paths'):
            return var_t[0], avgvar[0]
            
        return var_t, avgvar