"""
Variance Gamma Option Pricing Model
====================================
European option pricing via the Carr-Madan FFT method under the
Variance Gamma (VG) process of Madan, Carr & Chang (1998).

Provides:
  - FFT-based pricing for European calls and puts
  - Analytical Greeks via differentiation of the Fourier integrand
    (delta, gamma, theta, vega, rho, and VG-specific sensitivities)
  - Calibration of (sigma, nu, theta) to observed market prices

VG Parameters
-------------
  sigma : volatility of the Brownian motion component
  nu    : variance rate of the gamma subordinator
  theta : drift of the Brownian motion component (controls skewness)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class VarianceGammaModel:
    """Variance Gamma pricing engine for European options.

    Parameters
    ----------
    S     : float – spot price
    r     : float – continuously compounded risk-free rate
    q     : float – continuous dividend yield
    sigma : float – volatility of the BM component  (> 0)
    nu    : float – variance rate of the gamma time  (> 0)
    theta : float – drift of the BM component
    """

    def __init__(self, S: float, r: float, q: float,
                 sigma: float, nu: float, theta: float):
        self.S = S
        self.r = r
        self.q = q
        self.sigma = sigma
        self.nu = nu
        self.theta = theta

    # ------------------------------------------------------------------
    # Characteristic function machinery
    # ------------------------------------------------------------------

    def omega(self) -> float:
        """Martingale correction: w = (1/nu)*ln(1 - theta*nu - sigma^2*nu/2)."""
        return (1.0 / self.nu) * np.log(
            1.0 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu
        )

    def characteristic_function(self, u, T: float):
        """Characteristic function of ln(S_T) under the risk-neutral measure.

        phi(u) = exp(i*u*(ln S + (r-q+w)*T)) * (1 / (1 - i*u*theta*nu + sigma^2*nu*u^2/2))^(T/nu)
        """
        x = np.log(self.S)
        w = self.omega()
        drift = 1j * u * (x + (self.r - self.q + w) * T)
        vg_exp = -(T / self.nu) * np.log(
            1.0 - 1j * u * self.theta * self.nu
            + 0.5 * self.sigma ** 2 * self.nu * u ** 2
        )
        return np.exp(drift + vg_exp)

    # ------------------------------------------------------------------
    # Carr-Madan FFT pricer
    # ------------------------------------------------------------------

    @staticmethod
    def _simpson_weights(N: int):
        w = np.empty(N)
        w[0] = 1.0
        w[1::2] = 4.0
        w[2::2] = 2.0
        if N % 2 == 0:          # last weight for even N
            w[-1] = 1.0
        return w / 3.0

    def _carr_madan_psi(self, u, T: float, alpha: float):
        """Carr-Madan modified characteristic function Psi(u)."""
        v = u - (alpha + 1.0) * 1j
        phi = self.characteristic_function(v, T)
        denom = alpha ** 2 + alpha - u ** 2 + 1j * (2.0 * alpha + 1.0) * u
        return np.exp(-self.r * T) * phi / denom

    def _fft_prices(self, T: float, N: int = 4096, alpha: float = 1.5,
                    eta: float = 0.25):
        """Run the FFT and return (log-strike grid k, call price array)."""
        lam = 2.0 * np.pi / (N * eta)
        b = N * lam / 2.0

        j = np.arange(N)
        u = j * eta

        sw = self._simpson_weights(N)
        psi = self._carr_madan_psi(u, T, alpha)

        x = np.exp(1j * b * u) * psi * eta * sw
        fft_result = np.fft.fft(x)

        k = -b + lam * j
        call_prices = np.real(np.exp(-alpha * k) / np.pi * fft_result)
        return k, call_prices

    def price(self, K, T: float, option_type: str = "call",
              N: int = 4096, alpha: float = 1.5, eta: float = 0.25):
        """Price European call or put via FFT.

        Parameters
        ----------
        K : float or array – strike(s)
        T : float – time to expiry in years
        option_type : 'call' or 'put'

        Returns
        -------
        float or ndarray of option prices
        """
        K = np.atleast_1d(np.asarray(K, dtype=float))
        k_grid, call_prices = self._fft_prices(T, N, alpha, eta)
        log_K = np.log(K)
        prices = np.interp(log_K, k_grid, call_prices)

        if option_type.lower() == "put":
            prices = prices - self.S * np.exp(-self.q * T) + K * np.exp(-self.r * T)

        return float(prices[0]) if prices.size == 1 else prices

    # ------------------------------------------------------------------
    # Generic Greek via FFT with integrand modifier
    # ------------------------------------------------------------------

    def _greek_fft(self, T: float, modifier_fn, N: int, alpha: float,
                   eta: float):
        """Compute an FFT integral with *modifier_fn* applied to the integrand.

        modifier_fn(u, v, phi, T, alpha) -> modified integrand array
        where v = u - (alpha+1)*i and phi = char_fn(v, T).
        """
        lam = 2.0 * np.pi / (N * eta)
        b = N * lam / 2.0

        j = np.arange(N)
        u = j * eta
        v = u - (alpha + 1.0) * 1j

        sw = self._simpson_weights(N)

        phi = self.characteristic_function(v, T)
        denom = alpha ** 2 + alpha - u ** 2 + 1j * (2.0 * alpha + 1.0) * u
        base_psi = np.exp(-self.r * T) * phi / denom

        modified = modifier_fn(u, v, phi, denom, T, alpha)

        x = np.exp(1j * b * u) * modified * eta * sw
        fft_result = np.fft.fft(x)

        k = -b + lam * j
        values = np.real(np.exp(-alpha * k) / np.pi * fft_result)
        return k, values

    def _interp(self, K, k_grid, values):
        K = np.atleast_1d(np.asarray(K, dtype=float))
        res = np.interp(np.log(K), k_grid, values)
        return float(res[0]) if res.size == 1 else res

    # ------------------------------------------------------------------
    # Delta  =  dC/dS
    # ------------------------------------------------------------------

    def delta(self, K, T: float, option_type: str = "call",
              N: int = 4096, alpha: float = 1.5, eta: float = 0.25):
        r"""Analytical delta via FFT.

        dC/dS = (1/S) * (exp(-alpha*k)/pi) * Re int exp(-iuk) * (iu+alpha+1) * Psi(u) du
        """
        def modifier(u, v, phi, denom, T, alpha):
            psi = np.exp(-self.r * T) * phi / denom
            return (1j * u + alpha + 1.0) / self.S * psi

        k_grid, vals = self._greek_fft(T, modifier, N, alpha, eta)
        result = self._interp(K, k_grid, vals)

        K_arr = np.atleast_1d(np.asarray(K, dtype=float))
        if option_type.lower() == "put":
            adj = np.exp(-self.q * T)
            result = (np.atleast_1d(result) - adj)
            return float(result[0]) if result.size == 1 else result
        return result

    # ------------------------------------------------------------------
    # Gamma  =  d²C/dS²
    # ------------------------------------------------------------------

    def gamma(self, K, T: float,
              N: int = 4096, alpha: float = 1.5, eta: float = 0.25):
        r"""Analytical gamma via FFT (identical for calls and puts).

        d²C/dS² = (1/S²) * Re int … * (iu+alpha+1)(iu+alpha) * Psi du
        """
        def modifier(u, v, phi, denom, T, alpha):
            psi = np.exp(-self.r * T) * phi / denom
            a1 = 1j * u + alpha + 1.0
            a0 = 1j * u + alpha
            return a1 * a0 / (self.S ** 2) * psi

        k_grid, vals = self._greek_fft(T, modifier, N, alpha, eta)
        return self._interp(K, k_grid, vals)

    # ------------------------------------------------------------------
    # Theta  =  dC/dT   (with respect to time-to-expiry)
    # ------------------------------------------------------------------

    def _dphi_dT(self, v, T: float):
        """Partial derivative of the characteristic function w.r.t. T."""
        w = self.omega()
        drift_part = 1j * v * (self.r - self.q + w)
        vg_part = -(1.0 / self.nu) * np.log(
            1.0 - 1j * v * self.theta * self.nu
            + 0.5 * self.sigma ** 2 * self.nu * v ** 2
        )
        phi = self.characteristic_function(v, T)
        return (drift_part + vg_part) * phi

    def theta_greek(self, K, T: float, option_type: str = "call",
                    N: int = 4096, alpha: float = 1.5, eta: float = 0.25):
        r"""Analytical theta (dC/dT) via FFT.

        Note: this is the derivative with respect to *time to expiry* T.
        Option theta w.r.t. calendar time t is the negative of this.
        """
        def modifier(u, v, phi, denom, T, alpha):
            dphi = self._dphi_dT(v, T)
            return np.exp(-self.r * T) * (-self.r * phi + dphi) / denom

        k_grid, vals = self._greek_fft(T, modifier, N, alpha, eta)
        result = self._interp(K, k_grid, vals)

        if option_type.lower() == "put":
            K_arr = np.atleast_1d(np.asarray(K, dtype=float))
            adj = self.q * self.S * np.exp(-self.q * T) \
                  - self.r * K_arr * np.exp(-self.r * T)
            result = np.atleast_1d(result) + adj
            return float(result[0]) if result.size == 1 else result
        return result

    # ------------------------------------------------------------------
    # Vega  =  dC/d(sigma)
    # ------------------------------------------------------------------

    def _dphi_dsigma(self, v, T: float):
        """Partial derivative of the characteristic function w.r.t. sigma."""
        A = 1.0 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu
        domega = -self.sigma / A

        B = (1.0 - 1j * v * self.theta * self.nu
             + 0.5 * self.sigma ** 2 * self.nu * v ** 2)
        dvg = -(T / self.nu) * (self.sigma * self.nu * v ** 2) / B

        phi = self.characteristic_function(v, T)
        return (1j * v * domega * T + dvg) * phi

    def vega(self, K, T: float, option_type: str = "call",
             N: int = 4096, alpha: float = 1.5, eta: float = 0.25):
        """Analytical vega (dC/d sigma) via FFT.  Same for calls and puts."""
        def modifier(u, v, phi, denom, T, alpha):
            dphi = self._dphi_dsigma(v, T)
            return np.exp(-self.r * T) * dphi / denom

        k_grid, vals = self._greek_fft(T, modifier, N, alpha, eta)
        return self._interp(K, k_grid, vals)

    # ------------------------------------------------------------------
    # Rho  =  dC/dr
    # ------------------------------------------------------------------

    def rho(self, K, T: float, option_type: str = "call",
            N: int = 4096, alpha: float = 1.5, eta: float = 0.25):
        """Analytical rho (dC/dr) via FFT."""
        def modifier(u, v, phi, denom, T, alpha):
            dphi_dr = 1j * v * T * phi
            return np.exp(-self.r * T) * (-T * phi + dphi_dr) / denom

        k_grid, vals = self._greek_fft(T, modifier, N, alpha, eta)
        result = self._interp(K, k_grid, vals)

        if option_type.lower() == "put":
            K_arr = np.atleast_1d(np.asarray(K, dtype=float))
            result = np.atleast_1d(result) - K_arr * T * np.exp(-self.r * T)
            return float(result[0]) if result.size == 1 else result
        return result

    # ------------------------------------------------------------------
    # VG-specific parameter sensitivities
    # ------------------------------------------------------------------

    def _dphi_dtheta(self, v, T: float):
        """d phi / d theta  (VG skew parameter)."""
        A = 1.0 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu
        domega = -1.0 / A

        B = (1.0 - 1j * v * self.theta * self.nu
             + 0.5 * self.sigma ** 2 * self.nu * v ** 2)
        dvg = 1j * v * T / B

        phi = self.characteristic_function(v, T)
        return (1j * v * domega * T + dvg) * phi

    def _dphi_dnu(self, v, T: float):
        """d phi / d nu  (VG variance-rate parameter)."""
        A = 1.0 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu
        domega = (-(1.0 / self.nu ** 2) * np.log(A)
                  + (1.0 / self.nu) * (-self.theta - 0.5 * self.sigma ** 2) / A)

        B = (1.0 - 1j * v * self.theta * self.nu
             + 0.5 * self.sigma ** 2 * self.nu * v ** 2)
        dB = -1j * v * self.theta + 0.5 * self.sigma ** 2 * v ** 2
        dvg = (T / self.nu ** 2) * np.log(B) - (T / self.nu) * dB / B

        phi = self.characteristic_function(v, T)
        return (1j * v * domega * T + dvg) * phi

    def sensitivity_theta_param(self, K, T: float, option_type: str = "call",
                                N: int = 4096, alpha: float = 1.5,
                                eta: float = 0.25):
        """dC/d(theta_VG) via FFT.  Same for calls and puts."""
        def modifier(u, v, phi, denom, T, alpha):
            dphi = self._dphi_dtheta(v, T)
            return np.exp(-self.r * T) * dphi / denom

        k_grid, vals = self._greek_fft(T, modifier, N, alpha, eta)
        return self._interp(K, k_grid, vals)

    def sensitivity_nu(self, K, T: float, option_type: str = "call",
                       N: int = 4096, alpha: float = 1.5,
                       eta: float = 0.25):
        """dC/d(nu) via FFT.  Same for calls and puts."""
        def modifier(u, v, phi, denom, T, alpha):
            dphi = self._dphi_dnu(v, T)
            return np.exp(-self.r * T) * dphi / denom

        k_grid, vals = self._greek_fft(T, modifier, N, alpha, eta)
        return self._interp(K, k_grid, vals)

    # ------------------------------------------------------------------
    # Convenience: all Greeks at once
    # ------------------------------------------------------------------

    def greeks(self, K, T: float, option_type: str = "call", **fft_kw):
        """Return a dict of all Greeks for the given strike(s)."""
        return {
            "price": self.price(K, T, option_type, **fft_kw),
            "delta": self.delta(K, T, option_type, **fft_kw),
            "gamma": self.gamma(K, T, **fft_kw),
            "theta": self.theta_greek(K, T, option_type, **fft_kw),
            "vega": self.vega(K, T, option_type, **fft_kw),
            "rho": self.rho(K, T, option_type, **fft_kw),
            "d_theta_param": self.sensitivity_theta_param(K, T, option_type, **fft_kw),
            "d_nu": self.sensitivity_nu(K, T, option_type, **fft_kw),
        }

    # ------------------------------------------------------------------
    # Autodiff Greeks via PyTorch
    # ------------------------------------------------------------------

    @staticmethod
    def _torch_simpson_weights(N: int):
        w = torch.ones(N, dtype=torch.float64)
        w[0] = 1.0
        w[1::2] = 4.0
        w[2::2] = 2.0
        if N % 2 == 0:
            w[-1] = 1.0
        return w / 3.0

    @staticmethod
    def _torch_price_call(S, r, q, sigma, nu, theta_vg, K_val, T_val,
                          N=4096, alpha=1.5, eta=0.25):
        """Carr-Madan FFT call price built entirely from torch ops.

        All of S, r, q, sigma, nu, theta_vg, K_val, T_val may be
        torch tensors with requires_grad=True.  The returned scalar
        supports torch.autograd.grad.
        """
        lam = 2.0 * np.pi / (N * eta)
        b = N * lam / 2.0

        j = torch.arange(N, dtype=torch.float64)
        u_real = j * eta                        # real-valued grid

        # Simpson weights (constant, no grad needed)
        sw = VarianceGammaModel._torch_simpson_weights(N)

        # --- characteristic function at v = u - (alpha+1)*i -----------
        # We work with real / imag parts via torch complex tensors.
        v_real = u_real
        v_imag = -(alpha + 1.0) * torch.ones(N, dtype=torch.float64)
        v = torch.complex(v_real, v_imag)       # shape (N,)

        # omega (martingale correction)
        w = (1.0 / nu) * torch.log(
            1.0 - theta_vg * nu - 0.5 * sigma ** 2 * nu
        )

        x = torch.log(S)
        drift = 1j * v * (x + (r - q + w) * T_val)

        vg_inner = (1.0 - 1j * v * theta_vg * nu
                    + 0.5 * sigma ** 2 * nu * v ** 2)
        vg_exp = -(T_val / nu) * torch.log(vg_inner)
        phi = torch.exp(drift + vg_exp)

        # --- Carr-Madan integrand -------------------------------------
        u_cpx = torch.complex(u_real, torch.zeros(N, dtype=torch.float64))
        denom = (alpha ** 2 + alpha - u_cpx ** 2
                 + 1j * (2.0 * alpha + 1.0) * u_cpx)

        psi = torch.exp(-r * T_val) * phi / denom

        # --- FFT ------------------------------------------------------
        shift = torch.exp(
            torch.complex(
                torch.zeros(N, dtype=torch.float64),
                b * u_real
            )
        )
        fft_in = shift * psi * eta * sw
        fft_out = torch.fft.fft(fft_in)

        k_grid = -b + lam * j
        call_grid = (torch.exp(-alpha * k_grid) / np.pi) * fft_out.real

        # --- interpolate to desired log-strike ------------------------
        log_K = torch.log(K_val)
        # linear interp via searchsorted
        idx = torch.searchsorted(k_grid, log_K.detach()).clamp(1, N - 1)
        k_lo = k_grid[idx - 1]
        k_hi = k_grid[idx]
        c_lo = call_grid[idx - 1]
        c_hi = call_grid[idx]
        frac = (log_K - k_lo) / (k_hi - k_lo)
        call_price = c_lo + frac * (c_hi - c_lo)
        return call_price

    def greeks_ad(self, K, T: float, option_type: str = "call",
                  N: int = 4096, alpha: float = 1.5, eta: float = 0.25):
        """Compute all Greeks via PyTorch automatic differentiation.

        Returns a dict with keys: price, delta, gamma, theta, vega, rho,
        d_theta_param, d_nu.

        Requires PyTorch to be installed.
        """
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for greeks_ad(). "
                              "Install with:  pip install torch")

        # Build differentiable scalars
        S_t = torch.tensor(self.S, dtype=torch.float64, requires_grad=True)
        r_t = torch.tensor(self.r, dtype=torch.float64, requires_grad=True)
        q_t = torch.tensor(self.q, dtype=torch.float64, requires_grad=True)
        sigma_t = torch.tensor(self.sigma, dtype=torch.float64, requires_grad=True)
        nu_t = torch.tensor(self.nu, dtype=torch.float64, requires_grad=True)
        theta_t = torch.tensor(self.theta, dtype=torch.float64, requires_grad=True)
        T_t = torch.tensor(T, dtype=torch.float64, requires_grad=True)
        K_t = torch.tensor(float(K), dtype=torch.float64)

        # Forward pass — call price
        call = self._torch_price_call(
            S_t, r_t, q_t, sigma_t, nu_t, theta_t, K_t, T_t, N, alpha, eta
        )

        # First-order Greeks via autograd
        grads = torch.autograd.grad(
            call, [S_t, r_t, sigma_t, nu_t, theta_t, T_t],
            create_graph=True
        )
        delta_call = grads[0]
        rho_val = grads[1]
        vega_val = grads[2]
        dnu_val = grads[3]
        dtheta_param_val = grads[4]
        theta_val = grads[5]

        # Second-order: gamma = d(delta)/dS
        gamma_val = torch.autograd.grad(delta_call, S_t)[0]

        # Convert call to option price
        if option_type.lower() == "put":
            price_val = call.item() - self.S * np.exp(-self.q * T) \
                        + float(K) * np.exp(-self.r * T)
            delta_out = delta_call.item() - np.exp(-self.q * T)
            theta_out = theta_val.item() + self.q * self.S * np.exp(-self.q * T) \
                        - self.r * float(K) * np.exp(-self.r * T)
            rho_out = rho_val.item() - float(K) * T * np.exp(-self.r * T)
        else:
            price_val = call.item()
            delta_out = delta_call.item()
            theta_out = theta_val.item()
            rho_out = rho_val.item()

        return {
            "price": price_val,
            "delta": delta_out,
            "gamma": gamma_val.item(),
            "theta": theta_out,
            "vega": vega_val.item(),
            "rho": rho_out,
            "d_theta_param": dtheta_param_val.item(),
            "d_nu": dnu_val.item(),
        }

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def calibrate(S: float, r: float, q: float,
                  K_market, T_market, prices_market, option_types,
                  x0=None, bounds=None, weights=None,
                  use_global: bool = False):
        """Calibrate VG parameters (sigma, nu, theta) to market prices.

        Parameters
        ----------
        S, r, q        : market / model constants
        K_market        : array of strikes
        T_market        : array of expiries (one per option)
        prices_market   : array of observed option prices
        option_types    : list of 'call' / 'put' per option
        x0              : initial guess [sigma, nu, theta]
        bounds          : parameter bounds [(lo, hi), ...]
        weights         : array of weights per option (default: equal)
        use_global      : if True, run differential evolution first

        Returns
        -------
        (calibrated_model, OptimizeResult)
        """
        K_market = np.atleast_1d(np.asarray(K_market, dtype=float))
        T_market = np.atleast_1d(np.asarray(T_market, dtype=float))
        prices_market = np.atleast_1d(np.asarray(prices_market, dtype=float))

        if weights is None:
            weights = np.ones_like(prices_market)
        weights = np.atleast_1d(np.asarray(weights, dtype=float))

        if x0 is None:
            x0 = [0.2, 0.5, -0.1]
        if bounds is None:
            bounds = [(0.01, 2.0), (0.001, 5.0), (-1.0, 1.0)]

        def objective(params):
            sigma, nu, theta = params
            # Enforce the martingale existence condition
            if 1.0 - theta * nu - 0.5 * sigma ** 2 * nu <= 0:
                return 1e12
            try:
                model = VarianceGammaModel(S, r, q, sigma, nu, theta)
                err = 0.0
                for i in range(len(K_market)):
                    mp = model.price(K_market[i], T_market[i], option_types[i])
                    err += weights[i] * (mp - prices_market[i]) ** 2
                return err
            except Exception:
                return 1e12

        if use_global:
            res_global = differential_evolution(objective, bounds,
                                                seed=42, maxiter=200,
                                                tol=1e-10)
            x0 = res_global.x

        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-14})
        sigma_opt, nu_opt, theta_opt = result.x
        calibrated = VarianceGammaModel(S, r, q, sigma_opt, nu_opt, theta_opt)
        return calibrated, result


# ---------------------------------------------------------------------------
# Demo / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example parameters
    S = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    nu = 0.5
    theta = -0.1
    T = 0.5
    K = 100.0

    model = VarianceGammaModel(S, r, q, sigma, nu, theta)

    print("=" * 60)
    print("Variance Gamma European Option Pricer")
    print("=" * 60)
    print(f"  S={S}  K={K}  T={T}  r={r}  q={q}")
    print(f"  sigma={sigma}  nu={nu}  theta={theta}")
    print(f"  omega (martingale correction) = {model.omega():.6f}")
    print()

    call = model.price(K, T, "call")
    put = model.price(K, T, "put")
    print(f"  Call price : {call:.6f}")
    print(f"  Put  price : {put:.6f}")

    # Verify put-call parity
    parity = call - put - S * np.exp(-q * T) + K * np.exp(-r * T)
    print(f"  Put-call parity error : {parity:.2e}")
    print()

    print("--- Greeks (Call) ---")
    g = model.greeks(K, T, "call")
    for name, val in g.items():
        print(f"  {name:20s} = {val: .6f}")

    print()
    print("--- Greeks (Put) ---")
    g = model.greeks(K, T, "put")
    for name, val in g.items():
        print(f"  {name:20s} = {val: .6f}")

    # Quick finite-difference sanity check
    print()
    print("--- Finite-difference sanity check (call) ---")
    eps = 0.01
    m_up = VarianceGammaModel(S + eps, r, q, sigma, nu, theta)
    m_dn = VarianceGammaModel(S - eps, r, q, sigma, nu, theta)
    fd_delta = (m_up.price(K, T) - m_dn.price(K, T)) / (2 * eps)
    fd_gamma = (m_up.price(K, T) - 2 * model.price(K, T) + m_dn.price(K, T)) / eps ** 2
    print(f"  Analytical delta = {model.delta(K, T):.6f}   FD delta = {fd_delta:.6f}")
    print(f"  Analytical gamma = {model.gamma(K, T):.6f}   FD gamma = {fd_gamma:.6f}")

    eps_T = 1e-4
    fd_theta = (model.price(K, T + eps_T) - model.price(K, T - eps_T)) / (2 * eps_T)
    print(f"  Analytical theta = {model.theta_greek(K, T):.6f}   FD theta = {fd_theta:.6f}")

    eps_s = 1e-5
    m_su = VarianceGammaModel(S, r, q, sigma + eps_s, nu, theta)
    m_sd = VarianceGammaModel(S, r, q, sigma - eps_s, nu, theta)
    fd_vega = (m_su.price(K, T) - m_sd.price(K, T)) / (2 * eps_s)
    print(f"  Analytical vega  = {model.vega(K, T):.6f}   FD vega  = {fd_vega:.6f}")

    eps_r = 1e-5
    m_ru = VarianceGammaModel(S, r + eps_r, q, sigma, nu, theta)
    m_rd = VarianceGammaModel(S, r - eps_r, q, sigma, nu, theta)
    fd_rho = (m_ru.price(K, T) - m_rd.price(K, T)) / (2 * eps_r)
    print(f"  Analytical rho   = {model.rho(K, T):.6f}   FD rho   = {fd_rho:.6f}")

    # Autodiff Greeks comparison
    if _HAS_TORCH:
        print()
        print("--- Autodiff Greeks (Call) via PyTorch ---")
        g_ad = model.greeks_ad(K, T, "call")
        g_an = model.greeks(K, T, "call")
        print(f"  {'Greek':20s} {'Analytical':>14s} {'Autodiff':>14s} {'Diff':>12s}")
        for key in g_an:
            a = g_an[key]
            b = g_ad[key]
            print(f"  {key:20s} {a: 14.6f} {b: 14.6f} {abs(a-b): 12.2e}")

        print()
        print("--- Autodiff Greeks (Put) via PyTorch ---")
        g_ad = model.greeks_ad(K, T, "put")
        g_an = model.greeks(K, T, "put")
        print(f"  {'Greek':20s} {'Analytical':>14s} {'Autodiff':>14s} {'Diff':>12s}")
        for key in g_an:
            a = g_an[key]
            b = g_ad[key]
            print(f"  {key:20s} {a: 14.6f} {b: 14.6f} {abs(a-b): 12.2e}")
    else:
        print("\n  [PyTorch not installed — skipping autodiff comparison]")

    # Calibration demo
    print()
    print("=" * 60)
    print("Calibration demo")
    print("=" * 60)
    strikes = np.array([90, 95, 100, 105, 110], dtype=float)
    expiries = np.full_like(strikes, 0.5)
    opt_types = ["call"] * len(strikes)
    true_prices = np.array([model.price(k, 0.5, "call") for k in strikes])
    # Add small noise
    np.random.seed(0)
    noisy_prices = true_prices + np.random.normal(0, 0.02, len(strikes))

    print(f"  True params:  sigma={sigma}, nu={nu}, theta={theta}")
    cal_model, res = VarianceGammaModel.calibrate(
        S, r, q, strikes, expiries, noisy_prices, opt_types,
        use_global=True
    )
    print(f"  Calibrated:   sigma={cal_model.sigma:.4f}, "
          f"nu={cal_model.nu:.4f}, theta={cal_model.theta:.4f}")
    print(f"  Optimizer converged: {res.success}  |  SSE: {res.fun:.2e}")
