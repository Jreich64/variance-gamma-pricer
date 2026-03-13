"""
Arbitrary-precision Variance Gamma pricer using mpmath.

Provides call pricing, analytical delta & gamma, and finite-difference
delta & gamma at user-selectable decimal precision via mpmath.

All integrals are evaluated with mpmath.quad (adaptive Gauss-Legendre)
rather than FFT, giving true arbitrary precision at the cost of speed.
"""

import mpmath as mp


def set_precision(dps: int):
    """Set the number of decimal places for all mpmath operations."""
    mp.mp.dps = dps


def omega(sigma, theta, nu):
    """Martingale correction: w = (1/nu) * ln(1 - theta*nu - 0.5*sigma^2*nu)."""
    log_argument = mp.fadd(
        1.0,
        mp.fadd(mp.fmul(-theta, nu), mp.fmul(mp.fmul(-0.5, nu), mp.fmul(sigma, sigma))),
    )
    return mp.fdiv(mp.log(log_argument), nu)


def phi_x(sigma, theta, nu, t, u):
    """VG characteristic exponent (subordinated BM part)."""
    i = mp.mpc(0, 1)
    sigma_squared = mp.fmul(sigma, sigma)
    u_squared = mp.fmul(u, u)
    term_1 = 1.0
    term_2 = mp.fmul(mp.fmul(mp.fmul(-i, theta), nu), u)
    term_3 = mp.fmul(mp.fmul(mp.fmul(0.5, nu), sigma_squared), u_squared)
    exponent = mp.fdiv(-t, nu)
    argument = mp.fadd(mp.fadd(term_1, term_2), term_3)
    return mp.power(argument, exponent)


def phi_of_s_helper(S_0, r, q, sigma, theta, nu, t, u):
    """Drift + spot component of the full characteristic function."""
    i = mp.mpc(0, 1)
    w = omega(sigma, theta, nu)
    term_1 = mp.fadd(mp.log(S_0), mp.fmul(mp.fadd(mp.fsub(r, q), w), t))
    argument = mp.fmul(mp.fmul(i, u), term_1)
    return mp.exp(argument)


def phi_s(S_0, r, q, sigma, theta, nu, t, u):
    """Full characteristic function of ln(S_T)."""
    term_1 = phi_of_s_helper(S_0, r, q, sigma, theta, nu, t, u)
    term_2 = phi_x(sigma, theta, nu, t, u)
    return mp.fmul(term_1, term_2)


# ── Pi_2: risk-neutral exercise probability ──────────────────────────────

def pi_2_integrand_arg(S_0, r, q, sigma, theta, nu, t, u, K):
    i = mp.mpc(0, 1)
    curr_phi_s = phi_s(S_0, r, q, sigma, theta, nu, t, u)
    numerator = mp.fmul(mp.exp(mp.fmul(mp.fmul(-i, u), mp.log(K))), curr_phi_s)
    denominator = mp.fmul(i, u)
    return mp.re(mp.fdiv(numerator, denominator))


def pi_2(S_0, r, q, sigma, theta, nu, t, K):
    integral, error = mp.quad(
        lambda u: pi_2_integrand_arg(S_0, r, q, sigma, theta, nu, t, u, K),
        [0, mp.inf],
        error=True,
    )
    return mp.fadd(mp.fdiv(integral, mp.pi), 0.5), error


# ── Pi_1: delta probability ─────────────────────────────────────────────

def pi_1_integrand_arg(S_0, r, q, sigma, theta, nu, t, u, K):
    i = mp.mpc(0, 1)
    curr_phi_s1 = phi_s(S_0, r, q, sigma, theta, nu, t, mp.fsub(u, i))
    curr_phi_s2 = phi_s(S_0, r, q, sigma, theta, nu, t, mp.mpc(0, -1))
    numerator = mp.fmul(mp.exp(mp.fmul(mp.fmul(-i, u), mp.log(K))), curr_phi_s1)
    denominator = mp.fmul(mp.fmul(i, u), curr_phi_s2)
    return mp.re(mp.fdiv(numerator, denominator))


def pi_1(S_0, r, q, sigma, theta, nu, t, K):
    integral, error = mp.quad(
        lambda u: pi_1_integrand_arg(S_0, r, q, sigma, theta, nu, t, u, K),
        [0, mp.inf],
        error=True,
    )
    return mp.fadd(mp.fdiv(integral, mp.pi), 0.5), error


# ── Analytical gamma integrand ──────────────────────────────────────────

def gamma_integrand_arg(S_0, r, q, sigma, theta, nu, t, u, K):
    i = mp.mpc(0, 1)
    curr_phi_s1 = phi_s(S_0, r, q, sigma, theta, nu, t, mp.fsub(u, i))
    curr_phi_s2 = phi_s(S_0, r, q, sigma, theta, nu, t, mp.mpc(0, -1))
    numerator = mp.fmul(mp.exp(mp.fmul(mp.fmul(-i, u), mp.log(K))), curr_phi_s1)
    denominator = curr_phi_s2
    return mp.re(mp.fdiv(numerator, denominator))


def analytical_gamma(S_0, r, q, sigma, theta, nu, t, K):
    integral, error = mp.quad(
        lambda u: gamma_integrand_arg(S_0, r, q, sigma, theta, nu, t, u, K),
        [0, mp.inf],
        error=True,
    )
    gamma_val = mp.fmul(mp.exp(mp.fmul(-q, t)), mp.fdiv(integral, mp.fmul(S_0, mp.pi)))
    return gamma_val, error


# ── Price-only (2 integrals, no gamma) — used by FD functions ────────────

def _price_only(S_0, r, q, sigma, theta, nu, t, K):
    """Compute just the call price (2 integrals: pi_1 + pi_2, no gamma).

    Much faster than call_price when we only need the scalar price,
    e.g. inside finite-difference bumps.
    """
    curr_pi_1, _ = pi_1(S_0, r, q, sigma, theta, nu, t, K)
    curr_pi_2, _ = pi_2(S_0, r, q, sigma, theta, nu, t, K)
    term_1 = mp.fmul(mp.fmul(S_0, curr_pi_1), mp.exp(mp.fmul(-q, t)))
    term_2 = mp.fmul(mp.fmul(-K, mp.exp(mp.fmul(-r, t))), curr_pi_2)
    return mp.fadd(term_1, term_2)


# ── Call price (returns price, delta, gamma) ─────────────────────────────

def call_price(S_0, r, q, sigma, theta, nu, t, K):
    """Compute call price, analytical delta, and analytical gamma.

    Returns
    -------
    (call_price, delta, gamma, pi_1_error, pi_2_error, gamma_error)
    """
    curr_pi_1, pi_1_error = pi_1(S_0, r, q, sigma, theta, nu, t, K)
    curr_pi_2, pi_2_error = pi_2(S_0, r, q, sigma, theta, nu, t, K)
    term_1 = mp.fmul(mp.fmul(S_0, curr_pi_1), mp.exp(mp.fmul(-q, t)))
    term_2 = mp.fmul(mp.fmul(-K, mp.exp(mp.fmul(-r, t))), curr_pi_2)
    price = mp.fadd(term_1, term_2)
    delta = mp.fmul(mp.exp(mp.fmul(-q, t)), curr_pi_1)
    gamma_val, gamma_err = analytical_gamma(S_0, r, q, sigma, theta, nu, t, K)
    return price, delta, gamma_val, pi_1_error, pi_2_error, gamma_err


# ── Finite-difference Greeks (all use _price_only for speed) ─────────────

def fd_delta(S_0, r, q, sigma, theta, nu, t, K, eps=1e-20):
    eps = mp.mpf(eps)
    S_plus = mp.fadd(S_0, eps)
    S_minus = mp.fsub(S_0, eps)
    price_up = _price_only(S_plus, r, q, sigma, theta, nu, t, K)
    price_down = _price_only(S_minus, r, q, sigma, theta, nu, t, K)
    return mp.fdiv(mp.fsub(price_up, price_down), mp.fmul(2, eps))


def fd_gamma(S_0, r, q, sigma, theta, nu, t, K, eps=1e-20):
    """FD gamma via central difference of price: (C(S+h) - 2C(S) + C(S-h)) / h^2."""
    eps = mp.mpf(eps)
    S_plus = mp.fadd(S_0, eps)
    S_minus = mp.fsub(S_0, eps)
    p_up = _price_only(S_plus, r, q, sigma, theta, nu, t, K)
    p_mid = _price_only(S_0, r, q, sigma, theta, nu, t, K)
    p_dn = _price_only(S_minus, r, q, sigma, theta, nu, t, K)
    # (C(S+h) - 2*C(S) + C(S-h)) / h^2
    numer = mp.fadd(mp.fsub(p_up, mp.fmul(2, p_mid)), p_dn)
    return mp.fdiv(numer, mp.fmul(eps, eps))


def fd_theta(S_0, r, q, sigma, theta, nu, t, K, eps=1e-10):
    """Finite-difference theta: dC/dT."""
    eps = mp.mpf(eps)
    t_plus = mp.fadd(t, eps)
    t_minus = mp.fsub(t, eps)
    price_up = _price_only(S_0, r, q, sigma, theta, nu, float(t_plus), K)
    price_down = _price_only(S_0, r, q, sigma, theta, nu, float(t_minus), K)
    return mp.fdiv(mp.fsub(price_up, price_down), mp.fmul(2, eps))


def fd_vega(S_0, r, q, sigma, theta, nu, t, K, eps=1e-10):
    """Finite-difference vega: dC/d(sigma)."""
    eps = mp.mpf(eps)
    sig_plus = mp.fadd(sigma, eps)
    sig_minus = mp.fsub(sigma, eps)
    price_up = _price_only(S_0, r, q, float(sig_plus), theta, nu, t, K)
    price_down = _price_only(S_0, r, q, float(sig_minus), theta, nu, t, K)
    return mp.fdiv(mp.fsub(price_up, price_down), mp.fmul(2, eps))


def fd_rho(S_0, r, q, sigma, theta, nu, t, K, eps=1e-10):
    """Finite-difference rho: dC/dr."""
    eps = mp.mpf(eps)
    r_plus = mp.fadd(r, eps)
    r_minus = mp.fsub(r, eps)
    price_up = _price_only(S_0, float(r_plus), q, sigma, theta, nu, t, K)
    price_down = _price_only(S_0, float(r_minus), q, sigma, theta, nu, t, K)
    return mp.fdiv(mp.fsub(price_up, price_down), mp.fmul(2, eps))


def fd_theta_param(S_0, r, q, sigma, theta, nu, t, K, eps=1e-10):
    """Finite-difference sensitivity to VG theta parameter: dC/d(theta_VG)."""
    eps = mp.mpf(eps)
    th_plus = mp.fadd(theta, eps)
    th_minus = mp.fsub(theta, eps)
    price_up = _price_only(S_0, r, q, sigma, float(th_plus), nu, t, K)
    price_down = _price_only(S_0, r, q, sigma, float(th_minus), nu, t, K)
    return mp.fdiv(mp.fsub(price_up, price_down), mp.fmul(2, eps))


def fd_nu(S_0, r, q, sigma, theta, nu, t, K, eps=1e-10):
    """Finite-difference sensitivity to VG nu parameter: dC/d(nu)."""
    eps = mp.mpf(eps)
    nu_plus = mp.fadd(nu, eps)
    nu_minus = mp.fsub(nu, eps)
    price_up = _price_only(S_0, r, q, sigma, theta, float(nu_plus), t, K)
    price_down = _price_only(S_0, r, q, sigma, theta, float(nu_minus), t, K)
    return mp.fdiv(mp.fsub(price_up, price_down), mp.fmul(2, eps))


def all_greeks(S_0, r, q, sigma, theta, nu, t, K):
    """Compute price + all Greeks (analytical where available, FD otherwise).

    Returns dict with: price, delta, gamma (analytical via quadrature),
    fd_delta, fd_gamma, theta, vega, rho, d_theta_param, d_nu (all FD).

    Optimised integral count
    ────────────────────────
    • call_price       → 3 integrals (pi_1, pi_2, analytical gamma)
    • S bumps (shared)  → 2 × 2 = 4  (S+ε, S−ε — reuse px as centre)
      ↳ fd_delta + fd_gamma from the same 2 bumped prices + px
    • T bumps           → 2 × 2 = 4
    • σ bumps           → 2 × 2 = 4
    • r bumps           → 2 × 2 = 4
    • θ_VG bumps        → 2 × 2 = 4
    • ν bumps           → 2 × 2 = 4
    Total: 27 integrals  (was 33 before, 51 originally)
    """
    # ── 1. Analytical price, delta, gamma (3 integrals) ──────────────
    px, dlt, gma, e1, e2, e3 = call_price(S_0, r, q, sigma, theta, nu, t, K)

    # ── 2. S bumps: shared for fd_delta AND fd_gamma (4 integrals) ───
    eps_S = mp.mpf("1e-20")
    p_S_up = _price_only(mp.fadd(S_0, eps_S), r, q, sigma, theta, nu, t, K)
    p_S_dn = _price_only(mp.fsub(S_0, eps_S), r, q, sigma, theta, nu, t, K)
    fd_dlt = mp.fdiv(mp.fsub(p_S_up, p_S_dn), mp.fmul(2, eps_S))
    fd_gma = mp.fdiv(
        mp.fadd(mp.fsub(p_S_up, mp.fmul(2, px)), p_S_dn),
        mp.fmul(eps_S, eps_S),
    )

    # ── 3. Remaining FD Greeks (5 pairs × 4 integrals = 20) ─────────
    eps = mp.mpf("1e-10")

    # theta (dC/dT)
    p_t_up = _price_only(S_0, r, q, sigma, theta, nu, float(mp.fadd(t, eps)), K)
    p_t_dn = _price_only(S_0, r, q, sigma, theta, nu, float(mp.fsub(t, eps)), K)
    fd_th = mp.fdiv(mp.fsub(p_t_up, p_t_dn), mp.fmul(2, eps))

    # vega (dC/dσ)
    p_sig_up = _price_only(S_0, r, q, float(mp.fadd(sigma, eps)), theta, nu, t, K)
    p_sig_dn = _price_only(S_0, r, q, float(mp.fsub(sigma, eps)), theta, nu, t, K)
    fd_vg = mp.fdiv(mp.fsub(p_sig_up, p_sig_dn), mp.fmul(2, eps))

    # rho (dC/dr)
    p_r_up = _price_only(S_0, float(mp.fadd(r, eps)), q, sigma, theta, nu, t, K)
    p_r_dn = _price_only(S_0, float(mp.fsub(r, eps)), q, sigma, theta, nu, t, K)
    fd_rh = mp.fdiv(mp.fsub(p_r_up, p_r_dn), mp.fmul(2, eps))

    # d/d(theta_VG)
    p_thp_up = _price_only(S_0, r, q, sigma, float(mp.fadd(theta, eps)), nu, t, K)
    p_thp_dn = _price_only(S_0, r, q, sigma, float(mp.fsub(theta, eps)), nu, t, K)
    fd_thp = mp.fdiv(mp.fsub(p_thp_up, p_thp_dn), mp.fmul(2, eps))

    # d/d(nu)
    p_nu_up = _price_only(S_0, r, q, sigma, theta, float(mp.fadd(nu, eps)), t, K)
    p_nu_dn = _price_only(S_0, r, q, sigma, theta, float(mp.fsub(nu, eps)), t, K)
    fd_nv = mp.fdiv(mp.fsub(p_nu_up, p_nu_dn), mp.fmul(2, eps))

    return {
        "price": px,
        "delta": dlt,
        "gamma": gma,
        "fd_delta": fd_dlt,
        "fd_gamma": fd_gma,
        "theta": fd_th,
        "vega": fd_vg,
        "rho": fd_rh,
        "d_theta_param": fd_thp,
        "d_nu": fd_nv,
    }


# ── Quick self-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.mp.dps = 50
    T = 1.0
    r = 0.05
    q = 0.02
    theta = -0.14
    nu = 0.2
    sigma = 0.12
    S0 = 100
    K = 100

    price, delta, gamma, pi_1_error, pi_2_error, gamma_error = call_price(
        S0, r, q, sigma, theta, nu, T, K
    )
    my_fd_delta = fd_delta(S0, r, q, sigma, theta, nu, T, K)
    my_fd_gamma = fd_gamma(S0, r, q, sigma, theta, nu, T, K)

    print(f"Price:  {price}")
    print(f"Delta:  {delta}")
    print(f"Gamma:  {gamma}")
    print(f"FD Delta: {my_fd_delta}")
    print(f"FD Gamma: {my_fd_gamma}")
    print(f"Errors: pi1={pi_1_error}, pi2={pi_2_error}, gamma={gamma_error}")
