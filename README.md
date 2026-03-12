# Variance Gamma Option Pricing Model

## 1. Overview

This project implements a complete European option pricing engine under the **Variance Gamma (VG) process** of Madan, Carr & Chang (1998). The VG model extends the classical Black-Scholes framework by replacing geometric Brownian motion with a pure-jump process that captures the **skewness** and **excess kurtosis** observed in real-world asset return distributions.

The pricing engine uses the **Carr-Madan FFT method** (Carr & Madan, 1999) to evaluate option prices across an entire strike spectrum in a single Fast Fourier Transform pass. On top of this, the project provides two independent Greek computation pipelines --- analytical differentiation of the Fourier integrand and automatic differentiation via PyTorch --- together with a calibration routine and an interactive Streamlit dashboard.

Key capabilities:

- FFT-based pricing of European calls and puts for arbitrary strikes and maturities.
- Eight analytical Greeks computed by modifying the Carr-Madan integrand (delta, gamma, theta, vega, rho, and VG-specific parameter sensitivities).
- Autodiff Greeks via PyTorch that are mathematically identical to the analytical Greeks (verified to machine precision).
- Model calibration to observed market prices via differential evolution and L-BFGS-B.
- A four-tab Streamlit application for interactive exploration, visualization, and calibration.

---

## 2. The Variance Gamma Process --- Mathematical Foundation

### 2.1 Construction

The Variance Gamma process is constructed by evaluating Brownian motion at a random time given by a gamma process. Formally, the VG process is defined as:

$$X_{VG}(t;\,\sigma,\nu,\theta) \;=\; \theta\,G(t;\nu) \;+\; \sigma\,W\!\bigl(G(t;\nu)\bigr)$$

where:

- $W(\cdot)$ is a standard Brownian motion (zero drift, unit variance per unit time).
- $G(t;\nu)$ is a **gamma process** with mean rate 1 and variance rate $\nu$. That is, $G(t;\nu) \sim \text{Gamma}\!\bigl(\text{shape}=t/\nu,\;\text{scale}=\nu\bigr)$, so $\mathbb{E}[G(t;\nu)] = t$ and $\text{Var}[G(t;\nu)] = \nu\,t$.

The gamma process $G$ acts as a **stochastic clock** (or business time). Intuitively, on days with high market activity the gamma increment is large, compressing more Brownian motion into a single calendar day, while on quiet days the increment is small. This subordination produces a distribution for $X_{VG}(t)$ that has heavier tails and, if $\theta \neq 0$, asymmetry --- features absent from the Gaussian distribution underlying Black-Scholes.

### 2.2 Parameters

| Symbol | Name | Role |
|--------|------|------|
| $\sigma$ | Diffusion volatility | Controls the **overall volatility** of the process. Specifically, $\sigma$ is the volatility of the Brownian motion component. Must satisfy $\sigma > 0$. |
| $\nu$ | Variance rate | Controls the **kurtosis** (tail heaviness) of the return distribution. Larger $\nu$ produces fatter tails. When $\nu \to 0$ the gamma subordinator degenerates to deterministic time and the VG process converges to ordinary Brownian motion with drift, recovering Black-Scholes. Must satisfy $\nu > 0$. |
| $\theta$ | Drift / skew | Controls the **skewness** of the return distribution. When $\theta < 0$, the distribution is left-skewed (large downward moves are more probable than large upward moves), consistent with the empirically observed leverage effect. When $\theta = 0$ the distribution is symmetric. |

### 2.3 Characteristic Function of the VG Process

The moment generating structure of the gamma subordinator gives a closed-form characteristic function. For the VG increment $X_{VG}(t)$ alone (before embedding in the risk-neutral stock price dynamics):

$$\varphi_{VG}(u;\,t) \;=\; \left(\frac{1}{1 - i\,u\,\theta\,\nu + \tfrac{1}{2}\sigma^2\nu\,u^2}\right)^{t/\nu}$$

This can be derived by conditioning on the gamma time change: $\varphi_{VG}(u;\,t) = \mathbb{E}\!\bigl[e^{iu\,X_{VG}(t)}\bigr] = \mathbb{E}\!\bigl[\mathbb{E}[e^{iu(\theta\,g + \sigma\,W(g))} \mid G(t)=g]\bigr]$. Conditional on $G(t)=g$, the exponent $\theta\,g + \sigma\,W(g)$ is Gaussian with mean $\theta\,g$ and variance $\sigma^2 g$, so the inner expectation equals $\exp(iu\theta g - \tfrac{1}{2}\sigma^2 u^2 g)$. Taking the outer expectation over $g \sim \text{Gamma}(t/\nu,\,\nu)$ yields the closed form above.

### 2.4 Risk-Neutral Stock Price Dynamics

Under the risk-neutral measure, the log stock price at maturity $T$ is modeled as:

$$\ln S_T \;=\; \ln S \;+\; (r - q + \omega)\,T \;+\; X_{VG}(T)$$

where:

- $S$ is the current spot price.
- $r$ is the continuously compounded risk-free rate.
- $q$ is the continuous dividend yield.
- $\omega$ is the **martingale correction** (convexity adjustment) that ensures $\mathbb{E}^{\mathbb{Q}}[S_T] = S\,e^{(r-q)T}$.

### 2.5 The Martingale Correction

For the discounted stock price to be a martingale, we require:

$$\mathbb{E}^{\mathbb{Q}}[e^{X_{VG}(T)}] = e^{-\omega\,T}$$

Evaluating the characteristic function at $u = -i$ gives $\mathbb{E}[e^{X_{VG}(T)}] = \varphi_{VG}(-i;\,T)$. Setting this equal to $e^{-\omega T}$ and solving for $\omega$:

$$\omega \;=\; \frac{1}{\nu}\,\ln\!\left(1 - \theta\,\nu - \tfrac{1}{2}\sigma^2\nu\right)$$

This requires the **martingale existence condition**:

$$1 - \theta\,\nu - \tfrac{1}{2}\sigma^2\nu \;>\; 0$$

If this condition is violated, the exponential moment $\mathbb{E}[e^{X_{VG}(T)}]$ does not exist and the model cannot price options. The Streamlit interface enforces this constraint in real time.

### 2.6 Full Characteristic Function of $\ln S_T$

Combining the drift and the VG characteristic exponent, the characteristic function of the log stock price is:

$$\phi(u;\,T) \;=\; \exp\!\Bigl(i\,u\bigl[\ln S + (r - q + \omega)\,T\bigr]\Bigr) \;\cdot\; \left(\frac{1}{1 - i\,u\,\theta\,\nu + \tfrac{1}{2}\sigma^2\nu\,u^2}\right)^{T/\nu}$$

In the code this is implemented as:

$$\phi(u;\,T) = \exp\!\bigl(\text{drift} + \text{vg\_exp}\bigr)$$

where $\text{drift} = i\,u\,\bigl(\ln S + (r-q+\omega)T\bigr)$ and $\text{vg\_exp} = -\frac{T}{\nu}\,\ln\!\bigl(1 - iu\theta\nu + \tfrac{1}{2}\sigma^2\nu u^2\bigr)$.

---

## 3. Carr-Madan FFT Pricing Method

### 3.1 The Pricing Integral

The Carr-Madan approach (1999) expresses the European call price as a Fourier integral over the characteristic function of the log stock price. Define $k = \ln K$ (log-strike). The call price is:

$$C(K) \;=\; \frac{e^{-\alpha\,k}}{\pi}\;\text{Re}\!\int_0^{\infty} e^{-iuk}\;\Psi(u)\;du$$

where $\Psi(u)$ is the **modified characteristic function**:

$$\Psi(u) \;=\; \frac{e^{-rT}\;\phi\bigl(u - (\alpha+1)i;\,T\bigr)}{\alpha^2 + \alpha - u^2 + i(2\alpha+1)\,u}$$

### 3.2 The Dampening Factor $\alpha$

The parameter $\alpha > 0$ is a **dampening factor** introduced to ensure that the call price integral converges. Without it, the Fourier transform of the call payoff does not exist because $C(K) \to S\,e^{-qT}$ as $K \to 0$. Multiplying by $e^{\alpha k}$ (equivalently $K^{\alpha}$) forces the modified call price $c(k) = e^{\alpha k} C(e^k)$ into $L^1$, making the inversion well-defined.

In practice, $\alpha = 1.5$ is a standard choice that balances numerical stability and accuracy. The code exposes $\alpha$ as a tunable parameter.

### 3.3 Discretization and Simpson's Rule

The integral is discretized on an evenly spaced grid in frequency space:

$$u_j = j\,\eta, \qquad j = 0, 1, \ldots, N-1$$

where $\eta$ is the frequency spacing and $N$ is the FFT size (typically $N = 4096$). Simpson's rule weights are applied for higher-order accuracy:

$$w_j = \frac{1}{3} \times \begin{cases} 1 & j = 0 \\[2pt] 4 & j \text{ odd} \\[2pt] 2 & j \text{ even},\; j \neq 0 \end{cases}$$

with the final weight adjusted to 1/3 when $N$ is even.

### 3.4 The FFT Trick

The key insight is that the discretized integral, evaluated at a grid of log-strikes $k_m$, has the form of a Discrete Fourier Transform. Define the log-strike grid:

$$k_m = -b + \lambda\,m, \qquad m = 0, 1, \ldots, N-1$$

where $b = N\lambda/2$ centers the grid. The relationship between the frequency spacing $\eta$ and the log-strike spacing $\lambda$ is fixed by the DFT:

$$\lambda\,\eta = \frac{2\pi}{N}$$

The discretized integral becomes:

$$C(k_m) \approx \frac{e^{-\alpha\,k_m}}{\pi}\;\text{Re}\!\sum_{j=0}^{N-1} e^{-iu_j k_m}\;\Psi(u_j)\;\eta\;w_j$$

Substituting $k_m = -b + \lambda\,m$ and $u_j = j\eta$:

$$C(k_m) \approx \frac{e^{-\alpha\,k_m}}{\pi}\;\text{Re}\!\sum_{j=0}^{N-1} e^{ib\,u_j}\;\Psi(u_j)\;\eta\;w_j \;\cdot\; e^{-i\lambda\eta\,jm}$$

The factor $e^{-i\lambda\eta\,jm} = e^{-i2\pi jm/N}$ is exactly the DFT kernel. Defining:

$$x_j = e^{ib\,u_j}\;\Psi(u_j)\;\eta\;w_j$$

we compute the entire call price curve via a single FFT: $Y = \text{FFT}(x)$, then:

$$C(k_m) = \frac{e^{-\alpha\,k_m}}{\pi}\;\text{Re}(Y_m)$$

### 3.5 Pseudocode

```
 1.  Set N = 4096, alpha = 1.5, eta = 0.25
 2.  Compute lambda = 2*pi / (N * eta)
 3.  Compute b = N * lambda / 2
 4.  Build frequency grid:  u_j = j * eta       for j = 0, 1, ..., N-1
 5.  Compute Simpson weights  w_j
 6.  For each j, evaluate:
       v_j  = u_j - (alpha + 1) * i
       phi_j = characteristic_function(v_j, T)
       denom_j = alpha^2 + alpha - u_j^2 + i*(2*alpha + 1)*u_j
       Psi_j = exp(-r*T) * phi_j / denom_j
 7.  Form FFT input:  x_j = exp(i * b * u_j) * Psi_j * eta * w_j
 8.  Compute FFT:  Y = FFT(x)
 9.  Build log-strike grid:  k_m = -b + lambda * m
10.  Recover call prices:  C(k_m) = exp(-alpha * k_m) / pi * Re(Y_m)
11.  Interpolate to the desired strike K using  k = ln(K)
```

### 3.6 Put Prices via Put-Call Parity

Once the call price $C$ is known, the corresponding put price is obtained via put-call parity:

$$P = C - S\,e^{-qT} + K\,e^{-rT}$$

---

## 4. Analytical Greeks via FFT

Each Greek is computed by differentiating the Carr-Madan pricing integral with respect to the relevant parameter. The key observation is that differentiation passes inside the integral and modifies the integrand $\Psi(u)$ by a known multiplicative or additive factor. The modified integrand is then processed through the same FFT machinery to produce the Greek over the entire strike grid.

The generic Greek computation uses the same infrastructure as pricing:

$$\text{Greek}(k_m) = \frac{e^{-\alpha\,k_m}}{\pi}\;\text{Re}\!\sum_{j=0}^{N-1} e^{ib\,u_j}\;\widetilde{\Psi}(u_j)\;\eta\;w_j \;\cdot\; e^{-i2\pi jm/N}$$

where $\widetilde{\Psi}$ is the modified integrand specific to each Greek.

### 4.1 Delta --- $\partial C / \partial S$

**Derivation.** The characteristic function depends on $S$ only through the factor $\exp(iu\ln S)$ in the drift term. Since $\partial/\partial S\;\exp(iu\ln S) = (iu/S)\exp(iu\ln S)$, and the Carr-Madan integrand evaluates $\phi$ at $v = u - (\alpha+1)i$, the full derivative of $\Psi$ with respect to $S$ is:

$$\frac{\partial\Psi}{\partial S} = \frac{iv + 1}{S}\;\Psi(u) = \frac{iu + \alpha + 1}{S}\;\Psi(u)$$

where the shift $v = u - (\alpha+1)i$ introduces the additional real terms.

**Modified integrand:**

$$\widetilde{\Psi}_\Delta(u) = \frac{iu + \alpha + 1}{S}\;\Psi(u)$$

**Put adjustment:** $\Delta_{\text{put}} = \Delta_{\text{call}} - e^{-qT}$

### 4.2 Gamma --- $\partial^2 C / \partial S^2$

**Derivation.** Differentiating the delta integrand once more with respect to $S$:

$$\frac{\partial^2\Psi}{\partial S^2} = \frac{(iu + \alpha + 1)(iu + \alpha)}{S^2}\;\Psi(u)$$

**Modified integrand:**

$$\widetilde{\Psi}_\Gamma(u) = \frac{(iu + \alpha + 1)(iu + \alpha)}{S^2}\;\Psi(u)$$

Gamma is identical for calls and puts (put-call parity's $S$-dependent term $-Se^{-qT}$ has second derivative zero with respect to $S$, since $\partial^2(-Se^{-qT})/\partial S^2 = 0$).

### 4.3 Theta --- $\partial C / \partial T$

**Derivation.** The time-to-expiry $T$ appears in three places within $\Psi$: the discount factor $e^{-rT}$, the drift $\exp\!\bigl(iv(r-q+\omega)T\bigr)$, and the VG exponent $-(T/\nu)\ln(\cdots)$. Differentiating:

$$\frac{\partial\phi}{\partial T}(v,T) = \left[iv(r - q + \omega) - \frac{1}{\nu}\ln\!\bigl(1 - iv\theta\nu + \tfrac{1}{2}\sigma^2\nu v^2\bigr)\right]\phi(v,T)$$

The full derivative of the discounted integrand is:

$$\frac{\partial}{\partial T}\bigl[e^{-rT}\phi(v,T)/D(u)\bigr] = \frac{e^{-rT}}{D(u)}\left[-r\,\phi(v,T) + \frac{\partial\phi}{\partial T}(v,T)\right]$$

where $D(u) = \alpha^2 + \alpha - u^2 + i(2\alpha+1)u$ is the Carr-Madan denominator.

**Modified integrand:**

$$\widetilde{\Psi}_\Theta(u) = \frac{e^{-rT}}{D(u)}\left[-r\,\phi(v,T) + \frac{\partial\phi}{\partial T}(v,T)\right]$$

**Put adjustment:** $\Theta_{\text{put}} = \Theta_{\text{call}} + q\,S\,e^{-qT} - r\,K\,e^{-rT}$

**Note:** This is the derivative with respect to time-to-expiry $T$. The option theta with respect to calendar time $t$ is the negative: $\partial C/\partial t = -\partial C/\partial T$.

### 4.4 Vega --- $\partial C / \partial\sigma$

**Derivation.** The parameter $\sigma$ appears in two places: the martingale correction $\omega$ and the VG characteristic exponent. Define:

$$A = 1 - \theta\nu - \tfrac{1}{2}\sigma^2\nu, \qquad B(v) = 1 - iv\theta\nu + \tfrac{1}{2}\sigma^2\nu v^2$$

Then:

$$\frac{\partial\omega}{\partial\sigma} = \frac{-\sigma}{A}$$

$$\frac{\partial}{\partial\sigma}\!\left[-\frac{T}{\nu}\ln B(v)\right] = -\frac{T}{\nu}\;\frac{\sigma\nu v^2}{B(v)}$$

Combining:

$$\frac{\partial\phi}{\partial\sigma}(v,T) = \left[iv\,\frac{\partial\omega}{\partial\sigma}\,T - \frac{T}{\nu}\;\frac{\sigma\nu v^2}{B(v)}\right]\phi(v,T)$$

**Modified integrand:**

$$\widetilde{\Psi}_{\text{Vega}}(u) = \frac{e^{-rT}}{D(u)}\;\frac{\partial\phi}{\partial\sigma}(v,T)$$

Vega is identical for calls and puts (the put-call parity adjustment terms do not depend on $\sigma$).

### 4.5 Rho --- $\partial C / \partial r$

**Derivation.** The rate $r$ appears in the discount factor and the drift:

$$\frac{\partial\phi}{\partial r}(v,T) = iv\,T\;\phi(v,T)$$

$$\frac{\partial}{\partial r}\bigl[e^{-rT}\phi/D\bigr] = \frac{e^{-rT}}{D}\bigl[-T\,\phi + iv\,T\,\phi\bigr]$$

**Modified integrand:**

$$\widetilde{\Psi}_\rho(u) = \frac{e^{-rT}}{D(u)}\bigl(-T + iv\,T\bigr)\;\phi(v,T)$$

**Put adjustment:** $\rho_{\text{put}} = \rho_{\text{call}} - K\,T\,e^{-rT}$

### 4.6 Sensitivity to $\theta_{VG}$ --- $\partial C / \partial\theta$

**Derivation.** The VG skew parameter $\theta$ (denoted $\theta_{VG}$ to avoid confusion with the Greek theta) appears in $\omega$ and the VG exponent:

$$\frac{\partial\omega}{\partial\theta} = \frac{-1}{A}$$

$$\frac{\partial}{\partial\theta}\!\left[-\frac{T}{\nu}\ln B(v)\right] = -\frac{T}{\nu}\;\frac{-iv\nu}{B(v)} = \frac{ivT}{B(v)}$$

$$\frac{\partial\phi}{\partial\theta}(v,T) = \left[\frac{-iv\,T}{A} + \frac{iv\,T}{B(v)}\right]\phi(v,T)$$

**Modified integrand:**

$$\widetilde{\Psi}_{d\theta}(u) = \frac{e^{-rT}}{D(u)}\;\frac{\partial\phi}{\partial\theta}(v,T)$$

This sensitivity is the same for calls and puts.

### 4.7 Sensitivity to $\nu$ --- $\partial C / \partial\nu$

**Derivation.** The parameter $\nu$ appears in $\omega$ (both inside the $\ln$ and as $1/\nu$) and in the VG exponent (as $T/\nu$ and inside $B$):

$$\frac{\partial\omega}{\partial\nu} = -\frac{1}{\nu^2}\ln A + \frac{1}{\nu}\;\frac{-\theta - \tfrac{1}{2}\sigma^2}{A}$$

For the VG exponent, define $\partial B/\partial\nu = -i v\theta + \tfrac{1}{2}\sigma^2 v^2$. Then:

$$\frac{\partial}{\partial\nu}\!\left[-\frac{T}{\nu}\ln B\right] = \frac{T}{\nu^2}\ln B - \frac{T}{\nu}\;\frac{\partial B/\partial\nu}{B}$$

$$\frac{\partial\phi}{\partial\nu}(v,T) = \left[iv\,\frac{\partial\omega}{\partial\nu}\,T + \frac{T}{\nu^2}\ln B(v) - \frac{T}{\nu}\;\frac{\partial B/\partial\nu}{B(v)}\right]\phi(v,T)$$

**Modified integrand:**

$$\widetilde{\Psi}_{d\nu}(u) = \frac{e^{-rT}}{D(u)}\;\frac{\partial\phi}{\partial\nu}(v,T)$$

This sensitivity is the same for calls and puts.

### 4.8 Summary of Put Adjustments

All put-specific adjustments derive from differentiating the put-call parity relation $P = C - Se^{-qT} + Ke^{-rT}$:

| Greek | Put = Call adjustment |
|-------|----------------------|
| Delta | $\Delta_P = \Delta_C - e^{-qT}$ |
| Gamma | $\Gamma_P = \Gamma_C$ (no adjustment) |
| Theta | $\Theta_P = \Theta_C + qSe^{-qT} - rKe^{-rT}$ |
| Vega  | $\text{Vega}_P = \text{Vega}_C$ (no adjustment) |
| Rho   | $\rho_P = \rho_C - KTe^{-rT}$ |
| $\partial/\partial\theta_{VG}$ | Same for call and put |
| $\partial/\partial\nu$ | Same for call and put |

---

## 5. Autodiff Greeks via PyTorch

### 5.1 Concept

The analytical Greeks derived in Section 4 require manually differentiating the characteristic function with respect to each parameter --- a tedious and error-prone process. An alternative is **automatic differentiation** (autodiff): reimplement the entire Carr-Madan FFT pricing algorithm using differentiable tensor operations, then let the autodiff engine compute exact derivatives by backpropagating through the computation graph.

The implementation rebuilds the pricing pipeline entirely in PyTorch:

1. Each model parameter ($S$, $r$, $q$, $\sigma$, $\nu$, $\theta$, $T$) is wrapped as a `torch.Tensor` with `requires_grad=True`.
2. The omega computation, characteristic function evaluation, Carr-Madan integrand construction, FFT, and strike interpolation are all expressed using PyTorch tensor operations (`torch.log`, `torch.exp`, `torch.fft.fft`, `torch.complex`, etc.).
3. The output is a single scalar call price that sits at the root of a PyTorch computation graph.

### 5.2 Computing Greeks with `torch.autograd.grad`

First-order Greeks are obtained in a single call:

```python
grads = torch.autograd.grad(
    call_price,
    [S_t, r_t, sigma_t, nu_t, theta_t, T_t],
    create_graph=True
)
```

This returns `[dC/dS, dC/dr, dC/dsigma, dC/dnu, dC/dtheta, dC/dT]` in one backward pass through the graph.

### 5.3 Second-Order Derivatives (Gamma)

The flag `create_graph=True` tells PyTorch to build a computation graph for the gradient computation itself. This allows a second backward pass to compute second-order derivatives:

```python
gamma = torch.autograd.grad(delta_call, S_t)[0]
```

This computes $\Gamma = \partial\Delta/\partial S = \partial^2 C/\partial S^2$ exactly.

### 5.4 Complex Tensor Arithmetic

PyTorch supports complex-valued tensors with full gradient tracking. The VG characteristic function involves complex exponentials and logarithms; PyTorch's `torch.complex`, `torch.exp`, and `torch.log` handle these correctly and maintain the computation graph through complex arithmetic, ensuring gradient flow is preserved through the FFT.

### 5.5 Numerical Agreement

The autodiff Greeks are mathematically identical to the analytical Greeks --- both are exact derivatives of the same discretized Carr-Madan integral. In practice, differences arise only from floating-point rounding. The demo script verifies agreement to machine precision (differences on the order of $10^{-15}$). The Streamlit interface displays a side-by-side comparison with the absolute difference for each Greek, as well as an MSE summary table across strikes.

### 5.6 Put Adjustments

Autodiff naturally computes $\partial C_{\text{call}} / \partial(\cdot)$ from the call pricing graph. For puts, the same put-call parity adjustments described in Section 4.8 are applied to the autodiff results.

---

## 6. Calibration

### 6.1 Objective

Given a set of observed market option prices, the calibration routine finds the VG parameters $(\sigma, \nu, \theta)$ that minimize the weighted sum of squared pricing errors:

$$\min_{\sigma,\,\nu,\,\theta} \;\sum_{i=1}^{n} w_i\,\bigl(V_{\text{model}}(K_i, T_i;\,\sigma,\nu,\theta) - V_{\text{market}}^{(i)}\bigr)^2$$

where $V_{\text{model}}$ is the FFT model price and $V_{\text{market}}^{(i)}$ is the observed market price for strike $K_i$, expiry $T_i$, and option type (call or put).

### 6.2 Parameter Space and Constraints

| Parameter | Default bounds | Constraint |
|-----------|---------------|------------|
| $\sigma$ | $[0.01,\; 2.0]$ | Strictly positive |
| $\nu$ | $[0.001,\; 5.0]$ | Strictly positive |
| $\theta$ | $[-1.0,\; 1.0]$ | Unconstrained in sign |

Additionally, the **martingale condition** $1 - \theta\nu - \tfrac{1}{2}\sigma^2\nu > 0$ is enforced as a hard constraint. Any parameter combination violating it returns a penalty value of $10^{12}$.

### 6.3 Two-Phase Optimization

The optimization proceeds in two phases:

1. **Global search (optional):** `scipy.optimize.differential_evolution` with the given bounds, seed 42, up to 200 iterations, and tolerance $10^{-10}$. Differential evolution is a population-based stochastic optimizer that is effective at escaping local minima in the non-convex VG calibration landscape.

2. **Local refinement:** `scipy.optimize.minimize` with method `L-BFGS-B` (limited-memory BFGS with box constraints), starting from the global optimizer's solution (or the user-supplied initial guess if the global phase is skipped). Up to 500 iterations with function tolerance $10^{-14}$.

### 6.4 Input Format

The calibration function accepts parallel arrays:

- `K_market`: array of strikes.
- `T_market`: array of expiries (one per option, allowing mixed maturities).
- `prices_market`: array of observed market prices.
- `option_types`: list of `"call"` or `"put"` strings, one per option.
- `weights` (optional): array of per-option weights (default: equal weights of 1).

The default initial guess is $(\sigma_0, \nu_0, \theta_0) = (0.2, 0.5, -0.1)$.

---

## 7. Streamlit Application

The interactive dashboard (`app.py`) is organized into four tabs. All tabs share a common sidebar for specifying the VG model parameters ($S$, $r$, $q$, $\sigma$, $\nu$, $\theta$), with real-time enforcement of the martingale condition.

### Tab 1: Single-Point Pricer

Compute the price and all Greeks for a single option contract.

- **Inputs:** Strike $K$, time to expiry $T$, option type (call/put).
- **Outputs:**
  - Call and put prices displayed as metrics, along with the put-call parity error.
  - Analytical Greeks (FFT) displayed in a table on the left.
  - Autodiff Greeks (PyTorch) displayed in a table on the right, with the absolute difference from the analytical values shown in a third column.
  - If PyTorch is not installed, a notice is displayed in place of the autodiff column.

### Tab 2: Price Curves & Greeks

Sweep across a range of moneyness values $S/K$ and plot prices and all seven Greeks.

- **Inputs:** Expiry $T$, option type, number of grid points, min/max moneyness.
- **Outputs:**
  - An MSE comparison table (analytical vs. autodiff) across all Greeks and all grid points.
  - A price-vs-moneyness line chart with analytical and autodiff traces overlaid.
  - A 4x2 subplot grid with each Greek plotted against moneyness, both methods overlaid.
  - A 3D call price surface (moneyness vs. expiry vs. price) using `plotly.graph_objects.Surface`.

### Tab 3: Greeks (Custom)

Flexible Greek plotting with user-selectable computation method.

- **Inputs:** Expiry, option type, method selection (Analytical, Autodiff, or Both), moneyness range, grid points.
- **Outputs:** A 4x2 subplot grid of all seven Greeks plotted against moneyness, using only the selected method(s).

### Tab 4: Calibration

Calibrate VG parameters to market data uploaded from an Excel file.

- **Input format:** An `.xlsx` file with columns `K` (strike), `T` (expiry), `price` (market price), `type` (call/put). An optional `r` column overrides the sidebar risk-free rate.
- **Settings:** Spot price and dividend yield for calibration, option to enable the global optimizer (differential evolution).
- **Outputs:**
  - Calibrated parameters ($\sigma$, $\nu$, $\theta$) displayed as metrics.
  - SSE (sum of squared errors) and convergence status.
  - A results table with columns: moneyness, strike, expiry, type, market price, model price, residual.
  - A bar chart of calibration residuals (market minus model), color-coded by sign.
  - A scatter plot of model price vs. market price with a 45-degree perfect-fit reference line.
  - A smile overlay plot grouping options by expiry, showing market prices (markers) and model prices (lines) against moneyness.

---

## 8. Project Structure

```
Variance Gamma/
|-- variance_gamma.py    Core pricing engine: VarianceGammaModel class with
|                        FFT pricing, analytical Greeks, autodiff Greeks,
|                        and calibration.
|
|-- app.py               Streamlit web application with four tabs for
|                        interactive pricing, Greek visualization, and
|                        calibration.
|
|-- requirements.txt     Python package dependencies with minimum versions.
|
|-- README.md            This documentation file.
|
|-- docs/                Per-function documentation (25 .txt files, one per method):
|     |-- 01_init.txt  ...  13_theta_greek.txt   (core pricing & standard Greeks)
|     |-- 14_dphi_dsigma.txt ... 20_sensitivity_nu.txt  (vega, rho, VG sensitivities)
|     |-- 21_greeks.txt ... 25_calibrate.txt     (convenience, PyTorch autodiff, calibration)
```

---

## 9. Installation & Usage

### Prerequisites

- Python 3.9 or later.
- (Optional) A CUDA-capable GPU for PyTorch acceleration (not required; CPU is sufficient).

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install numpy scipy torch streamlit plotly pandas openpyxl
```

### Run the Pricing Engine Standalone

```bash
python variance_gamma.py
```

This executes a built-in demo that prices a European call and put, computes all analytical and autodiff Greeks, performs finite-difference sanity checks, and runs a calibration round-trip test.

### Launch the Streamlit Application

```bash
streamlit run app.py
```

This opens the interactive dashboard in your default web browser (typically at `http://localhost:8501`).

---

## 10. Variables Reference

### 10.1 Model Parameters

| Variable | Symbol | Type | Description |
|----------|--------|------|-------------|
| `S` | $S$ | `float` | Spot (current) price of the underlying asset |
| `r` | $r$ | `float` | Continuously compounded risk-free interest rate |
| `q` | $q$ | `float` | Continuous dividend yield |
| `sigma` | $\sigma$ | `float` | Volatility of the Brownian motion component ($\sigma > 0$) |
| `nu` | $\nu$ | `float` | Variance rate of the gamma subordinator ($\nu > 0$); controls kurtosis |
| `theta` / `theta_vg` | $\theta$ | `float` | Drift of the Brownian motion component; controls skewness of returns |

### 10.2 Option Contract Parameters

| Variable | Symbol | Type | Description |
|----------|--------|------|-------------|
| `K` | $K$ | `float` or `ndarray` | Strike price(s) |
| `T` | $T$ | `float` | Time to expiry in years |
| `option_type` | --- | `str` | `"call"` or `"put"` |

### 10.3 FFT Configuration

| Variable | Symbol | Type | Default | Description |
|----------|--------|------|---------|-------------|
| `N` | $N$ | `int` | 4096 | Number of FFT grid points (must be a power of 2 for efficiency) |
| `alpha` | $\alpha$ | `float` | 1.5 | Carr-Madan dampening factor; ensures integrability of the call transform |
| `eta` | $\eta$ | `float` | 0.25 | Spacing in the frequency domain ($u$-grid) |
| `lam` | $\lambda$ | `float` | derived | Log-strike spacing; $\lambda = 2\pi/(N\eta)$ |
| `b` | $b$ | `float` | derived | Half-width of the log-strike grid; $b = N\lambda/2$ |

### 10.4 Intermediate Quantities

| Variable | Symbol / Expression | Type | Description |
|----------|---------------------|------|-------------|
| `omega` / `w` | $\omega = \frac{1}{\nu}\ln(1 - \theta\nu - \frac{1}{2}\sigma^2\nu)$ | `float` | Martingale correction (convexity adjustment) |
| `u` | $u_j = j\eta$ | `ndarray` (N,) | Frequency-domain grid |
| `v` | $v_j = u_j - (\alpha+1)i$ | `ndarray` (N,) complex | Shifted frequency for Carr-Madan |
| `phi` | $\phi(v, T)$ | `ndarray` (N,) complex | Characteristic function evaluated at $v$ |
| `denom` | $D(u) = \alpha^2+\alpha-u^2+i(2\alpha+1)u$ | `ndarray` (N,) complex | Carr-Madan denominator |
| `psi` | $\Psi(u) = e^{-rT}\phi(v,T)/D(u)$ | `ndarray` (N,) complex | Modified characteristic function (Carr-Madan integrand) |
| `sw` / `w_j` | $w_j$ | `ndarray` (N,) | Simpson's rule quadrature weights |
| `x` | $x_j = e^{ibu_j}\Psi(u_j)\eta w_j$ | `ndarray` (N,) complex | FFT input vector |
| `fft_result` / `Y` | $Y = \text{FFT}(x)$ | `ndarray` (N,) complex | Raw FFT output |
| `k` / `k_grid` | $k_m = -b + \lambda m$ | `ndarray` (N,) | Log-strike grid |
| `call_prices` | $C(k_m) = e^{-\alpha k_m}\text{Re}(Y_m)/\pi$ | `ndarray` (N,) | Call prices on the log-strike grid |
| `log_K` | $\ln K$ | `float` or `ndarray` | Log of the target strike(s) for interpolation |

### 10.5 Greek Derivative Intermediates

| Variable | Expression | Description |
|----------|------------|-------------|
| `A` | $1 - \theta\nu - \frac{1}{2}\sigma^2\nu$ | Argument of $\ln$ in the martingale correction |
| `B` / `B(v)` | $1 - iv\theta\nu + \frac{1}{2}\sigma^2\nu v^2$ | Inner argument of the VG characteristic exponent |
| `domega` | $\partial\omega/\partial(\cdot)$ | Derivative of the martingale correction w.r.t. a parameter |
| `dvg` | $\partial(\text{VG exponent})/\partial(\cdot)$ | Derivative of the VG log-characteristic exponent |
| `dphi` | $\partial\phi/\partial(\cdot)$ | Derivative of the full characteristic function |
| `dB` | $\partial B/\partial\nu$ | Derivative of $B$ w.r.t. $\nu$: $-iv\theta + \frac{1}{2}\sigma^2 v^2$ |

### 10.6 Calibration Variables

| Variable | Type | Description |
|----------|------|-------------|
| `K_market` | `ndarray` | Array of observed market strikes |
| `T_market` | `ndarray` | Array of observed market expiries |
| `prices_market` | `ndarray` | Array of observed market option prices |
| `option_types` | `list[str]` | List of `"call"`/`"put"` per observed option |
| `weights` | `ndarray` | Per-option weights in the objective function |
| `x0` | `list[float]` | Initial guess $[\sigma_0, \nu_0, \theta_0]$; default $[0.2, 0.5, -0.1]$ |
| `bounds` | `list[tuple]` | Parameter bounds for the optimizer |
| `use_global` | `bool` | Whether to run differential evolution before L-BFGS-B |
| `objective` | `callable` | The weighted SSE loss function |
| `res` | `OptimizeResult` | Scipy optimization result object |
| `cal_model` | `VarianceGammaModel` | Model instance with calibrated parameters |

### 10.7 PyTorch Autodiff Variables

| Variable | Type | Description |
|----------|------|-------------|
| `S_t`, `r_t`, `q_t`, `sigma_t`, `nu_t`, `theta_t`, `T_t` | `torch.Tensor` (scalar, float64) | Differentiable tensors for each model parameter with `requires_grad=True` |
| `K_t` | `torch.Tensor` (scalar, float64) | Strike tensor (no gradient required) |
| `grads` | `tuple[torch.Tensor]` | Tuple of first-order partial derivatives from `torch.autograd.grad` |
| `delta_call` | `torch.Tensor` | $\partial C/\partial S$ with graph retained for second-order differentiation |
| `gamma_val` | `torch.Tensor` | $\partial^2 C/\partial S^2$ obtained by differentiating `delta_call` w.r.t. `S_t` |

---

## References

- Madan, D. B., Carr, P., & Chang, E. C. (1998). The Variance Gamma Process and Option Pricing. *Review of Finance*, 2(1), 79--105.
- Carr, P., & Madan, D. B. (1999). Option Valuation Using the Fast Fourier Transform. *Journal of Computational Finance*, 2(4), 61--73.
