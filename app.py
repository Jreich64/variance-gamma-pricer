import time

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from variance_gamma import VarianceGammaModel, _HAS_TORCH, _TORCH_DEVICE, _TORCH_GPU_NAME
import vg_mpmath

st.set_page_config(page_title="Variance Gamma Pricer", layout="wide")
st.title("Variance Gamma Option Pricer")

# ── Hardware info banner ──────────────────────────────────────────────────
if _HAS_TORCH:
    import torch
    _dev_label = f"**GPU:** {_TORCH_GPU_NAME}" if _TORCH_GPU_NAME else f"**CPU only** (CUDA not available)"
    st.caption(f"PyTorch {torch.__version__}  ·  Device: {_TORCH_DEVICE}  ·  {_dev_label}")
else:
    st.caption("PyTorch not installed — autodiff Greeks disabled")

# ── Sidebar: model parameters ──────────────────────────────────────────────
st.sidebar.header("Model Parameters")
S = st.sidebar.number_input("Spot Price (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.005, format="%.4f")
q = st.sidebar.number_input("Dividend Yield (q)", value=0.02, step=0.005, format="%.4f")

st.sidebar.header("VG Parameters")
sigma = st.sidebar.number_input("sigma (BM vol)", value=0.2, min_value=0.001, step=0.01, format="%.4f")
nu = st.sidebar.number_input("nu (variance rate)", value=0.5, min_value=0.001, step=0.05, format="%.4f")
theta_vg = st.sidebar.number_input("theta (BM drift / skew)", value=-0.1, step=0.01, format="%.4f")

# Validate martingale condition
mart_cond = 1.0 - theta_vg * nu - 0.5 * sigma ** 2 * nu
if mart_cond <= 0:
    st.sidebar.error("Martingale condition violated: 1 - theta*nu - sigma^2*nu/2 must be > 0")
    st.stop()

model = VarianceGammaModel(S, r, q, sigma, nu, theta_vg)

# ── Sidebar: FFT / Numerical Settings ─────────────────────────────────────
st.sidebar.header("FFT / Numerical Settings")
alpha_damp = st.sidebar.number_input(
    "Damping parameter (alpha)", value=1.5, min_value=0.01, max_value=50.0,
    step=0.1, format="%.2f",
    help="Carr-Madan dampening factor ensuring Fourier integrability of the call price."
)
fft_N = int(st.sidebar.selectbox("FFT grid size (N)", [1024, 2048, 4096, 8192, 16384], index=2))
fft_eta = st.sidebar.number_input("Frequency spacing (eta)", value=0.25, min_value=0.01, max_value=1.0, step=0.05, format="%.4f")
cos_N = int(st.sidebar.number_input("COS expansion terms", value=256, min_value=32, max_value=4096, step=32))
cos_L = st.sidebar.number_input("COS truncation range (L)", value=10.0, min_value=2.0, max_value=50.0, step=1.0, format="%.1f")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Damping parameter (alpha):** The Carr-Madan FFT method multiplies the "
    "call price by $e^{\\alpha k}$ (where $k = \\ln K$) to force the modified "
    "call price into $L^1$ space, making the Fourier integral converge. "
    "A value of $\\alpha = 1.5$ is the standard recommendation. Too small "
    "causes oscillation; too large amplifies numerical noise at extreme strikes."
)

PLOT_MARGIN = dict(t=50, b=20)

_dl_counter = 0


def plot_3d_with_download(fig, filename, key_prefix=""):
    """Display a 3D plotly figure and offer an HTML download button."""
    global _dl_counter
    _dl_counter += 1
    st.plotly_chart(fig, width="stretch")
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        label=f"Download {filename}",
        data=html_bytes,
        file_name=filename,
        mime="text/html",
        key=f"dl_{key_prefix}_{_dl_counter}",
    )


def _sidebar_params_key():
    return (S, r, q, sigma, nu, theta_vg, alpha_damp, fft_N, fft_eta, cos_N, cos_L)


def _eta_text(label, step, total, t0):
    """Return progress-bar text with ETA."""
    frac = step / total
    elapsed = time.perf_counter() - t0
    if step > 0:
        remaining = elapsed / step * (total - step)
        if remaining >= 60:
            eta_str = f"{remaining / 60:.1f} min"
        else:
            eta_str = f"{remaining:.0f}s"
    else:
        eta_str = "..."
    return f"{label} — {step}/{total} ({frac:.0%}) · ETA {eta_str}"


# ── Tabs ───────────────────────────────────────────────────────────────────
tab_single, tab_curves, tab_greeks, tab_3d, tab_mse, tab_calib, tab_explain = st.tabs([
    "Single-Point Pricer",
    "Price Curves & Greeks",
    "Greeks (Custom)",
    "3D Surfaces by Method",
    "Method Comparison (MSE)",
    "Calibration",
    "Method Explanations",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Single-Point Pricer (all methods + timing)
# ═══════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Single-Point Option Value & Greeks")
    st.caption("All four pricing methods execute simultaneously for easy comparison.")

    with st.form("single_form"):
        c1, c2, c3, c4 = st.columns(4)
        K_single = c1.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
        T_single = c2.number_input("Time to Expiry (T, years)", value=0.5, min_value=0.01, step=0.05, format="%.4f")
        opt_type_single = c3.selectbox("Option Type", ["call", "put"])
        arb_dps_sp = int(c4.number_input("Arb. precision (dps)", value=30, min_value=10, max_value=200, step=5, key="sp_dps"))
        submitted_single = st.form_submit_button("Calculate")

    if submitted_single:
        fft_kw = dict(N=fft_N, alpha=alpha_damp, eta=fft_eta)

        # FFT
        t0 = time.perf_counter()
        price_fft = model.price(K_single, T_single, opt_type_single, **fft_kw)
        t_fft = time.perf_counter() - t0

        # FRFT
        t0 = time.perf_counter()
        price_frft = model.price_frft(K_single, T_single, opt_type_single, **fft_kw)
        t_frft = time.perf_counter() - t0

        # COS
        t0 = time.perf_counter()
        price_cos = model.price_cos(K_single, T_single, opt_type_single, N_cos=cos_N, L=cos_L)
        t_cos = time.perf_counter() - t0

        # Arbitrary Precision
        vg_mpmath.set_precision(arb_dps_sp)
        t0 = time.perf_counter()
        arb_call = float(vg_mpmath._price_only(S, r, q, sigma, theta_vg, nu, T_single, K_single))
        if opt_type_single == "put":
            arb_call = arb_call - S * np.exp(-q * T_single) + K_single * np.exp(-r * T_single)
        t_arb = time.perf_counter() - t0

        # Put-call parity check (FFT)
        call_fft = model.price(K_single, T_single, "call", **fft_kw)
        put_fft = model.price(K_single, T_single, "put", **fft_kw)
        parity_err = call_fft - put_fft - S * np.exp(-q * T_single) + K_single * np.exp(-r * T_single)

        st.markdown("#### Prices & Timing")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("FFT Price", f"{price_fft:.6f}", delta=f"{t_fft*1000:.1f} ms")
        mc2.metric("FRFT Price", f"{price_frft:.6f}", delta=f"{t_frft*1000:.1f} ms")
        mc3.metric("COS Price", f"{price_cos:.6f}", delta=f"{t_cos*1000:.1f} ms")
        mc4.metric(f"Arb. Precision ({arb_dps_sp} dps)", f"{arb_call:.6f}", delta=f"{t_arb*1000:.1f} ms")

        st.metric("Put-Call Parity Error (FFT)", f"{parity_err:.2e}")

        st.markdown("---")
        col_an, col_ad = st.columns(2)

        with col_an:
            st.markdown("**Analytical Greeks (FFT)**")
            g_an = model.greeks(K_single, T_single, opt_type_single, **fft_kw)
            df_an = pd.DataFrame({"Greek": list(g_an.keys()), "Value": [f"{v:.6f}" for v in g_an.values()]})
            st.dataframe(df_an, hide_index=True, width="stretch")

        with col_ad:
            if _HAS_TORCH:
                st.markdown("**Autodiff Greeks (PyTorch FFT)**")
                g_ad = model.greeks_ad(K_single, T_single, opt_type_single, **fft_kw)
                df_ad = pd.DataFrame({
                    "Greek": list(g_ad.keys()),
                    "Autodiff": [f"{v:.6f}" for v in g_ad.values()],
                    "|Diff|": [f"{abs(g_an[k] - g_ad[k]):.2e}" for k in g_an],
                })
                st.dataframe(df_ad, hide_index=True, width="stretch")

                st.markdown("---")
                st.markdown("**Autodiff Greeks — FRFT & COS (PyTorch)**")
                ad_col1, ad_col2 = st.columns(2)
                with ad_col1:
                    st.markdown("*FRFT*")
                    t0 = time.perf_counter()
                    g_ad_frft = model.greeks_ad_frft(K_single, T_single, opt_type_single, **fft_kw)
                    t_ad_frft = time.perf_counter() - t0
                    df_frft_ad = pd.DataFrame({
                        "Greek": list(g_ad_frft.keys()),
                        "Value": [f"{v:.6f}" for v in g_ad_frft.values()],
                    })
                    st.dataframe(df_frft_ad, hide_index=True, width="stretch")
                    st.caption(f"⏱ {t_ad_frft*1000:.1f} ms")
                with ad_col2:
                    st.markdown("*COS*")
                    t0 = time.perf_counter()
                    g_ad_cos = model.greeks_ad_cos(K_single, T_single, opt_type_single, N_cos=cos_N, L=cos_L)
                    t_ad_cos = time.perf_counter() - t0
                    df_cos_ad = pd.DataFrame({
                        "Greek": list(g_ad_cos.keys()),
                        "Value": [f"{v:.6f}" for v in g_ad_cos.values()],
                    })
                    st.dataframe(df_cos_ad, hide_index=True, width="stretch")
                    st.caption(f"⏱ {t_ad_cos*1000:.1f} ms")
            else:
                st.info("Install PyTorch for autodiff Greeks: `pip install torch`")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Price Curves & Greeks  (moneyness x-axis throughout)
# ═══════════════════════════════════════════════════════════════════════════
with tab_curves:
    st.subheader("Prices & Greeks vs Moneyness")
    st.caption("Moneyness is defined as S/K (spot / strike).  "
               "Values > 1 are in-the-money for calls, < 1 for puts.")

    with st.form("curves_form"):
        cc1, cc2, cc3 = st.columns(3)
        T_curve = cc1.number_input("Expiry (T)", value=0.5, min_value=0.01, step=0.05, format="%.4f", key="t_curve")
        opt_type_curve = cc2.selectbox("Option Type", ["call", "put"], key="ot_curve")
        n_pts = int(cc3.number_input("Grid points", value=60, min_value=10, max_value=300, step=10, key="npts_curve"))
        m_lo, m_hi = st.columns(2)
        moneyness_lo = m_lo.number_input("Min moneyness (S/K)", value=0.7, min_value=0.01, step=0.05, format="%.2f")
        moneyness_hi = m_hi.number_input("Max moneyness (S/K)", value=1.3, min_value=0.1, step=0.05, format="%.2f")
        t_lo, t_hi = st.columns(2)
        T_surf_lo = t_lo.number_input("3D Min expiry (T)", value=0.1, min_value=0.01, step=0.05, format="%.2f", key="t_surf_lo")
        T_surf_hi = t_hi.number_input("3D Max expiry (T)", value=2.0, min_value=0.05, step=0.1, format="%.2f", key="t_surf_hi")
        submitted_curves = st.form_submit_button("Compute Curves")

    # Compute on form submit, cache in session_state
    if submitted_curves:
        moneyness = np.linspace(moneyness_lo, moneyness_hi, n_pts)
        K_arr = S / moneyness

        all_greek_names = ["price", "delta", "gamma", "theta", "vega", "rho", "d_theta_param", "d_nu"]

        # ── Compute analytical Greeks with timing ─────────────────────
        an_data = {gn: [] for gn in all_greek_names}
        progress = st.progress(0, text="Analytical Greeks...")
        t0_an = time.perf_counter()
        for idx, k in enumerate(K_arr):
            g = model.greeks(k, T_curve, opt_type_curve, N=fft_N, alpha=alpha_damp, eta=fft_eta)
            for gn in an_data:
                an_data[gn].append(g[gn])
            progress.progress((idx + 1) / len(K_arr),
                              text=_eta_text("Analytical Greeks", idx + 1, len(K_arr), t0_an))
        t_analytical = time.perf_counter() - t0_an
        progress.empty()
        for gn in an_data:
            an_data[gn] = np.array(an_data[gn])

        # ── Compute autodiff Greeks with timing ───────────────────────
        ad_data = None
        t_autodiff = None
        if _HAS_TORCH:
            ad_data = {gn: [] for gn in all_greek_names}
            progress2 = st.progress(0, text="Autodiff Greeks...")
            t0_ad = time.perf_counter()
            for idx, k in enumerate(K_arr):
                g = model.greeks_ad(k, T_curve, opt_type_curve, N=fft_N, alpha=alpha_damp, eta=fft_eta)
                for gn in ad_data:
                    ad_data[gn].append(g[gn])
                progress2.progress((idx + 1) / len(K_arr),
                                   text=_eta_text("Autodiff Greeks", idx + 1, len(K_arr), t0_ad))
            t_autodiff = time.perf_counter() - t0_ad
            progress2.empty()
            for gn in ad_data:
                ad_data[gn] = np.array(ad_data[gn])

        # ── 3D Surfaces ───────────────────────────────────────────────
        T_range = np.linspace(T_surf_lo, T_surf_hi, n_pts)
        m_range = np.linspace(moneyness_lo, moneyness_hi, n_pts)
        K_surf = S / m_range

        surface_names = ["price", "delta", "gamma", "theta", "vega", "rho", "d_theta_param", "d_nu"]
        surface_labels = {"price": "Call Price", "delta": "Delta", "gamma": "Gamma",
                          "theta": "Theta (dC/dT)", "vega": "Vega", "rho": "Rho",
                          "d_theta_param": "d/d(theta_VG)", "d_nu": "d/d(nu)"}

        progress_3d = st.progress(0, text="Computing analytical 3D surfaces...")
        t0_3d = time.perf_counter()
        an_surfaces = {gn: np.zeros((n_pts, n_pts)) for gn in surface_names}
        for i, t in enumerate(T_range):
            for j, k in enumerate(K_surf):
                g = model.greeks(k, t, "call", N=fft_N, alpha=alpha_damp, eta=fft_eta)
                for gn in surface_names:
                    an_surfaces[gn][i, j] = g[gn]
            progress_3d.progress((i + 1) / n_pts,
                                 text=_eta_text("Analytical 3D", i + 1, n_pts, t0_3d))
        progress_3d.empty()

        ad_surfaces = None
        if _HAS_TORCH:
            progress_3d_ad = st.progress(0, text="Computing autodiff 3D surfaces...")
            t0_3d_ad = time.perf_counter()
            ad_surfaces = {gn: np.zeros((n_pts, n_pts)) for gn in surface_names}
            for i, t in enumerate(T_range):
                for j, k in enumerate(K_surf):
                    g = model.greeks_ad(k, t, "call", N=fft_N, alpha=alpha_damp, eta=fft_eta)
                    for gn in surface_names:
                        ad_surfaces[gn][i, j] = g[gn]
                progress_3d_ad.progress((i + 1) / n_pts,
                                        text=_eta_text("Autodiff 3D", i + 1, n_pts, t0_3d_ad))
            progress_3d_ad.empty()

        # Store everything in session_state
        st.session_state["tab2"] = {
            "moneyness": moneyness,
            "an_data": an_data,
            "ad_data": ad_data,
            "t_analytical": t_analytical,
            "t_autodiff": t_autodiff,
            "T_curve": T_curve,
            "opt_type_curve": opt_type_curve,
            "an_surfaces": an_surfaces,
            "ad_surfaces": ad_surfaces,
            "T_range": T_range,
            "m_range": m_range,
            "surface_names": surface_names,
            "surface_labels": surface_labels,
            "moneyness_lo": moneyness_lo,
            "moneyness_hi": moneyness_hi,
            "params_key": _sidebar_params_key(),
        }

    # Display results from session_state (survives download-button reruns)
    if "tab2" in st.session_state and st.session_state["tab2"]["params_key"] == _sidebar_params_key():
        _t2 = st.session_state["tab2"]
        moneyness = _t2["moneyness"]
        an_data = _t2["an_data"]
        ad_data = _t2["ad_data"]
        t_analytical = _t2["t_analytical"]
        t_autodiff = _t2["t_autodiff"]
        T_curve_disp = _t2["T_curve"]
        opt_type_curve_disp = _t2["opt_type_curve"]
        an_surfaces = _t2["an_surfaces"]
        ad_surfaces = _t2["ad_surfaces"]
        T_range = _t2["T_range"]
        m_range = _t2["m_range"]
        surface_names = _t2["surface_names"]
        surface_labels = _t2["surface_labels"]

        all_greek_names = ["price", "delta", "gamma", "theta", "vega", "rho", "d_theta_param", "d_nu"]

        # ── Timing metrics ────────────────────────────────────────────
        st.markdown("#### Computation Time")
        tc1, tc2 = st.columns(2)
        tc1.metric("Analytical Greeks (FFT)", f"{t_analytical:.3f} s")
        if t_autodiff is not None:
            tc2.metric("Autodiff Greeks (PyTorch)", f"{t_autodiff:.3f} s")
        else:
            tc2.info("PyTorch not installed")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── MSE table: analytical vs autodiff ─────────────────────────
        if ad_data is not None:
            st.markdown("#### MSE: Analytical vs Autodiff Greeks")
            mse_rows = []
            for gn in all_greek_names:
                mse = float(np.mean((an_data[gn] - ad_data[gn]) ** 2))
                mse_rows.append({"Greek": gn, "MSE": f"{mse:.4e}"})
            st.dataframe(pd.DataFrame(mse_rows), hide_index=True, width="stretch")
            st.markdown("")

        # ── Price curve ───────────────────────────────────────────────
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=moneyness, y=an_data["price"],
                                       name="Analytical", line=dict(color="#2196F3")))
        if ad_data is not None:
            fig_price.add_trace(go.Scatter(x=moneyness, y=ad_data["price"],
                                           name="Autodiff", line=dict(color="#F44336", dash="dash")))
        fig_price.update_layout(
            title=f"{opt_type_curve_disp.title()} Price vs Moneyness (T={T_curve_disp})",
            xaxis_title="Moneyness (S/K)", yaxis_title="Price",
            height=420, margin=PLOT_MARGIN,
        )
        st.plotly_chart(fig_price, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Greek curves: one plot per Greek ──────────────────────────
        greek_plot_names = ["delta", "gamma", "theta", "vega", "rho", "d_theta_param", "d_nu"]
        nice = {"delta": "Delta", "gamma": "Gamma", "theta": "Theta (dC/dT)",
                "vega": "Vega", "rho": "Rho", "d_theta_param": "d/d(theta_VG)", "d_nu": "d/d(nu)"}

        fig_greeks = make_subplots(
            rows=4, cols=2,
            subplot_titles=[nice[g] for g in greek_plot_names] + [""],
            vertical_spacing=0.08, horizontal_spacing=0.08,
        )
        for i, gn in enumerate(greek_plot_names):
            row = i // 2 + 1
            col = i % 2 + 1
            fig_greeks.add_trace(
                go.Scatter(x=moneyness, y=an_data[gn], name="Analytical",
                           line=dict(color="#2196F3"), legendgroup="an", showlegend=(i == 0)),
                row=row, col=col,
            )
            if ad_data is not None:
                fig_greeks.add_trace(
                    go.Scatter(x=moneyness, y=ad_data[gn], name="Autodiff",
                               line=dict(color="#F44336", dash="dash"), legendgroup="ad", showlegend=(i == 0)),
                    row=row, col=col,
                )
            fig_greeks.update_xaxes(title_text="S/K", row=row, col=col)

        fig_greeks.update_layout(
            height=1200,
            title_text=f"Greeks vs Moneyness  ({opt_type_curve_disp.title()}, T={T_curve_disp})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=80, b=20),
        )
        st.plotly_chart(fig_greeks, width="stretch")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ── 3D Surfaces ───────────────────────────────────────────────
        st.markdown("#### 3D Surfaces vs Moneyness & Time to Expiry")

        st.markdown("##### Analytical (FFT) Surfaces")
        for gn in surface_names:
            fig_s = go.Figure(data=[go.Surface(
                x=m_range, y=T_range, z=an_surfaces[gn], colorscale="Viridis"
            )])
            fig_s.update_layout(
                title=f"Analytical {surface_labels[gn]}",
                scene=dict(xaxis_title="Moneyness (S/K)", yaxis_title="Expiry (T)",
                           zaxis_title=surface_labels[gn]),
                height=500, margin=dict(t=50, b=10),
            )
            plot_3d_with_download(fig_s, f"analytical_{gn}_surface.html", key_prefix=f"an_{gn}")
            st.markdown("<br>", unsafe_allow_html=True)

        if ad_surfaces is not None:
            st.markdown("##### Autodiff (PyTorch) Surfaces")
            for gn in surface_names:
                fig_s = go.Figure(data=[go.Surface(
                    x=m_range, y=T_range, z=ad_surfaces[gn], colorscale="Inferno"
                )])
                fig_s.update_layout(
                    title=f"Autodiff {surface_labels[gn]}",
                    scene=dict(xaxis_title="Moneyness (S/K)", yaxis_title="Expiry (T)",
                               zaxis_title=surface_labels[gn]),
                    height=500, margin=dict(t=50, b=10),
                )
                plot_3d_with_download(fig_s, f"autodiff_{gn}_surface.html", key_prefix=f"ad_{gn}")
                st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Greeks (Custom)
# ═══════════════════════════════════════════════════════════════════════════
with tab_greeks:
    st.subheader("Greeks vs Moneyness — Analytical & Autodiff Overlay")
    if not _HAS_TORCH:
        st.warning("PyTorch is not installed. Only analytical Greeks will be shown.")

    with st.form("greeks_form"):
        gc1, gc2 = st.columns(2)
        T_greek = gc1.number_input("Expiry (T)", value=0.5, min_value=0.01, step=0.05, format="%.4f", key="t_grk")
        opt_type_grk = gc2.selectbox("Option Type", ["call", "put"], key="ot_grk")

        mg1, mg2, mg3 = st.columns(3)
        m_lo_grk = mg1.number_input("Min moneyness (S/K)", value=0.7, min_value=0.01, step=0.05, format="%.2f", key="mlo_grk")
        m_hi_grk = mg2.number_input("Max moneyness (S/K)", value=1.3, min_value=0.1, step=0.05, format="%.2f", key="mhi_grk")
        n_grk = int(mg3.number_input("Grid points", value=60, min_value=10, max_value=300, step=10, key="ng_grk"))
        submitted_greeks = st.form_submit_button("Compute Greeks")

    if submitted_greeks:
        moneyness_grk = np.linspace(m_lo_grk, m_hi_grk, n_grk)
        K_arr = S / moneyness_grk

        greek_names = ["delta", "gamma", "theta", "vega", "rho", "d_theta_param", "d_nu"]

        # Always compute analytical
        data_an = {gn: [] for gn in greek_names}
        progress_an = st.progress(0, text="Computing analytical Greeks...")
        t0_an = time.perf_counter()
        for idx, k in enumerate(K_arr):
            g = model.greeks(k, T_greek, opt_type_grk, N=fft_N, alpha=alpha_damp, eta=fft_eta)
            for gn in greek_names:
                data_an[gn].append(g[gn])
            progress_an.progress((idx + 1) / len(K_arr),
                                 text=_eta_text("Analytical Greeks", idx + 1, len(K_arr), t0_an))
        t_an = time.perf_counter() - t0_an
        progress_an.empty()

        # Always compute autodiff if available
        data_ad = None
        t_ad = None
        if _HAS_TORCH:
            data_ad = {gn: [] for gn in greek_names}
            progress_ad = st.progress(0, text="Computing autodiff Greeks...")
            t0_ad = time.perf_counter()
            for idx, k in enumerate(K_arr):
                g = model.greeks_ad(k, T_greek, opt_type_grk, N=fft_N, alpha=alpha_damp, eta=fft_eta)
                for gn in greek_names:
                    data_ad[gn].append(g[gn])
                progress_ad.progress((idx + 1) / len(K_arr),
                                     text=_eta_text("Autodiff Greeks", idx + 1, len(K_arr), t0_ad))
            t_ad = time.perf_counter() - t0_ad
            progress_ad.empty()

        st.session_state["tab3"] = {
            "moneyness_grk": moneyness_grk,
            "data_an": data_an,
            "data_ad": data_ad,
            "t_an": t_an,
            "t_ad": t_ad,
            "T_greek": T_greek,
            "opt_type_grk": opt_type_grk,
            "params_key": _sidebar_params_key(),
        }

    if "tab3" in st.session_state and st.session_state["tab3"]["params_key"] == _sidebar_params_key():
        _t3 = st.session_state["tab3"]
        moneyness_grk = _t3["moneyness_grk"]
        data_an = _t3["data_an"]
        data_ad = _t3["data_ad"]
        t_an = _t3["t_an"]
        t_ad = _t3["t_ad"]
        T_greek_disp = _t3["T_greek"]
        opt_type_grk_disp = _t3["opt_type_grk"]

        greek_names = ["delta", "gamma", "theta", "vega", "rho", "d_theta_param", "d_nu"]
        nice_names = {"delta": "Delta", "gamma": "Gamma", "theta": "Theta (dC/dT)",
                      "vega": "Vega", "rho": "Rho", "d_theta_param": "d/d(theta_VG)",
                      "d_nu": "d/d(nu)"}

        tc1, tc2 = st.columns(2)
        tc1.metric("Analytical (FFT)", f"{t_an:.3f} s")
        if t_ad is not None:
            tc2.metric("Autodiff (PyTorch)", f"{t_ad:.3f} s")

        st.markdown("<br>", unsafe_allow_html=True)

        fig = make_subplots(rows=4, cols=2, subplot_titles=[nice_names[g] for g in greek_names] + [""],
                            vertical_spacing=0.08, horizontal_spacing=0.08)

        for i, gn in enumerate(greek_names):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(go.Scatter(x=moneyness_grk, y=data_an[gn], name="Analytical (FFT)",
                                     line=dict(color="#2196F3"), legendgroup="an", showlegend=(i == 0)),
                          row=row, col=col)
            if data_ad is not None:
                fig.add_trace(go.Scatter(x=moneyness_grk, y=data_ad[gn], name="Autodiff (PyTorch)",
                                         line=dict(color="#F44336", dash="dash"), legendgroup="ad", showlegend=(i == 0)),
                              row=row, col=col)
            fig.update_xaxes(title_text="S/K", row=row, col=col)

        fig.update_layout(
            height=1200, title_text=f"Greeks vs Moneyness  ({opt_type_grk_disp.title()}, T={T_greek_disp})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        itemsizing="constant"),
            margin=dict(t=80, b=20),
        )
        st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — 3D Surfaces by Method (FFT, FRFT, COS, Arb Precision)
# ═══════════════════════════════════════════════════════════════════════════
with tab_3d:
    st.subheader("3D Price Surfaces by Method")
    st.caption(
        "Computes a price surface (moneyness × time to expiry) for each method "
        "individually, with timing. Arbitrary Precision is the slowest — keep its "
        "grid small (≤ 15)."
    )

    with st.form("surf3d_form"):
        sc1, sc2, sc3 = st.columns(3)
        surf_n_m = int(sc1.number_input("Moneyness grid points", value=20, min_value=3, max_value=100, step=1, key="surf_nm"))
        surf_n_t = int(sc2.number_input("Expiry grid points", value=20, min_value=3, max_value=100, step=1, key="surf_nt"))
        surf_opt = sc3.selectbox("Option Type", ["call", "put"], key="surf_opt")

        sm1, sm2 = st.columns(2)
        surf_m_lo = sm1.number_input("Min moneyness (S/K)", value=0.7, min_value=0.01, step=0.05, format="%.2f", key="surf_mlo")
        surf_m_hi = sm2.number_input("Max moneyness (S/K)", value=1.3, min_value=0.1, step=0.05, format="%.2f", key="surf_mhi")

        st1, st2 = st.columns(2)
        surf_T_lo = st1.number_input("Min expiry (T)", value=0.1, min_value=0.01, step=0.05, format="%.2f", key="surf_tlo")
        surf_T_hi = st2.number_input("Max expiry (T)", value=2.0, min_value=0.05, step=0.1, format="%.2f", key="surf_thi")

        sp1, sp2 = st.columns(2)
        surf_dps = int(sp1.number_input("Arb. precision (dps)", value=30, min_value=10, max_value=200, step=5, key="surf_dps"))
        surf_arb_pts = int(sp2.number_input("Arb. precision grid points (may differ)", value=8, min_value=3, max_value=50, step=1, key="surf_arb_pts"))

        submitted_3d = st.form_submit_button("Compute All Surfaces")

    if submitted_3d:
        m_vals_3d = np.linspace(surf_m_lo, surf_m_hi, surf_n_m)
        T_vals_3d = np.linspace(surf_T_lo, surf_T_hi, surf_n_t)
        K_vals_3d = S / m_vals_3d
        fft_kw = dict(N=fft_N, alpha=alpha_damp, eta=fft_eta)

        total = surf_n_m * surf_n_t

        # ── FFT ──────────────────────────────────────────────────────
        fft_surf = np.zeros((surf_n_t, surf_n_m))
        prog = st.progress(0, text="Computing FFT surface...")
        t0 = time.perf_counter()
        step = 0
        for i, tv in enumerate(T_vals_3d):
            for j, kv in enumerate(K_vals_3d):
                fft_surf[i, j] = model.price(kv, float(tv), surf_opt, **fft_kw)
                step += 1
                prog.progress(step / total, text=_eta_text("FFT surface", step, total, t0))
        t_fft_surf = time.perf_counter() - t0
        prog.empty()

        # ── FRFT ─────────────────────────────────────────────────────
        frft_surf = np.zeros((surf_n_t, surf_n_m))
        prog = st.progress(0, text="Computing FRFT surface...")
        t0 = time.perf_counter()
        step = 0
        for i, tv in enumerate(T_vals_3d):
            for j, kv in enumerate(K_vals_3d):
                frft_surf[i, j] = model.price_frft(kv, float(tv), surf_opt, **fft_kw)
                step += 1
                prog.progress(step / total, text=_eta_text("FRFT surface", step, total, t0))
        t_frft_surf = time.perf_counter() - t0
        prog.empty()

        # ── COS ──────────────────────────────────────────────────────
        cos_surf = np.zeros((surf_n_t, surf_n_m))
        prog = st.progress(0, text="Computing COS surface...")
        t0 = time.perf_counter()
        step = 0
        for i, tv in enumerate(T_vals_3d):
            for j, kv in enumerate(K_vals_3d):
                cos_surf[i, j] = model.price_cos(kv, float(tv), surf_opt, N_cos=cos_N, L=cos_L)
                step += 1
                prog.progress(step / total, text=_eta_text("COS surface", step, total, t0))
        t_cos_surf = time.perf_counter() - t0
        prog.empty()

        # ── Arbitrary Precision ──────────────────────────────────────
        vg_mpmath.set_precision(surf_dps)
        m_vals_arb = np.linspace(surf_m_lo, surf_m_hi, surf_arb_pts)
        T_vals_arb = np.linspace(surf_T_lo, surf_T_hi, surf_arb_pts)
        K_vals_arb = S / m_vals_arb
        total_arb = surf_arb_pts * surf_arb_pts
        arb_surf = np.zeros((surf_arb_pts, surf_arb_pts))
        prog = st.progress(0, text=f"Computing Arb. Precision surface ({surf_dps} dps)...")
        t0 = time.perf_counter()
        step = 0
        for i, tv in enumerate(T_vals_arb):
            for j, kv in enumerate(K_vals_arb):
                arb_call = float(vg_mpmath._price_only(
                    S, r, q, sigma, theta_vg, nu, float(tv), float(kv)
                ))
                if surf_opt == "put":
                    arb_call = arb_call - S * np.exp(-q * float(tv)) + float(kv) * np.exp(-r * float(tv))
                arb_surf[i, j] = arb_call
                step += 1
                prog.progress(step / total_arb, text=_eta_text(f"Arb. Precision ({surf_dps} dps)", step, total_arb, t0))
        t_arb_surf = time.perf_counter() - t0
        prog.empty()

        st.session_state["tab_3d"] = {
            "fft_surf": fft_surf, "frft_surf": frft_surf,
            "cos_surf": cos_surf, "arb_surf": arb_surf,
            "m_vals_3d": m_vals_3d, "T_vals_3d": T_vals_3d,
            "m_vals_arb": m_vals_arb, "T_vals_arb": T_vals_arb,
            "t_fft": t_fft_surf, "t_frft": t_frft_surf,
            "t_cos": t_cos_surf, "t_arb": t_arb_surf,
            "surf_opt": surf_opt, "surf_dps": surf_dps,
            "params_key": _sidebar_params_key(),
        }

    if "tab_3d" in st.session_state and st.session_state["tab_3d"]["params_key"] == _sidebar_params_key():
        _ts = st.session_state["tab_3d"]

        st.markdown("#### Timing Comparison")
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("FFT", f"{_ts['t_fft']:.3f} s")
        tc2.metric("FRFT", f"{_ts['t_frft']:.3f} s")
        tc3.metric("COS", f"{_ts['t_cos']:.3f} s")
        tc4.metric(f"Arb. Prec. ({_ts['surf_dps']} dps)", f"{_ts['t_arb']:.3f} s")

        st.markdown("---")

        colorscales = {"FFT": "Viridis", "FRFT": "Cividis", "COS": "Plasma", "Arb. Precision": "Inferno"}
        method_data = [
            ("FFT", _ts["fft_surf"], _ts["m_vals_3d"], _ts["T_vals_3d"], _ts["t_fft"]),
            ("FRFT", _ts["frft_surf"], _ts["m_vals_3d"], _ts["T_vals_3d"], _ts["t_frft"]),
            ("COS", _ts["cos_surf"], _ts["m_vals_3d"], _ts["T_vals_3d"], _ts["t_cos"]),
            ("Arb. Precision", _ts["arb_surf"], _ts["m_vals_arb"], _ts["T_vals_arb"], _ts["t_arb"]),
        ]

        for mname, surf, m_v, T_v, t_elapsed in method_data:
            st.markdown(f"##### {mname}  ({t_elapsed:.3f} s)")
            fig_s = go.Figure(data=[go.Surface(
                x=m_v, y=T_v, z=surf, colorscale=colorscales[mname]
            )])
            fig_s.update_layout(
                title=f"{mname} — {_ts['surf_opt'].title()} Price Surface",
                scene=dict(xaxis_title="Moneyness (S/K)", yaxis_title="Expiry (T)",
                           zaxis_title="Price"),
                height=500, margin=dict(t=50, b=10),
            )
            plot_3d_with_download(fig_s, f"{mname.lower().replace(' ', '_')}_surface.html",
                                 key_prefix=f"s3d_{mname}")
            st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Method Comparison (MSE Matrix)
# ═══════════════════════════════════════════════════════════════════════════
with tab_mse:
    st.subheader("Method Comparison — MSE Matrix")
    st.markdown(
        "Computes prices across a grid of **time-to-expiry** and **moneyness** "
        "using four methods (FFT, FRFT, COS, Arbitrary Precision) and shows "
        "the Mean Squared Error between each pair. "
        "Arbitrary Precision (mpmath) is used as the reference when available."
    )

    st.markdown("---")

    with st.form("mse_form"):
        mse_c1, mse_c2, mse_c3 = st.columns(3)
        mse_n_m = int(mse_c1.number_input("Moneyness grid points", value=3, min_value=2, max_value=20, step=1, key="mse_nm"))
        mse_n_t = int(mse_c2.number_input("Expiry grid points", value=3, min_value=2, max_value=20, step=1, key="mse_nt"))
        mse_opt = mse_c3.selectbox("Option Type", ["call", "put"], key="mse_opt")

        mse_m1, mse_m2 = st.columns(2)
        mse_m_lo = mse_m1.number_input("Min moneyness (S/K)", value=0.8, min_value=0.01, step=0.05, format="%.2f", key="mse_mlo")
        mse_m_hi = mse_m2.number_input("Max moneyness (S/K)", value=1.2, min_value=0.1, step=0.05, format="%.2f", key="mse_mhi")

        mse_t1, mse_t2 = st.columns(2)
        mse_T_lo = mse_t1.number_input("Min expiry (T)", value=0.1, min_value=0.01, step=0.05, format="%.2f", key="mse_tlo")
        mse_T_hi = mse_t2.number_input("Max expiry (T)", value=1.0, min_value=0.05, step=0.1, format="%.2f", key="mse_thi")

        mse_dps = int(mse_t1.number_input("Arb. precision (dps)", value=30, min_value=10, max_value=200, step=5, key="mse_dps"))

        submitted_mse = st.form_submit_button("Compute MSE Matrix")

    if submitted_mse:
        vg_mpmath.set_precision(mse_dps)
        m_vals = np.linspace(mse_m_lo, mse_m_hi, mse_n_m)
        T_vals = np.linspace(mse_T_lo, mse_T_hi, mse_n_t)
        K_vals = S / m_vals

        total = mse_n_m * mse_n_t
        fft_prices = np.zeros((mse_n_t, mse_n_m))
        frft_prices = np.zeros((mse_n_t, mse_n_m))
        cos_prices = np.zeros((mse_n_t, mse_n_m))
        arb_prices = np.zeros((mse_n_t, mse_n_m))

        fft_kw = dict(N=fft_N, alpha=alpha_damp, eta=fft_eta)

        progress_mse = st.progress(0, text="Computing prices across grid...")
        t0_mse = time.perf_counter()
        step = 0
        for i, t_val in enumerate(T_vals):
            for j, k_val in enumerate(K_vals):
                fft_prices[i, j] = model.price(k_val, float(t_val), mse_opt, **fft_kw)
                frft_prices[i, j] = model.price_frft(k_val, float(t_val), mse_opt, **fft_kw)
                cos_prices[i, j] = model.price_cos(k_val, float(t_val), mse_opt, N_cos=cos_N, L=cos_L)

                arb_call = float(vg_mpmath._price_only(
                    S, r, q, sigma, theta_vg, nu, float(t_val), float(k_val)
                ))
                if mse_opt == "put":
                    arb_call = arb_call - S * np.exp(-q * float(t_val)) + float(k_val) * np.exp(-r * float(t_val))
                arb_prices[i, j] = arb_call

                step += 1
                progress_mse.progress(step / total,
                                      text=_eta_text("MSE grid", step, total, t0_mse))
        t_mse_elapsed = time.perf_counter() - t0_mse
        progress_mse.empty()

        methods = {"FFT": fft_prices, "FRFT": frft_prices, "COS": cos_prices, "Arb. Precision": arb_prices}
        method_names = list(methods.keys())

        # Pairwise MSE matrix
        n_methods = len(method_names)
        mse_matrix = np.zeros((n_methods, n_methods))
        for a_idx in range(n_methods):
            for b_idx in range(n_methods):
                diff = methods[method_names[a_idx]] - methods[method_names[b_idx]]
                mse_matrix[a_idx, b_idx] = np.mean(diff ** 2)

        st.session_state["tab_mse"] = {
            "mse_matrix": mse_matrix,
            "method_names": method_names,
            "methods": {k: v.tolist() for k, v in methods.items()},
            "m_vals": m_vals.tolist(),
            "T_vals": T_vals.tolist(),
            "t_mse_elapsed": t_mse_elapsed,
            "mse_opt": mse_opt,
            "mse_dps": mse_dps,
            "params_key": _sidebar_params_key(),
        }

    if "tab_mse" in st.session_state and st.session_state["tab_mse"]["params_key"] == _sidebar_params_key():
        _tm = st.session_state["tab_mse"]
        st.metric("Computation Time", f"{_tm['t_mse_elapsed']:.2f} s")
        st.info(f"Option: **{_tm['mse_opt'].title()}** | Arb. precision: **{_tm['mse_dps']}** dps")

        st.markdown("#### Pairwise MSE Matrix")
        df_mse = pd.DataFrame(
            _tm["mse_matrix"],
            index=_tm["method_names"],
            columns=_tm["method_names"],
        )
        # Format for display
        df_mse_display = df_mse.map(lambda x: f"{x:.4e}")
        st.dataframe(df_mse_display, width="stretch")

        st.markdown("#### Price Comparison Table (sample)")
        m_vals = _tm["m_vals"]
        T_vals = _tm["T_vals"]
        rows = []
        for i, t_val in enumerate(T_vals):
            for j, m_val in enumerate(m_vals):
                row = {"T": f"{t_val:.3f}", "S/K": f"{m_val:.3f}"}
                for mname in _tm["method_names"]:
                    row[mname] = f"{_tm['methods'][mname][i][j]:.6f}"
                rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        # Heatmap of MSE
        fig_heat = go.Figure(data=go.Heatmap(
            z=_tm["mse_matrix"],
            x=_tm["method_names"],
            y=_tm["method_names"],
            colorscale="Reds",
            text=[[f"{v:.2e}" for v in row] for row in _tm["mse_matrix"]],
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        fig_heat.update_layout(
            title="Pairwise MSE Heatmap",
            height=400, margin=PLOT_MARGIN,
        )
        st.plotly_chart(fig_heat, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Calibration
# ═══════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.subheader("Calibrate VG Parameters to Market Prices")
    st.markdown(
        "Upload an **Excel file (.xlsx)** with columns: "
        "`K` (strike), `T` (expiry), `price` (market price), "
        "`type` (call / put).  "
        "Optionally include an `r` column; otherwise the sidebar value is used."
    )
    st.markdown(
        "> **Why is a `price` column required?** Calibration works by finding "
        "the VG parameters ($\\sigma$, $\\nu$, $\\theta$) that minimise the sum of "
        "squared errors (SSE) between the **model** prices and the **market** "
        "prices you supply.  Without observed market prices the optimiser has "
        "no target to fit against — the `price` column *is* the calibration "
        "target.  If you only have implied-volatility data, convert it to "
        "prices first using a Black-Scholes pricer and upload those."
    )

    uploaded = st.file_uploader("Upload market data (.xlsx)", type=["xlsx"])

    if uploaded is not None:
        df_raw = pd.read_excel(uploaded)
        df_raw.columns = [c.strip().lower() for c in df_raw.columns]
        st.markdown("**Uploaded data preview:**")
        st.dataframe(df_raw.head(20), width="stretch")

        required = {"k", "t", "price", "type"}
        if not required.issubset(set(df_raw.columns)):
            st.error(f"Missing columns. Need at least: {required}. Found: {set(df_raw.columns)}")
            st.stop()

        K_mkt = df_raw["k"].values.astype(float)
        T_mkt = df_raw["t"].values.astype(float)
        prices_mkt = df_raw["price"].values.astype(float)
        types_mkt = [s.strip().lower() for s in df_raw["type"].values]

        r_cal = float(df_raw["r"].values[0]) if "r" in df_raw.columns else r

        with st.form("calib_form"):
            st.markdown("**Calibration Settings**")
            cal1, cal2 = st.columns(2)
            S_cal = cal1.number_input("Spot for calibration", value=float(S), min_value=0.01, step=1.0, format="%.2f", key="s_cal")
            q_cal = cal2.number_input("Div yield for calibration", value=float(q), step=0.005, format="%.4f", key="q_cal")
            use_global = st.checkbox("Use global optimizer (differential evolution) first", value=True)
            submitted_cal = st.form_submit_button("Run Calibration")

        if submitted_cal:
            with st.spinner("Calibrating..."):
                cal_model, res = VarianceGammaModel.calibrate(
                    S_cal, r_cal, q_cal, K_mkt, T_mkt, prices_mkt, types_mkt,
                    use_global=use_global,
                )

            model_prices = np.array([
                cal_model.price(K_mkt[i], T_mkt[i], types_mkt[i])
                for i in range(len(K_mkt))
            ])
            residuals = prices_mkt - model_prices
            moneyness_cal = S_cal / K_mkt

            st.session_state["tab_cal"] = {
                "cal_sigma": cal_model.sigma,
                "cal_nu": cal_model.nu,
                "cal_theta": cal_model.theta,
                "sse": res.fun,
                "converged": res.success,
                "model_prices": model_prices,
                "residuals": residuals,
                "moneyness_cal": moneyness_cal,
                "K_mkt": K_mkt,
                "T_mkt": T_mkt,
                "prices_mkt": prices_mkt,
                "types_mkt": types_mkt,
                "S_cal": S_cal,
            }

        if "tab_cal" in st.session_state:
            _t4 = st.session_state["tab_cal"]

            st.success("Calibration complete!")
            p1, p2, p3 = st.columns(3)
            p1.metric("sigma", f"{_t4['cal_sigma']:.6f}")
            p2.metric("nu", f"{_t4['cal_nu']:.6f}")
            p3.metric("theta", f"{_t4['cal_theta']:.6f}")

            m1, m2 = st.columns(2)
            m1.metric("SSE", f"{_t4['sse']:.4e}")
            m2.metric("Converged", str(_t4["converged"]))

            df_result = pd.DataFrame({
                "S/K": np.round(_t4["moneyness_cal"], 4),
                "K": _t4["K_mkt"], "T": _t4["T_mkt"], "Type": _t4["types_mkt"],
                "Market Price": _t4["prices_mkt"],
                "Model Price": np.round(_t4["model_prices"], 6),
                "Residual": np.round(_t4["residuals"], 6),
            })
            st.dataframe(df_result, width="stretch")

            st.markdown("<br>", unsafe_allow_html=True)

            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=np.arange(len(_t4["residuals"])), y=_t4["residuals"], name="Residual",
                marker_color=["#2196F3" if rv >= 0 else "#F44336" for rv in _t4["residuals"]],
            ))
            fig_res.update_layout(title="Calibration Residuals (Market - Model)",
                                  xaxis_title="Option #", yaxis_title="Residual",
                                  height=350, margin=PLOT_MARGIN)
            st.plotly_chart(fig_res, width="stretch")

            st.markdown("<br>", unsafe_allow_html=True)

            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(x=_t4["prices_mkt"], y=_t4["model_prices"], mode="markers",
                                         marker=dict(size=8, color="#2196F3"), name="Options"))
            mn = min(_t4["prices_mkt"].min(), _t4["model_prices"].min()) * 0.9
            mx = max(_t4["prices_mkt"].max(), _t4["model_prices"].max()) * 1.1
            fig_fit.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                         line=dict(dash="dash", color="gray"), name="Perfect fit"))
            fig_fit.update_layout(title="Model vs Market Prices",
                                  xaxis_title="Market Price", yaxis_title="Model Price",
                                  height=400, margin=PLOT_MARGIN)
            st.plotly_chart(fig_fit, width="stretch")

            st.markdown("<br>", unsafe_allow_html=True)

            unique_T = np.unique(_t4["T_mkt"])
            if len(unique_T) <= 10:
                fig_smile = go.Figure()
                colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
                          "#00BCD4", "#795548", "#607D8B", "#E91E63", "#CDDC39"]
                for j, t_val in enumerate(unique_T):
                    mask = _t4["T_mkt"] == t_val
                    m_plot = _t4["moneyness_cal"][mask]
                    sort_idx = np.argsort(m_plot)
                    m_sorted = m_plot[sort_idx]
                    mkt_sorted = _t4["prices_mkt"][mask][sort_idx]
                    mod_sorted = _t4["model_prices"][mask][sort_idx]
                    c = colors[j % len(colors)]
                    fig_smile.add_trace(go.Scatter(x=m_sorted, y=mkt_sorted, mode="markers",
                                                    marker=dict(color=c, size=8),
                                                    name=f"Market T={t_val:.3f}"))
                    fig_smile.add_trace(go.Scatter(x=m_sorted, y=mod_sorted, mode="lines",
                                                    line=dict(color=c),
                                                    name=f"Model T={t_val:.3f}"))
                fig_smile.update_layout(title="Market vs Model Prices by Expiry",
                                        xaxis_title="Moneyness (S/K)", yaxis_title="Price",
                                        height=450, margin=PLOT_MARGIN)
                st.plotly_chart(fig_smile, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — Method Explanations
# ═══════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.subheader("Method Explanations")
    st.caption(
        "Each pricing method is explained below with its key formulas and "
        "simplified pseudo code showing how it is implemented in this application."
    )

    # ── Damping Parameter ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Damping Parameter ($\\alpha$)")
    st.markdown(
        "The raw Fourier transform of the call price $C(K)$ does not converge "
        "because $C(K) \\to S$ as $K \\to 0$, so it is not in $L^1$.  "
        "Carr & Madan (1999) fix this by multiplying by $e^{\\alpha k}$ "
        "(where $k = \\ln K$), producing a *dampened* call price that **is** "
        "square-integrable and therefore has a well-defined Fourier transform."
    )
    st.latex(
        r"c_T(k) = e^{\alpha k}\, C(e^k)"
        r"\quad\Longrightarrow\quad"
        r"\hat c_T(u) = \int_{-\infty}^{\infty} e^{iuk}\, c_T(k)\, dk"
        r"\quad\text{converges for } \alpha > 0"
    )
    st.markdown(
        "The standard choice $\\alpha = 1.5$ balances convergence speed "
        "against numerical noise.  Smaller values cause oscillation; "
        "larger values amplify rounding errors at extreme strikes.  "
        "The parameter is adjustable in the sidebar."
    )

    # ── VG Process ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### The Variance Gamma Process")
    st.markdown(
        "The VG process models the log-return of the asset as a Brownian "
        "motion evaluated at a random (gamma-distributed) time.  "
        "Under the risk-neutral measure:"
    )
    st.latex(
        r"\ln\!\frac{S_T}{S_0} = (r - q + \omega)\,T + \theta\,G_T + \sigma\,W_{G_T}"
    )
    st.markdown("where $G_T \\sim \\text{Gamma}(T/\\nu,\\, \\nu)$ and the "
                "martingale correction is:")
    st.latex(
        r"\omega = \frac{1}{\nu}\,\ln\!\bigl(1 - \theta\,\nu - \tfrac{1}{2}\sigma^2\nu\bigr)"
    )
    st.markdown(
        "**Parameters:**\n"
        "- $\\sigma$ — volatility of the Brownian motion component\n"
        "- $\\nu$ — variance rate of the gamma subordinator (controls kurtosis)\n"
        "- $\\theta$ — drift of the BM component (controls skewness)"
    )
    st.markdown("The characteristic function of $\\ln S_T$ is:")
    st.latex(
        r"\varphi(u) = e^{iu[\ln S_0 + (r-q+\omega)T]}"
        r"\;\Bigl(1 - iu\,\theta\,\nu + \tfrac{1}{2}\sigma^2\nu\,u^2\Bigr)^{-T/\nu}"
    )

    # ── FFT Method ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1. FFT — Carr-Madan (1999)")
    st.markdown(
        "The standard Fourier approach.  The dampened call price is recovered "
        "by inverting its Fourier transform on a uniform grid, using the FFT "
        "for $O(N \\log N)$ efficiency."
    )
    st.latex(
        r"C(K) = \frac{e^{-\alpha k}}{\pi}"
        r"\int_0^\infty e^{-iuk}\,\Psi(u)\,du"
    )
    st.markdown("where the modified characteristic function is:")
    st.latex(
        r"\Psi(u) = \frac{e^{-rT}\,\varphi\bigl(u - (\alpha+1)i\bigr)}"
        r"{\alpha^2 + \alpha - u^2 + i(2\alpha+1)u}"
    )
    st.markdown("**Key constraint:** the frequency spacing $\\eta$ and "
                "log-strike spacing $\\lambda$ are *coupled*: "
                "$\\lambda \\cdot \\eta = 2\\pi / N$.")
    st.markdown("**Pseudo code:**")
    st.code("""
def fft_price(K, T, alpha, eta, N):
    lam = 2*pi / (N * eta)
    b   = N * lam / 2

    u = [0, eta, 2*eta, ..., (N-1)*eta]
    w = simpson_weights(N)

    # Evaluate modified char fn at each grid point
    psi = carr_madan_psi(u, T, alpha)

    # Build FFT input with shift factor
    x = exp(i*b*u) * psi * eta * w

    # Single FFT call
    fft_result = FFT(x)

    # Recover call prices on log-strike grid
    k = [-b, -b+lam, ..., -b+(N-1)*lam]
    call_prices = Re[ exp(-alpha*k) / pi * fft_result ]

    # Interpolate to desired strike
    return interp(log(K), k, call_prices)
""", language="python")

    # ── FRFT Method ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2. FRFT — Chourdakis (2005)")
    st.markdown(
        "The *Fractional* FFT decouples $\\eta$ and $\\lambda$ via the "
        "Bluestein chirp-z decomposition.  This lets you independently "
        "control the frequency resolution and the strike grid density.  "
        "The cost is three FFTs of size $2N$ instead of one FFT of size $N$."
    )
    st.markdown(
        "The fractional parameter $\\beta = \\lambda\\eta / (2\\pi)$ replaces the "
        "standard DFT kernel $e^{-2\\pi i jk/N}$ with "
        "$e^{-2\\pi i jk\\beta}$."
    )
    st.markdown("**Why use it?**  When the FFT's coupled grid is too coarse "
                "for accurate interpolation at your target strikes, the FRFT "
                "provides a finer strike grid without increasing $N$.")
    st.markdown("**Pseudo code:**")
    st.code("""
def frft(x, beta):
    \"\"\"Fractional FFT via Bluestein chirp-z.\"\"\"
    N = len(x)
    M = next_power_of_2(2*N)

    chirp = exp(-i*pi*beta * [0,1,4,9,...,(N-1)^2])

    a = zero_pad(x * chirp, M)
    b = zero_pad(exp(+i*pi*beta * [0,1,4,...]), M)
    b[M-N+1:] = reversed(b[1:N])      # wrap-around

    c = IFFT( FFT(a) * FFT(b) )       # 3 FFTs
    return c[:N] * chirp

def frft_price(K, T, alpha, eta, lam, N):
    beta = lam * eta / (2*pi)
    b    = N * lam / 2
    u    = [0, eta, ..., (N-1)*eta]
    w    = simpson_weights(N)
    psi  = carr_madan_psi(u, T, alpha)
    x    = exp(i*b*u) * psi * eta * w

    fft_result = frft(x, beta)         # fractional FFT

    k = [-b, -b+lam, ..., -b+(N-1)*lam]
    call_prices = Re[ exp(-alpha*k) / pi * fft_result ]
    return interp(log(K), k, call_prices)
""", language="python")

    # ── COS Method ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3. COS Method — Fang & Oosterlee (2008)")
    st.markdown(
        "Expands the risk-neutral density of the log-return $Z = \\ln(S_T/S_0)$ "
        "as a truncated cosine series on an interval $[a, b]$ derived from "
        "the VG cumulants.  Option prices follow analytically from the "
        "payoff's cosine coefficients."
    )
    st.latex(
        r"C(K) \approx e^{-rT} \sum_{k=0}^{N_{\cos}-1}"
        r"\,' \operatorname{Re}\!\bigl[\varphi_Z(u_k)\,e^{-iu_k a}\bigr]"
        r"\; V_k"
    )
    st.markdown("where $u_k = k\\pi/(b-a)$, the prime means the $k=0$ term "
                "is halved, and $V_k$ are the analytical payoff coefficients:")
    st.latex(
        r"V_k^{\text{call}} = \frac{2}{b-a}\,K\,"
        r"\bigl[e^x\,\chi_k(c, b) - \psi_k(c, b)\bigr]"
        r"\qquad x = \ln(S/K)"
    )
    st.markdown(
        "The helper integrals $\\chi_k$ (exponential cosine) and $\\psi_k$ "
        "(plain cosine) have closed-form expressions."
    )
    st.markdown(
        "**Truncation range** $[a, b]$ is computed from the first four "
        "cumulants of the log-return, scaled by the parameter $L$:"
    )
    st.latex(
        r"[a,b] = \bigl[c_1 - L\sqrt{|c_2| + \sqrt{|c_4|}}\,,\;"
        r"c_1 + L\sqrt{|c_2| + \sqrt{|c_4|}}\bigr]"
    )
    st.markdown("**Pseudo code:**")
    st.code("""
def cos_price(K, T, option_type, N_cos, L):
    # 1. Truncation range from VG cumulants
    a, b = cos_truncation(T, L)

    # 2. Cosine frequencies
    u_k = k * pi / (b - a)    for k = 0..N_cos-1

    # 3. Density expansion coefficients (strike-independent)
    F_k = Re[ phi_Z(u_k) * exp(-i*u_k*a) ]
    F_k[0] *= 0.5             # halve k=0 term

    # 4. For each strike K:
    x = log(S / K)             # log-moneyness
    c_lo = max(a, -x)

    if call:
        chi_k = integral_exp_cos(k, a, b, c_lo, b)
        psi_k = integral_cos(k, a, b, c_lo, b)
        V_k = (2/(b-a)) * K * (exp(x)*chi_k - psi_k)
    else:
        c_hi = min(b, -x)
        chi_k = integral_exp_cos(k, a, b, a, c_hi)
        psi_k = integral_cos(k, a, b, a, c_hi)
        V_k = (2/(b-a)) * K * (-exp(x)*chi_k + psi_k)

    price = exp(-r*T) * sum(F_k * V_k)
    return price
""", language="python")

    # ── Arbitrary Precision ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4. Arbitrary Precision — mpmath Quadrature")
    st.markdown(
        "Direct numerical integration of the pricing integrals using "
        "mpmath's adaptive Gauss-Legendre quadrature at user-selectable "
        "decimal precision.  This is the *slowest* method but produces "
        "a high-accuracy reference for validating the FFT / FRFT / COS "
        "results."
    )
    st.markdown("The call price is decomposed as:")
    st.latex(
        r"C = S\,e^{-qT}\,\Pi_1 - K\,e^{-rT}\,\Pi_2"
    )
    st.markdown("where $\\Pi_1$ and $\\Pi_2$ are Gil-Pelaez style integrals "
                "of the characteristic function:")
    st.latex(
        r"\Pi_j = \frac{1}{2} + \frac{1}{\pi}"
        r"\int_0^\infty \operatorname{Re}\!\Bigl["
        r"\frac{e^{-iu\ln K}\,\varphi_j(u)}{iu}\Bigr]\,du"
    )
    st.markdown(
        "Delta and gamma are computed analytically via differentiation "
        "of the integrand; all other Greeks use central finite differences "
        "at arbitrary precision (bump size $\\varepsilon \\approx 10^{-10}$ "
        "to $10^{-20}$)."
    )
    st.markdown("**Pseudo code:**")
    st.code("""
def arb_price(S, r, q, sigma, theta, nu, T, K, dps):
    set_precision(dps)         # e.g. 50 decimal places

    # Gil-Pelaez integrals via adaptive quadrature
    pi_1 = 0.5 + (1/pi) * quad(pi1_integrand, [0, inf])
    pi_2 = 0.5 + (1/pi) * quad(pi2_integrand, [0, inf])

    call = S * exp(-q*T) * pi_1 - K * exp(-r*T) * pi_2
    delta = exp(-q*T) * pi_1   # analytical
    gamma = quad(gamma_integrand, [0, inf])  # analytical

    # Other Greeks via central finite differences
    theta_greek = (price(T+eps) - price(T-eps)) / (2*eps)
    vega        = (price(sigma+eps) - price(sigma-eps)) / (2*eps)
    # ... similarly for rho, d/d_theta, d/d_nu

    return call, delta, gamma, theta_greek, vega, ...
""", language="python")

    # ── Autodiff Greeks ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 5. Autodiff Greeks — PyTorch")
    st.markdown(
        "When PyTorch is installed, the FFT pricing formula is re-implemented "
        "using torch tensors so that `torch.autograd.grad` can compute exact "
        "gradients (delta, gamma, vega, rho, theta, and VG sensitivities) "
        "in a single backward pass.  This runs on GPU if CUDA is available."
    )
    st.markdown("**Pseudo code:**")
    st.code("""
def greeks_autodiff(S, r, q, sigma, nu, theta, K, T):
    # Build differentiable tensors (requires_grad=True)
    S_t, r_t, sigma_t, nu_t, theta_t, T_t = ...

    # Forward: Carr-Madan FFT in pure torch
    call = torch_fft_price(S_t, r_t, ..., K, T_t)

    # Backward: automatic differentiation
    delta, rho, vega, d_nu, d_theta, theta_greek = \\
        autograd.grad(call, [S_t, r_t, sigma_t, nu_t, theta_t, T_t],
                      create_graph=True)

    gamma = autograd.grad(delta, S_t)   # 2nd order

    return {delta, gamma, theta, vega, rho, d_theta, d_nu}
""", language="python")

    st.markdown("---")
    st.markdown(
        "*All methods share the same VG characteristic function and model "
        "parameters; they differ only in how the pricing integral is evaluated.  "
        "FFT and FRFT work in Fourier space, COS uses cosine-series expansion, "
        "and Arbitrary Precision performs direct quadrature.*"
    )
