import time

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from variance_gamma import VarianceGammaModel, _HAS_TORCH
import vg_mpmath

st.set_page_config(page_title="Variance Gamma Pricer", layout="wide")
st.title("Variance Gamma Option Pricer")

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

PLOT_MARGIN = dict(t=50, b=20)

_dl_counter = 0


def plot_3d_with_download(fig, filename, key_prefix=""):
    """Display a 3D plotly figure and offer an HTML download button."""
    global _dl_counter
    _dl_counter += 1
    st.plotly_chart(fig, use_container_width=True)
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        label=f"Download {filename}",
        data=html_bytes,
        file_name=filename,
        mime="text/html",
        key=f"dl_{key_prefix}_{_dl_counter}",
    )


def _sidebar_params_key():
    """Return a tuple of sidebar params to detect changes."""
    return (S, r, q, sigma, nu, theta_vg)


# ── Tabs ───────────────────────────────────────────────────────────────────
tab_single, tab_curves, tab_greeks, tab_arb, tab_calib = st.tabs([
    "Single-Point Pricer",
    "Price Curves & Greeks",
    "Greeks (Custom)",
    "Arbitrary Precision 3D",
    "Calibration",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Single-Point Pricer
# ═══════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Single-Point Option Value & Greeks")
    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        K_single = c1.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
        T_single = c2.number_input("Time to Expiry (T, years)", value=0.5, min_value=0.01, step=0.05, format="%.4f")
        opt_type_single = c3.selectbox("Option Type", ["call", "put"])
        submitted_single = st.form_submit_button("Calculate")

    if submitted_single:
        call_px = model.price(K_single, T_single, "call")
        put_px = model.price(K_single, T_single, "put")
        parity_err = call_px - put_px - S * np.exp(-q * T_single) + K_single * np.exp(-r * T_single)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Call Price", f"{call_px:.6f}")
        mc2.metric("Put Price", f"{put_px:.6f}")
        mc3.metric("Put-Call Parity Error", f"{parity_err:.2e}")

        st.markdown("---")
        col_an, col_ad = st.columns(2)

        with col_an:
            st.markdown("**Analytical Greeks (FFT)**")
            g_an = model.greeks(K_single, T_single, opt_type_single)
            df_an = pd.DataFrame({"Greek": list(g_an.keys()), "Value": [f"{v:.6f}" for v in g_an.values()]})
            st.dataframe(df_an, hide_index=True, use_container_width=True)

        with col_ad:
            if _HAS_TORCH:
                st.markdown("**Autodiff Greeks (PyTorch)**")
                g_ad = model.greeks_ad(K_single, T_single, opt_type_single)
                df_ad = pd.DataFrame({
                    "Greek": list(g_ad.keys()),
                    "Autodiff": [f"{v:.6f}" for v in g_ad.values()],
                    "|Diff|": [f"{abs(g_an[k] - g_ad[k]):.2e}" for k in g_an],
                })
                st.dataframe(df_ad, hide_index=True, use_container_width=True)
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
            g = model.greeks(k, T_curve, opt_type_curve)
            for gn in an_data:
                an_data[gn].append(g[gn])
            progress.progress((idx + 1) / len(K_arr))
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
                g = model.greeks_ad(k, T_curve, opt_type_curve)
                for gn in ad_data:
                    ad_data[gn].append(g[gn])
                progress2.progress((idx + 1) / len(K_arr))
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
        an_surfaces = {gn: np.zeros((n_pts, n_pts)) for gn in surface_names}
        for i, t in enumerate(T_range):
            for j, k in enumerate(K_surf):
                g = model.greeks(k, t, "call")
                for gn in surface_names:
                    an_surfaces[gn][i, j] = g[gn]
            progress_3d.progress((i + 1) / n_pts)
        progress_3d.empty()

        ad_surfaces = None
        if _HAS_TORCH:
            progress_3d_ad = st.progress(0, text="Computing autodiff 3D surfaces...")
            ad_surfaces = {gn: np.zeros((n_pts, n_pts)) for gn in surface_names}
            for i, t in enumerate(T_range):
                for j, k in enumerate(K_surf):
                    g = model.greeks_ad(k, t, "call")
                    for gn in surface_names:
                        ad_surfaces[gn][i, j] = g[gn]
                progress_3d_ad.progress((i + 1) / n_pts)
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
            st.dataframe(pd.DataFrame(mse_rows), hide_index=True, use_container_width=True)
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
        st.plotly_chart(fig_price, use_container_width=True)

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
        st.plotly_chart(fig_greeks, use_container_width=True)

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
            g = model.greeks(k, T_greek, opt_type_grk)
            for gn in greek_names:
                data_an[gn].append(g[gn])
            progress_an.progress((idx + 1) / len(K_arr))
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
                g = model.greeks_ad(k, T_greek, opt_type_grk)
                for gn in greek_names:
                    data_ad[gn].append(g[gn])
                progress_ad.progress((idx + 1) / len(K_arr))
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
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Arbitrary Precision 3D Surfaces (mpmath)
# ═══════════════════════════════════════════════════════════════════════════
with tab_arb:
    st.subheader("Arbitrary Precision 3D Surfaces (mpmath)")
    st.caption(
        "Uses mpmath adaptive quadrature (not FFT) with user-selectable decimal precision. "
        "Computes call price, analytical delta, and analytical gamma. "
        "**Much slower** than FFT — keep grid sizes small (≤ 15)."
    )

    with st.form("arb_form"):
        ac1, ac2, ac3 = st.columns(3)
        arb_dps = int(ac1.number_input(
            "Decimal places (dps)", value=30, min_value=10, max_value=200, step=5, key="arb_dps"
        ))
        arb_n_pts = int(ac2.number_input(
            "Grid points", value=8, min_value=3, max_value=30, step=1, key="arb_npts"
        ))
        arb_opt = ac3.selectbox("Option Type", ["call", "put"], key="arb_opt")

        am1, am2 = st.columns(2)
        arb_m_lo = am1.number_input("Min moneyness (S/K)", value=0.8, min_value=0.01, step=0.05, format="%.2f", key="arb_mlo")
        arb_m_hi = am2.number_input("Max moneyness (S/K)", value=1.2, min_value=0.1, step=0.05, format="%.2f", key="arb_mhi")

        at1, at2 = st.columns(2)
        arb_T_lo = at1.number_input("Min expiry (T)", value=0.1, min_value=0.01, step=0.05, format="%.2f", key="arb_tlo")
        arb_T_hi = at2.number_input("Max expiry (T)", value=1.5, min_value=0.05, step=0.1, format="%.2f", key="arb_thi")

        submitted_arb = st.form_submit_button("Compute Surfaces")

    if submitted_arb:
        vg_mpmath.set_precision(arb_dps)

        m_range_arb = np.linspace(arb_m_lo, arb_m_hi, arb_n_pts)
        T_range_arb = np.linspace(arb_T_lo, arb_T_hi, arb_n_pts)
        K_range_arb = S / m_range_arb

        arb_surfaces = {
            "price": np.zeros((arb_n_pts, arb_n_pts)),
            "delta": np.zeros((arb_n_pts, arb_n_pts)),
            "gamma": np.zeros((arb_n_pts, arb_n_pts)),
        }

        total_steps = arb_n_pts * arb_n_pts
        progress_arb = st.progress(0, text=f"Computing mpmath surfaces ({arb_dps} dps)...")
        t0_arb = time.perf_counter()

        step = 0
        for i, t_val in enumerate(T_range_arb):
            for j, k_val in enumerate(K_range_arb):
                px, dlt, gma, _, _, _ = vg_mpmath.call_price(
                    S, r, q, sigma, theta_vg, nu, float(t_val), float(k_val)
                )
                arb_surfaces["price"][i, j] = float(px)
                arb_surfaces["delta"][i, j] = float(dlt)
                arb_surfaces["gamma"][i, j] = float(gma)

                # Put-call parity adjustments
                if arb_opt == "put":
                    call_px = float(px)
                    put_px = call_px - S * np.exp(-q * float(t_val)) + float(k_val) * np.exp(-r * float(t_val))
                    arb_surfaces["price"][i, j] = put_px
                    arb_surfaces["delta"][i, j] = float(dlt) - np.exp(-q * float(t_val))
                    # gamma is same for call and put

                step += 1
                progress_arb.progress(step / total_steps)

        t_arb_elapsed = time.perf_counter() - t0_arb
        progress_arb.empty()

        st.session_state["tab_arb"] = {
            "arb_surfaces": arb_surfaces,
            "m_range_arb": m_range_arb,
            "T_range_arb": T_range_arb,
            "t_arb_elapsed": t_arb_elapsed,
            "arb_dps": arb_dps,
            "arb_opt": arb_opt,
            "params_key": _sidebar_params_key(),
        }

    if "tab_arb" in st.session_state and st.session_state["tab_arb"]["params_key"] == _sidebar_params_key():
        _ta = st.session_state["tab_arb"]

        st.metric("Computation Time", f"{_ta['t_arb_elapsed']:.2f} s")
        st.info(f"Precision: **{_ta['arb_dps']}** decimal places  |  Option: **{_ta['arb_opt'].title()}**")

        arb_labels = {"price": "Call Price" if _ta["arb_opt"] == "call" else "Put Price",
                      "delta": "Delta", "gamma": "Gamma"}

        for gn in ["price", "delta", "gamma"]:
            fig_arb = go.Figure(data=[go.Surface(
                x=_ta["m_range_arb"], y=_ta["T_range_arb"], z=_ta["arb_surfaces"][gn],
                colorscale="Plasma",
            )])
            fig_arb.update_layout(
                title=f"Arbitrary Precision {arb_labels[gn]} ({_ta['arb_dps']} dps)",
                scene=dict(
                    xaxis_title="Moneyness (S/K)",
                    yaxis_title="Expiry (T)",
                    zaxis_title=arb_labels[gn],
                ),
                height=500, margin=dict(t=50, b=10),
            )
            plot_3d_with_download(fig_arb, f"mpmath_{gn}_surface.html", key_prefix=f"arb_{gn}")
            st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Calibration
# ═══════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.subheader("Calibrate VG Parameters to Market Prices")
    st.markdown(
        "Upload an **Excel file (.xlsx)** with columns: "
        "`K` (strike), `T` (expiry), `price` (market price), "
        "`type` (call / put).  "
        "Optionally include an `r` column; otherwise the sidebar value is used."
    )

    uploaded = st.file_uploader("Upload market data (.xlsx)", type=["xlsx"])

    if uploaded is not None:
        df_raw = pd.read_excel(uploaded)
        df_raw.columns = [c.strip().lower() for c in df_raw.columns]
        st.markdown("**Uploaded data preview:**")
        st.dataframe(df_raw.head(20), use_container_width=True)

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

            st.session_state["tab4"] = {
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

        if "tab4" in st.session_state:
            _t4 = st.session_state["tab4"]

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
            st.dataframe(df_result, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=np.arange(len(_t4["residuals"])), y=_t4["residuals"], name="Residual",
                marker_color=["#2196F3" if rv >= 0 else "#F44336" for rv in _t4["residuals"]],
            ))
            fig_res.update_layout(title="Calibration Residuals (Market - Model)",
                                  xaxis_title="Option #", yaxis_title="Residual",
                                  height=350, margin=PLOT_MARGIN)
            st.plotly_chart(fig_res, use_container_width=True)

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
            st.plotly_chart(fig_fit, use_container_width=True)

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
                st.plotly_chart(fig_smile, use_container_width=True)
