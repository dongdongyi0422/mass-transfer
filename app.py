# app.py — Unified Transport Simulators (Gas / Ion / Vascular)
# - Ion membrane
# - Gas membrane
# - Drug in vessel: 1D ADR transient

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import to_rgba

# -------------------- constants --------------------
R  = 8.314462618
F  = 96485.33212
kB = 1.380649e-23
h  = 6.62607015e-34
NA = 6.02214076e23

# =====================================================================
# UI helpers
# =====================================================================
def nudged_slider(label, vmin, vmax, vstep, vinit, key, unit="", decimals=3, help=None):
    if key not in st.session_state: st.session_state[key] = float(vinit)
    if f"{key}__who" not in st.session_state: st.session_state[f"{key}__who"] = ""
    lab = f"{label}{(' ['+unit+']') if unit else ''}"
    fmt = f"%.{int(decimals)}f"
    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"
    st.slider(lab, float(vmin), float(vmax), float(st.session_state[key]),
              float(vstep), help=help, key=f"{key}__s", format=fmt, on_change=_mark_s)
    st.number_input("", float(vmin), float(vmax), float(st.session_state[key]),
                    float(vstep), key=f"{key}__n", format=fmt, on_change=_mark_n)
    if st.session_state[f"{key}__who"] == "n":
        new = float(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        new = float(st.session_state[f"{key}__s"])
    else:
        new = float(st.session_state[key])
    st.session_state[key] = float(np.clip(new, vmin, vmax))
    return float(st.session_state[key])

def nudged_int(label, vmin, vmax, vstep, vinit, key, help=None):
    if key not in st.session_state: st.session_state[key] = int(vinit)
    if f"{key}__who" not in st.session_state: st.session_state[f"{key}__who"] = ""
    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"
    st.slider(label, int(vmin), int(vmax), int(st.session_state[key]), int(vstep),
              help=help, key=f"{key}__s", on_change=_mark_s)
    st.number_input("", int(vmin), int(vmax), int(st.session_state[key]), int(vstep),
                    key=f"{key}__n", on_change=_mark_n)
    if st.session_state[f"{key}__who"] == "n":
        new = int(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        new = int(st.session_state[f"{key}__s"])
    else:
        new = int(st.session_state[key])
    st.session_state[key] = int(np.clip(new, vmin, vmax))
    return int(st.session_state[key])

def log_slider(label, exp_min, exp_max, exp_step, exp_init, key, unit="", help=None):
    """Returns 10^x (real value). (세션엔 지수 x 저장)"""
    if key not in st.session_state: st.session_state[key] = float(exp_init)
    if f"{key}__who" not in st.session_state: st.session_state[f"{key}__who"] = ""
    lab = f"{label}{(' ['+unit+']') if unit else ''}"
    def _mark_s(): st.session_state[f"{key}__who"] = "s"
    def _mark_n(): st.session_state[f"{key}__who"] = "n"
    st.slider(lab, float(exp_min), float(exp_max), float(st.session_state[key]),
              float(exp_step), help=help, key=f"{key}__s", format="%.2f", on_change=_mark_s)
    st.number_input("exp (10^x)", float(exp_min), float(exp_max), float(st.session_state[key]),
                    float(exp_step), key=f"{key}__n", format="%.2f", on_change=_mark_n)
    if st.session_state[f"{key}__who"] == "n":
        exp_val = float(st.session_state[f"{key}__n"])
    elif st.session_state[f"{key}__who"] == "s":
        exp_val = float(st.session_state[f"{key}__s"])
    else:
        exp_val = float(st.session_state[key])
    st.session_state[key] = float(exp_val)
    return 10.0 ** float(st.session_state[key])

# =====================================================================
# MODE 2 — ION MEMBRANE (ONLY NP + Donnan)
# =====================================================================
ION_DB = {
    "Na+": {"z": +1, "D": 1.33e-9},
    "K+":  {"z": +1, "D": 1.96e-9},
    "Li+": {"z": +1, "D": 1.03e-9},
    "Cl-": {"z": -1, "D": 2.03e-9},
    "OH-": {"z": -1, "D": 5.27e-9},
    "H+":  {"z": +1, "D": 9.31e-9},
}

def eta_water_PaS(T_K: float) -> float:
    T_C = float(T_K) - 273.15
    A, B, C = 2.414e-5, 247.8, 140.0
    return A * (10.0 ** (B / (T_C - C)))

def D_temp_correction(D_ref: float, T_ref: float, T: float, eta_ref: float, eta: float) -> float:
    return float(D_ref) * (float(T)/float(T_ref)) * (float(eta_ref)/float(eta))

def donnan_potential_general_multi(c_bulk, z_map, K_map, Xf, T):
    """
    Solve electroneutrality at interface:
        sum_i z_i C_m,i + Xf = 0,
        C_m,i = K_i C_b,i exp(-z_i F ψ / RT)
    """
    psi = 0.0
    for _ in range(80):
        t  = -F*psi/(R*T)
        g  = Xf
        dg = 0.0
        for sp, cb in c_bulk.items():
            z  = z_map[sp]
            Ki = K_map.get(sp, 1.0)
            cm = Ki*cb*np.exp(-z*t)
            g  += z*cm
            dg += (F/(R*T))*(z**2)*cm
        step = -g/(dg + 1e-30)
        psi += step
        if abs(step) < 1e-12: break
    return psi

def membrane_side_conc_from_donnan(c_bulk, z_map, K_map, psi, T):
    return {sp: K_map.get(sp,1.0)*c_bulk[sp]*np.exp(-(z_map[sp]*F*psi)/(R*T))
            for sp in c_bulk}

def np_flux_constfield(D_eff, z, Cin_m, Cout_m, dphi_mem, L, T, v_s=0.0):
    """
    1D steady Nernst–Planck with constant coefficients:
    J_i = -D_eff * (ΔC/L) - (z_i D_eff F / RT) * C_avg * (Δφ_mem/L) + v_s * C_avg
    """
    L = max(float(L), 1e-12)
    Cavg = 0.5*(Cin_m + Cout_m)
    dC   = float(Cout_m - Cin_m)  # x: feed -> permeate
    term_d = -D_eff * (dC / L)
    term_e = -(z * D_eff * F / (R*T)) * Cavg * (dphi_mem / L)
    term_c = v_s * Cavg
    return term_d + term_e + term_c

def run_ion_membrane():
    st.header("Ion-Exchange / NF Membrane — NP + Donnan (GHK 제거)")

    with st.sidebar:
        st.subheader("Membrane & Operation")
        T   = nudged_slider("Temperature", 273.15, 360.0, 0.5, 298.15, key="T_i", unit="K")
        Lnm = nudged_slider("Active layer thickness", 10.0, 5000.0, 1.0, 200.0, key="Lnm_i", unit="nm")
        eps = nudged_slider("Porosity ε", 0.05, 0.8, 0.01, 0.30, key="eps_i", unit="–")
        tau = nudged_slider("Tortuosity τ", 1.0, 5.0, 0.1, 2.0, key="tau_i", unit="–")
        Cf  = nudged_slider("Fixed charge C_f", -3000.0, 3000.0, 10.0, -500.0, key="Cf_i", unit="mol/m³")
        dV  = nudged_slider("Applied potential ΔV", -0.3, 0.3, 0.005, 0.05, key="dV_i", unit="V")
        v_s = nudged_slider("Solvent velocity v", -1e-6, 1e-6, 1e-7, 0.0, key="vsolv_i", unit="m/s")

        st.subheader("Species set")
        all_cations = [k for k in ION_DB if ION_DB[k]["z"]>0]
        all_anions  = [k for k in ION_DB if ION_DB[k]["z"]<0]
        sel_cat = st.multiselect("Cations", all_cations, default=["Na+","K+"], key="sel_cat_i")
        sel_an  = st.multiselect("Anions",  all_anions,  default=["Cl-","OH-"], key="sel_an_i")

        st.subheader("Bulk concentrations (mol/m³)")
        c_feed = {}; c_perm = {}
        for sp in sel_cat + sel_an:
            c_feed[sp] = nudged_slider(f"{sp} feed", 0.0, 2000.0, 1.0, 100.0, key=f"cf_{sp}", unit="mol/m³")
            c_perm[sp] = nudged_slider(f"{sp} permeate", 0.0, 2000.0, 1.0, 10.0,  key=f"cp_{sp}", unit="mol/m³")

        st.subheader("Partition K & Diffusivity D(T)")
        auto_kd = st.checkbox("Auto D via T/η (K=1)", value=True, key="autoKD_i")

        if auto_kd:
            K_map = {sp: 1.0 for sp in (sel_cat + sel_an)}
            T_ref = 298.15
            eta_ref = eta_water_PaS(T_ref)
            eta_now = eta_water_PaS(T)
            D_map = {sp: D_temp_correction(ION_DB[sp]["D"], T_ref, T, eta_ref, eta_now)
                     for sp in (sel_cat + sel_an)}
            with st.expander("Auto values"):
                for sp in (sel_cat + sel_an):
                    st.write(f"{sp}: D(T)={D_map[sp]:.3e} m²/s,  K=1.0")
        else:
            K_map = {sp: nudged_slider(f"K {sp}", 0.01, 10.0, 0.01, 1.0, key=f"K_{sp}") for sp in (sel_cat+sel_an)}
            D_map = {sp: log_slider(f"D {sp}", -11.0, -8.0, 0.1, np.log10(ION_DB[sp]['D']), key=f"D_{sp}", unit="m²/s")
                     for sp in (sel_cat+sel_an)}

    # ---- NP + Donnan 계산 ----
    L = Lnm*1e-9
    z_map = {sp: int(ION_DB[sp]["z"]) for sp in (sel_cat+sel_an)}
    D_eff = {sp: D_map[sp]*eps/max(tau,1e-9) for sp in (sel_cat+sel_an)}

    # Donnan at both interfaces
    psi_f = donnan_potential_general_multi(c_feed, z_map, K_map, Cf, T)
    psi_p = donnan_potential_general_multi(c_perm, z_map, K_map, Cf, T)

    Cm_f = membrane_side_conc_from_donnan(c_feed, z_map, K_map, psi_f, T)
    Cm_p = membrane_side_conc_from_donnan(c_perm, z_map, K_map, psi_p, T)

    # Membrane internal potential drop
    dphi_mem = float(dV) - float(psi_p - psi_f)

    # Fluxes
    J = {}
    for sp in (sel_cat+sel_an):
        J[sp] = np_flux_constfield(D_eff[sp], z_map[sp], Cm_f[sp], Cm_p[sp],
                                   dphi_mem, L, T, v_s=v_s)
    i_net = F*sum(z_map[sp]*J[sp] for sp in (sel_cat+sel_an))

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Potentials")
        st.metric("ψ_feed (Donnan)", f"{psi_f:.3e} V")
        st.metric("ψ_perm (Donnan)", f"{psi_p:.3e} V")
        st.metric("Δφ_mem (internal)", f"{dphi_mem:.3e} V")
        st.subheader("Net current")
        st.metric("i_net", f"{i_net:.3e} A/m²")
    with col2:
        species = sel_cat + sel_an
        vals = [J[s] for s in species]
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(range(len(species)), vals)
        ax.set_xticks(range(len(species))); ax.set_xticklabels(species)
        ax.set_ylabel("Flux J (mol m⁻² s⁻¹)")
        ax.grid(True, axis='y')
        st.pyplot(fig, use_container_width=True); plt.close(fig)

# =====================================================================
# MODE 1 — GAS MEMBRANE (간결/안정: 밴드 + 퍼미언스/선택도)
# =====================================================================
GPU_UNIT = 3.35e-10
MECH_ORDER = ["Blocked","Sieving","Knudsen","Surface","Capillary","Solution"]
MECH_COLOR = {
    "Blocked":"#bdbdbd","Sieving":"#1f78b4","Knudsen":"#33a02c",
    "Surface":"#e31a1c","Capillary":"#ff7f00","Solution":"#6a3d9a"
}
GAS_PARAMS = {
    "H2":  {"M":2.016e-3,  "d":2.89, "Ea_s":8.0e3},
    "D2":  {"M":4.028e-3,  "d":2.89, "Ea_s":8.5e3},
    "He":  {"M":4.003e-3,  "d":2.60, "Ea_s":6.0e3},
    "N2":  {"M":28.0134e-3,"d":3.64, "Ea_s":9.0e3},
    "O2":  {"M":31.998e-3, "d":3.46, "Ea_s":9.0e3},
    "CO2": {"M":44.01e-3,  "d":3.30, "Ea_s":9.5e3},
    "CH4": {"M":16.043e-3, "d":3.80, "Ea_s":9.5e3},
    "C3H6":{"M":42.081e-3, "d":4.00, "Ea_s":1.05e4},
    "C3H8":{"M":44.097e-3, "d":4.30, "Ea_s":1.05e4},
}

def de_broglie_lambda_m(T, M):
    m = M / NA
    return h / np.sqrt(2.0*np.pi*m*kB*max(float(T),1e-9))

def effective_diameter_A(gas, T, alpha):
    dA = GAS_PARAMS[gas]["d"]
    lamA = de_broglie_lambda_m(T, GAS_PARAMS[gas]["M"]) * 1e10  # m->Å
    d_eff = dA - float(alpha)*lamA
    return float(max(0.5*dA, d_eff))

def pintr_knudsen_SI(d_nm, T, M, L_m):
    r = max(d_nm*1e-9/2.0, 1e-12)
    Dk = (2.0/3.0)*r*np.sqrt((8.0*R*T)/(np.pi*M))
    return Dk/(L_m*R*T)

# 균형 조정된 스케일 (Sieving 약화, Surface/Solution 강화, Capillary 완화)
E_SIEVE = 6.0e3
DELTA_SOFT_A = 0.50

def pintr_sieving_SI(d_nm, gas, T, L_m, d_eff_A):
    pA = d_nm*10.0
    delta = d_eff_A - pA
    if delta > 0:
        return max(5e-7*np.exp(-(delta/0.5)**2)*np.exp(-E_SIEVE/(R*T)), 1e-14)
    x = max(1.0 - (d_eff_A/pA)**2, 0.0); f = x**2
    return max(5e-7*f*np.exp(-E_SIEVE/(R*T)), 1e-14)

def pintr_surface_SI(d_nm, gas, T, L_m, dqdp):
    Ds = 1e-10 * np.exp(-GAS_PARAMS[gas]["Ea_s"]/(R*T))
    return max((Ds/L_m) * (dqdp * 5e5), 0.0)

def pintr_solution_SI(gas, T, L_m, dqdp):
    Dsol = 3e-9*np.exp(-1.8e4/(R*T))/np.sqrt(GAS_PARAMS[gas]["M"]/1e-3)
    return max((Dsol/L_m) * (dqdp * 1e5), 0.0)

def pintr_capillary_SI(d_nm, rp, L_m):
    if not (d_nm >= 1.5 and rp > 0.4): return 0.0
    r_m = max(d_nm*1e-9/2.0, 1e-12)
    base = 1e-6*np.sqrt(r_m)/L_m
    return base * np.clip((rp - 0.4)/0.6, 0, 1)**0.5

def dsl_loading_and_slope_b(gas,T,Pbar,relP,q1,q2,b1,b2):
    P0=Pbar*1e5
    b1=max(float(b1),0.0); b2=max(float(b2),0.0)
    q1_molkg=q1*1e-3*1e3; q2_molkg=q2*1e-3*1e3
    q_vec=np.zeros_like(relP,float); dqdp=np.zeros_like(relP,float)
    for i,rp in enumerate(relP):
        P=max(float(rp),1e-9)*P0
        th1=(b1*P)/(1.0+b1*P); th2=(b2*P)/(1.0+b2*P)
        q_molkg=q1_molkg*th1 + q2_molkg*th2
        q_vec[i]=q_molkg/1e3
        dqdp[i]=(q1_molkg*b1)/(1.0+b1*P)**2 + (q2_molkg*b2)/(1.0+b2*P)**2
    return q_vec,dqdp

def intrinsic_permeances(gas,T,Pbar,d_nm,rp,L_nm,dqdp,alpha):
    L_m = max(float(L_nm), 1e-3) * 1e-9
    M   = GAS_PARAMS[gas]["M"]
    d_eff_A = effective_diameter_A(gas, T, alpha)
    Pi = {
        "Blocked":   1e-14,
        "Sieving":   pintr_sieving_SI(d_nm, gas, T, L_m, d_eff_A),
        "Knudsen":   pintr_knudsen_SI(d_nm, T, M, L_m),
        "Surface":   pintr_surface_SI(d_nm, gas, T, L_m, dqdp),
        "Capillary": pintr_capillary_SI(d_nm, rp, L_m),
        "Solution":  pintr_solution_SI(gas, T, L_m, dqdp),
    }
    return Pi

def weights_from_intrinsic(Pi_intr, gamma=0.8):
    keys = MECH_ORDER
    x=np.array([np.log(max(float(Pi_intr.get(k,0.0)),1e-30)) for k in keys])
    e=np.exp(gamma*x); s=float(e.sum())
    w=e/s if s>0 and np.isfinite(s) else np.array([0,0,0,1,0,0],float)
    return {k:float(w[i]) for i,k in enumerate(keys)}

def run_gas_membrane():
    st.header("Gas Membrane (balanced mechanisms)")

    with st.sidebar:
        T    = nudged_slider("Temperature", 10.0, 600.0, 1.0, 300.0, key="T_g",    unit="K")
        Pbar = nudged_slider("Total Pressure", 0.1, 10.0, 0.1, 1.0,  key="Pbar_g", unit="bar")
        d_nm = nudged_slider("Pore diameter", 0.05, 50.0, 0.01, 0.36, key="d_nm_g", unit="nm")
        L_nm = nudged_slider("Membrane thickness", 10.0, 100000.0, 1.0, 100.0, key="L_nm_g", unit="nm")
        gases = list(GAS_PARAMS.keys())
        gas1  = st.selectbox("Gas1 (numerator)",   gases, index=gases.index("CO2"), key="gas1_g")
        gas2  = st.selectbox("Gas2 (denominator)", gases, index=gases.index("CH4"), key="gas2_g")
        st.subheader("DSL parameters (mmol/g, Pa⁻¹)")
        q11 = nudged_slider("q1 Gas1", 0.0, 100.0, 0.01, 1.50, key="q11_g", unit="mmol/g")
        q12 = nudged_slider("q2 Gas1", 0.0, 100.0, 0.01, 0.50, key="q12_g", unit="mmol/g")
        b11 = nudged_slider("b1 Gas1", 1e-10, 1e-1, 1e-8, 1e-5, key="b11_g", unit="Pa⁻¹", decimals=8)
        b12 = nudged_slider("b2 Gas1", 1e-10, 1e-1, 1e-8, 5e-6, key="b12_g", unit="Pa⁻¹", decimals=8)
        q21 = nudged_slider("q1 Gas2", 0.0, 100.0, 0.01, 1.00, key="q21_g", unit="mmol/g")
        q22 = nudged_slider("q2 Gas2", 0.0, 100.0, 0.01, 0.40, key="q22_g", unit="mmol/g")
        b21 = nudged_slider("b1 Gas2", 1e-10, 1e-1, 1e-8, 5e-6, key="b21_g", unit="Pa⁻¹", decimals=8)
        b22 = nudged_slider("b2 Gas2", 1e-10, 1e-1, 1e-8, 1e-6, key="b22_g", unit="Pa⁻¹", decimals=8)
        st.subheader("Quantum correction")
        alpha = nudged_slider("α (manual)", 0.0, 0.6, 0.01, 0.0, key="alpha_g")

        rp_focus = nudged_slider("rₚ focus", 0.01, 0.99, 0.01, 0.35, key="rp_focus_g", unit="–")
        zoom_band = st.checkbox("Zoom band around rₚ focus", value=False, key="zoom_band_g")
        if st.session_state.get("zoom_band_g", False):
            zoom_half = nudged_slider("Zoom half-width Δrₚ", 0.01, 0.49, 0.01, 0.15, key="zoom_half_g", unit="–")
        else:
            zoom_half = 0.0

    # X-axis (relative pressure)
    if zoom_band:
        c = float(rp_focus); w = float(zoom_half)
        rmin, rmax = max(0.01, c - w), min(0.99, c + w)
        relP = np.linspace(rmin, rmax, 400)
    else:
        relP = np.linspace(0.01, 0.99, 500)

    q1_mg, dqdp1 = dsl_loading_and_slope_b(gas1, T, Pbar, relP, q11, q12, b11, b12)
    q2_mg, dqdp2 = dsl_loading_and_slope_b(gas2, T, Pbar, relP, q21, q22, b21, b22)

    Pi1 = np.zeros_like(relP, float)
    Pi2 = np.zeros_like(relP, float)
    winners = []
    for i, rp in enumerate(relP):
        A = intrinsic_permeances(gas1, T, Pbar, d_nm, rp, L_nm, float(dqdp1[i]), alpha)
        B = intrinsic_permeances(gas2, T, Pbar, d_nm, rp, L_nm, float(dqdp2[i]), alpha)
        wA = weights_from_intrinsic(A); wB = weights_from_intrinsic(B)
        Pi1[i] = sum(wA[k]*A[k] for k in ("Sieving","Knudsen","Surface","Capillary","Solution"))
        Pi2[i] = sum(wB[k]*B[k] for k in ("Sieving","Knudsen","Surface","Capillary","Solution"))
        winners.append(max(A, key=A.get))
    rgba = np.array([to_rgba(MECH_COLOR[n]) for n in winners])[None, :, :]

    Sel     = np.divide(Pi1, Pi2, out=np.zeros_like(Pi1), where=(Pi2 > 0))
    Pi1_gpu = Pi1 / GPU_UNIT
    Pi2_gpu = Pi2 / GPU_UNIT

    # Plot
    figBand, axBand = plt.subplots(figsize=(9, 0.7))
    axBand.imshow(rgba, extent=(relP[0], relP[-1], 0, 1), aspect="auto", origin="lower")
    axBand.set_xlim(relP[0], relP[-1]); axBand.set_yticks([])
    axBand.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    handles = [plt.Rectangle((0,0),1,1,fc=MECH_COLOR[n], ec='none', label=n) for n in MECH_ORDER]
    leg = axBand.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.7),
                        ncol=6, frameon=True)
    leg.get_frame().set_alpha(0.85); leg.get_frame().set_facecolor("white")
    st.pyplot(figBand, use_container_width=True); plt.close(figBand)

    fig1, ax1 = plt.subplots(figsize=(9,3))
    ax1.plot(relP, Pi1_gpu, label=f"{gas1}")
    ax1.plot(relP, Pi2_gpu, "--", label=f"{gas2}")
    ax1.set_xlabel(r"$P/P_0$"); ax1.set_ylabel(r"$\Pi$ (GPU)")
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1, use_container_width=True); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9,3))
    ax2.plot(relP, Sel, label=f"{gas1}/{gas2}")
    ax2.set_xlabel(r"$P/P_0$"); ax2.set_ylabel("Selectivity (–)")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

# =====================================================================
# MODE 3 — DRUG IN VESSEL (1D ADR transient)
# =====================================================================
DRUG_DB = {
    "Caffeine":      {"Db": 6.5e-10, "Pv": 2.0e-6, "kel": 1.0e-4},
    "Acetaminophen": {"Db": 6.0e-10, "Pv": 1.5e-6, "kel": 8.0e-5},
    "Dopamine":      {"Db": 6.0e-10, "Pv": 1.2e-6, "kel": 1.2e-4},
    "Glucose":       {"Db": 6.7e-10, "Pv": 1.0e-6, "kel": 7.0e-5},
}

def run_vascular_drug():
    st.header("Drug Transport in a Vessel (1D ADR, transient)")

    with st.sidebar:
        Rv_um = nudged_slider("Vessel radius", 2.0, 100.0, 1.0, 4.0, key="Rv_um_v", unit="μm")
        L_mm  = nudged_slider("Segment length", 1.0, 200.0, 1.0, 20.0, key="Lv_mm_v", unit="mm")
        U     = nudged_slider("Mean velocity U", 0.1e-3, 20e-3, 0.1e-3, 1.0e-3, key="U_v", unit="m/s")
        drug  = st.selectbox("Drug", list(DRUG_DB.keys()), index=0, key="drug_v")

        T_body = nudged_slider("Temperature", 293.15, 312.15, 0.5, 310.15, key="Tdrug_v", unit="K")
        auto_Db = st.checkbox("Auto-compute D_b (T/η)", value=True, key="autoDb_v")
        if auto_Db:
            T_ref = 298.15
            eta_ref = eta_water_PaS(T_ref)
            eta_now = eta_water_PaS(T_body)
            Db = D_temp_correction(DRUG_DB[drug]["Db"], T_ref, T_body, eta_ref, eta_now)
            st.caption(f"D_b auto: {Db:.3e} m²/s")
        else:
            Db = log_slider("D_b",  -12.0, -8.0, 0.1, np.log10(DRUG_DB[drug]["Db"]), key="Db_v",  unit="m²/s")

        Pv   = log_slider("P_v",   -9.0,  -5.0, 0.1, np.log10(DRUG_DB[drug]["Pv"]), key="Pv_v",  unit="m/s")
        kel  = log_slider("k_elim",-6.0,  -2.0, 0.1, np.log10(DRUG_DB[drug]["kel"]), key="kel_v", unit="s⁻¹")

        C0   = nudged_slider("Reference conc. C₀", 0.0, 5.0, 0.01, 1.0, key="C0_v", unit="mol/m³")
        profile = st.selectbox("Inlet profile", ["Bolus (Gaussian)","Constant infusion"], index=0, key="pulse_v")
        t_end = nudged_slider("Sim time", 0.1, 600.0, 0.1, 60.0, key="tend_v", unit="s")
        dt    = nudged_slider("Δt", 1e-3, 0.5, 1e-3, 0.01, key="dt_v", unit="s")
        Nx    = nudged_int("Grid Nx", 50, 600, 10, 200, key="Nx_v")

    Rv = Rv_um*1e-6
    L  = L_mm*1e-3
    x  = np.linspace(0, L, Nx); dx = x[1]-x[0]
    t  = np.arange(0.0, t_end+dt, dt)

    Pe_r = U*Rv/Db
    Deff = Db*(1.0 + (Pe_r**2)/192.0)
    k_leak = 2.0*Pv/max(Rv,1e-12)

    if profile == "Bolus (Gaussian)":
        t0 = 0.2*t_end; sig = max(0.05*t_end, 1e-6)
        Cin_t = C0*np.exp(-((t-t0)**2)/(2*sig**2))
    else:
        Cin_t = C0*np.ones_like(t)

    C = np.zeros((len(t), Nx), dtype=float)
    lam_a = U*dt/dx
    lam_d = Deff*dt/(dx*dx)
    if lam_a>1.0: st.warning(f"Advection CFL>1 (UΔt/Δx={lam_a:.2f}). Reduce Δt or increase Nx.")
    if lam_d>0.5: st.warning(f"Diffusion number>0.5 (D_effΔt/Δx²={lam_d:.2f}). Use smaller Δt.")

    for n in range(1, len(t)):
        Cn = C[n-1,:].copy()
        Cnp = Cn.copy()
        Cnp[0] = Cin_t[n]
        adv = -lam_a*(Cn[1:] - Cn[:-1])
        dif = lam_d*(np.roll(Cn,-1)[1:-1] - 2*Cn[1:-1] + Cn[0:-2])
        react = -dt*(k_leak + kel)*Cn[1:-1]
        Cnp[1:-1] = Cn[1:-1] + adv[:-1] + dif + react
        Cnp[-1] = Cnp[-2]
        C[n,:] = np.maximum(Cnp, 0.0)

    Cout_t = C[:, -1]

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Key numbers")
        st.metric("Pe_r = U·R/D_b", f"{Pe_r:.2f}")
        st.metric("D_eff", f"{Deff:.3e} m²/s")
        st.metric("k_leak = 2P_v/R", f"{k_leak:.3e} s⁻¹")
    with col2:
        fig, ax = plt.subplots(figsize=(8,3))
        im = ax.imshow(C.T, aspect='auto', origin='lower',
                       extent=(t[0], t[-1], x[0]*1e3, x[-1]*1e3))
        ax.set_xlabel("t (s)"); ax.set_ylabel("x (mm)")
        cb = plt.colorbar(im, ax=ax); cb.set_label("C (mol/m³)")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(t, Cout_t, label="Outlet")
        ax2.set_xlabel("t (s)"); ax2.set_ylabel("C_out (mol/m³)")
        ax2.grid(True); ax2.legend()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)

# =====================================================================
# App shell — Mode selection
# =====================================================================
st.title("Unified Transport Simulators (Gas / Ion / Vascular)")
mode_main = st.sidebar.radio("Select simulation",
                             ["Gas membrane", "Ion membrane (NP + Donnan)", "Drug in vessel"],
                             index=1)

if mode_main == "Gas membrane":
    run_gas_membrane()
elif mode_main == "Ion membrane (NP + Donnan)":
    run_ion_membrane()
else:
    run_vascular_drug()
