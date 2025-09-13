import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.colors import to_rgba

st.set_page_config(page_title="Membrane Mechanisms Simulator", layout="wide")

R = 8.314  # J/mol-K
THICK_M = 100e-9
FLEX_A  = 0.8     # Å
SOL_TH_NM = 0.30  # nm (solution-diffusion 우선 임계)
E_D_SOL   = 18000.0  # J/mol

PARAMS = {
    "H2":  {"M":2.016,  "d":2.89, "b0":1e-4, "Ea_s":8000,  "Ccap":0.10},
    "D2":  {"M":4.028,  "d":2.89, "b0":1e-4, "Ea_s":8500,  "Ccap":0.10},
    "He":  {"M":4.003,  "d":2.60, "b0":1e-4, "Ea_s":6000,  "Ccap":0.05},
    "N2":  {"M":28.0134,"d":3.64, "b0":1e-4, "Ea_s":9000,  "Ccap":0.20},
    "O2":  {"M":31.998, "d":3.46, "b0":1e-4, "Ea_s":9000,  "Ccap":0.20},
    "CO2": {"M":44.01,  "d":3.30, "b0":1e-4, "Ea_s":9500,  "Ccap":0.60},
    "CH4": {"M":16.043, "d":3.80, "b0":1e-4, "Ea_s":9500,  "Ccap":0.50},
    "C2H6":{"M":30.070, "d":4.44, "b0":1e-4, "Ea_s":10000, "Ccap":0.70},
    "C3H6":{"M":42.081, "d":4.00, "b0":1e-4, "Ea_s":10500, "Ccap":0.80},
    "C3H8":{"M":44.097, "d":4.30, "b0":1e-4, "Ea_s":10500, "Ccap":0.85},
}

MECH_ORDER = ["Blocked", "Sieving", "Knudsen", "Surface", "Capillary", "Solution"]
MECH_COLOR = {
    "Blocked":  "#bdbdbd",
    "Sieving":  "#1f78b4",
    "Knudsen":  "#33a02c",
    "Surface":  "#e31a1c",
    "Capillary":"#ff7f00",
    "Solution": "#6a3d9a",
}

def dsl_loading_series(gas, T, P_bar, relP_vec, Qst1_kJ, Qst2_kJ, q1_mmolg, q2_mmolg):
    par = PARAMS[gas]
    P0 = P_bar * 1e5
    Q1 = max(Qst1_kJ, 0.0) * 1e3
    Q2 = max(Qst2_kJ, 0.0) * 1e3
    b1 = par["b0"] * np.exp(Q1/(R*T))
    b2 = par["b0"] * np.exp(Q2/(R*T))
    out = np.zeros_like(relP_vec, float)
    for i, rp in enumerate(relP_vec):
        P = max(rp, 1e-9) * P0
        th1 = (b1*P)/(1.0 + b1*P)
        th2 = (b2*P)/(1.0 + b2*P)
        out[i] = q1_mmolg*th1 + q2_mmolg*th2
    return out

def mean_free_path_nm(T, P_bar, d_ang):
    kB = 1.380649e-23; P = P_bar*1e5; d = d_ang*1e-10
    return (kB*T/(np.sqrt(2)*np.pi*d*d*P))*1e9

def intrinsic_knudsen(pore_d_nm, M_gmol):
    return np.maximum(pore_d_nm, 1e-9) / np.sqrt(M_gmol)

def intrinsic_surface(pore_d_nm, gas, T):
    Ds = np.exp(-PARAMS[gas]["Ea_s"]/(R*T))
    return Ds * np.maximum(1.0/np.maximum(pore_d_nm,1e-9), 1e-9)

def intrinsic_sieving(pore_d_nm, d_ang, T):
    p_eff = pore_d_nm*10.0 + FLEX_A
    if p_eff <= d_ang:
        return np.exp(-20000.0/(R*T)) * 1e-6
    x = 1.0 - (d_ang/p_eff)**2
    return max(x,0.0)**2 * np.exp(-9000.0/(R*T))

def intrinsic_capillary(pore_d_nm, gas, T, relP):
    r_nm = max(pore_d_nm/2.0, 1e-9)
    thresh = np.exp(-120.0/(r_nm*T))
    return 5.0*np.sqrt(r_nm) if relP > thresh else 0.0

def intrinsic_solution(gas, T, q_mmolg):
    D = np.exp(-E_D_SOL/(R*T)) / np.sqrt(PARAMS[gas]["M"])
    return D * max(q_mmolg, 0.0)

def classify_mechanism(pore_d_nm, gas1, gas2, T, P_bar, relP):
    if pore_d_nm < SOL_TH_NM:
        return "Solution"
    d1, d2 = PARAMS[gas1]["d"], PARAMS[gas2]["d"]
    lam = mean_free_path_nm(T, P_bar, 0.5*(d1+d2))
    if pore_d_nm >= 2.0 and relP > 0.5:
        return "Capillary"
    if pore_d_nm <= 2.0:
        p_eff = pore_d_nm*10.0 + FLEX_A
        return "Blocked" if p_eff <= min(d1,d2) else "Sieving"
    if pore_d_nm < 0.5*lam:
        return "Knudsen"
    return "Surface"

def best_matching_mechanism(pore_d_nm, gas, T, P_bar, relP, q_mid=1.0):
    M, d_ang = PARAMS[gas]["M"], PARAMS[gas]["d"]
    cand = {
        "Knudsen":  intrinsic_knudsen(pore_d_nm, M),
        "Surface":  intrinsic_surface(pore_d_nm, gas, T),
        "Sieving":  intrinsic_sieving(pore_d_nm, d_ang, T),
        "Capillary":intrinsic_capillary(pore_d_nm, gas, T, relP),
        "Solution": intrinsic_solution(gas, T, q_mid),
        "Blocked":  0.0
    }
    return max(cand, key=cand.get), cand

def permeance_series(pore_d_nm, gas, other, T, P_bar, relP_vec, q_self, q_other):
    M, d_ang = PARAMS[gas]["M"], PARAMS[gas]["d"]
    perm = np.zeros_like(relP_vec, float)
    for i, rp in enumerate(relP_vec):
        rule = classify_mechanism(pore_d_nm, gas, other, T, P_bar, rp)
        if   rule == "Blocked":   Pintr = 0.0
        elif rule == "Sieving":   Pintr = intrinsic_sieving(pore_d_nm, d_ang, T)
        elif rule == "Knudsen":   Pintr = intrinsic_knudsen(pore_d_nm, M)
        elif rule == "Capillary": Pintr = intrinsic_capillary(pore_d_nm, gas, T, rp)
        elif rule == "Solution":  Pintr = intrinsic_solution(gas, T, q_self[i])
        else:                     Pintr = intrinsic_surface(pore_d_nm, gas, T)
        qi, qj = q_self[i], q_other[i]
        theta_eff = qi/(qi+qj) if (qi+qj) > 0 else 0.0
        perm[i] = (Pintr * theta_eff) / THICK_M
    return perm

def mechanism_rgba_row(g1, g2, T, P_bar, d_nm, relP):
    names = [classify_mechanism(d_nm, g1, g2, T, P_bar, rp) for rp in relP]
    return np.array([to_rgba(MECH_COLOR[n]) for n in names])[None, :, :], names

# --------------------------- UI ---------------------------
st.title("Membrane Transport Mechanisms – Web Simulator")

left, right = st.columns([1, 2])

with left:
    st.subheader("Global Conditions")
    T = st.slider("Temperature (K)", 10.0, 600.0, 300.0, 1.0)
    P_bar = st.slider("Total Pressure (bar)", 0.1, 10.0, 1.0, 0.1)
    d_nm = st.slider("Pore diameter (nm)", 0.01, 50.0, 0.34, 0.01)

    gases = list(PARAMS.keys())
    gas1 = st.selectbox("Gas1 (numerator)", gases, index=gases.index("C3H6"))
    gas2 = st.selectbox("Gas2 (denominator)", gases, index=gases.index("C3H8"))

    st.subheader("DSL Adsorption Params")
    st.caption("Qst: 0–100 kJ/mol, q: 0–5 mmol/g")
    Q11 = st.slider("Qst1 Gas1 (kJ/mol)", 0.0, 100.0, 27.0, 0.1)
    Q12 = st.slider("Qst2 Gas1 (kJ/mol)", 0.0, 100.0, 18.0, 0.1)
    q11 = st.slider("q1 Gas1 (mmol/g)", 0.0, 5.0, 0.70, 0.01)
    q12 = st.slider("q2 Gas1 (mmol/g)", 0.0, 5.0, 0.30, 0.01)

    Q21 = st.slider("Qst1 Gas2 (kJ/mol)", 0.0, 100.0, 26.5, 0.1)
    Q22 = st.slider("Qst2 Gas2 (kJ/mol)", 0.0, 100.0, 17.0, 0.1)
    q21 = st.slider("q1 Gas2 (mmol/g)", 0.0, 5.0, 0.70, 0.01)
    q22 = st.slider("q2 Gas2 (mmol/g)", 0.0, 5.0, 0.30, 0.01)

with right:
    st.subheader("Results")

    relP = np.linspace(0.01, 0.99, 400)
    A1 = dsl_loading_series(gas1, T, P_bar, relP, Q11, Q12, q11, q12)
    A2 = dsl_loading_series(gas2, T, P_bar, relP, Q21, Q22, q21, q22)
    P1 = permeance_series(d_nm, gas1, gas2, T, P_bar, relP, A1, A2)
    P2 = permeance_series(d_nm, gas2, gas1, T, P_bar, relP, A2, A1)
    S  = np.divide(P1, P2, out=np.zeros_like(P1), where=(P2>0))

    # Mechanism band
    rgba, names = mechanism_rgba_row(gas1, gas2, T, P_bar, d_nm, relP)
    figBand, axBand = plt.subplots(figsize=(8, 0.7))
    axBand.imshow(rgba, extent=(0,1,0,1), aspect='auto', origin='lower')
    axBand.set_yticks([])
    axBand.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
    axBand.set_xlim(0,1)
    # 범례
    handles = [plt.Rectangle((0,0),1,1, fc=MECH_COLOR[n], ec='none', label=n) for n in MECH_ORDER]
    leg = axBand.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,-0.7), ncol=6, frameon=True)
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_facecolor("white")
    st.pyplot(figBand, use_container_width=True); plt.close(figBand)

    # Permeance plot
    fig1, ax1 = plt.subplots(figsize=(8,3))
    ax1.plot(relP, P1, label=f"Permeance {gas1}")
    ax1.plot(relP, P2, '--', label=f"Permeance {gas2}")
    ax1.set_ylabel(r"$\Pi$ (mol m$^{-2}$ s$^{-1}$ Pa$^{-1}$)")
    ax1.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1, use_container_width=True); plt.close(fig1)

    # Selectivity plot
    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(relP, S, label=f"Selectivity {gas1}/{gas2}")
    ax2.set_ylabel("Selectivity (–)")
    ax2.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    # Mechanism info (rule vs intrinsic)
    rp_mid = relP[len(relP)//2]
    rule = classify_mechanism(d_nm, gas1, gas2, T, P_bar, rp_mid)
    best, cand = best_matching_mechanism(d_nm, gas1, T, P_bar, rp_mid, A1[len(relP)//2])

    st.markdown(
        f"**Mechanism (rule):** {rule}  &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Best intrinsic (Gas1):** {best}  \n"
        f"Knudsen: {cand['Knudsen']:.3e} • Surface: {cand['Surface']:.3e} • "
        f"Sieving: {cand['Sieving']:.3e} • Capillary: {cand['Capillary']:.3e} • "
        f"Solution: {cand['Solution']:.3e}"
    )
