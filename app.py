# app.py
# Streamlit web app: Permeance (SI units), Selectivity, Mechanism band
# Mechanisms: Blocked / Sieving / Knudsen / Surface / Capillary / Solution
# Adsorption: Double-Site Langmuir (DSL) with Qst (kJ/mol) & q (mmol/g)
# Author: you + ChatGPT

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import to_rgba

# ---------------------------- Physical constants ----------------------------
R = 8.314  # J/mol/K

# ---------------------------- Model global knobs ----------------------------
# Effective solid density to convert [mol/kg] -> [mol/m^3] (surface/solution terms)
RHO_EFF = 500.0  # kg/m^3  (tunable; represents accessible sorbent per unit membrane volume)
# Surface diffusion prefactor
D0_SURF = 1e-9   # m^2/s   (typical order for surface hopping)
# Solution diffusion prefactor
D0_SOL  = 1e-10  # m^2/s   (dense polymer-like)
E_D_SOL = 1.8e4  # J/mol   (solution diffusion activation)
# Capillary permeance scale (empirical SI proxy): Pi_cap ~ K_cap * sqrt(r[m]) / L
K_CAP   = 1e-7   # (mol m^-1.5 s^-1 Pa^-1)  tuned for visualization in SI scale
# Sieving barrier (Arrhenius-like) when pore ~ molecule
E_SIEVE = 9.0e3  # J/mol
# Tiny permeance used for "Blocked" instead of perfect zero (more physical)
PI_TINY = 1e-14  # mol m^-2 s^-1 Pa^-1
# When pore is truly nanodense (very small), favor solution-diffusion
SOL_TH_NM = 0.30  # nm: below this, treat as dense/solution-diffusion dominated

# ---------------------------- Gas parameters ----------------------------
# M [kg/mol], kinetic diameter d [Å], surface Ea_s for Ds Arrhenius [J/mol]
PARAMS = {
    "H2":  {"M":2.016e-3,  "d":2.89, "Ea_s":8.0e3},
    "D2":  {"M":4.028e-3,  "d":2.89, "Ea_s":8.5e3},
    "He":  {"M":4.003e-3,  "d":2.60, "Ea_s":6.0e3},
    "N2":  {"M":28.0134e-3,"d":3.64, "Ea_s":9.0e3},
    "O2":  {"M":31.998e-3, "d":3.46, "Ea_s":9.0e3},
    "CO2": {"M":44.01e-3,  "d":3.30, "Ea_s":9.5e3},
    "CH4": {"M":16.043e-3, "d":3.80, "Ea_s":9.5e3},
    "C2H6":{"M":30.070e-3, "d":4.44, "Ea_s":1.0e4},
    "C3H6":{"M":42.081e-3, "d":4.00, "Ea_s":1.05e4},
    "C3H8":{"M":44.097e-3, "d":4.30, "Ea_s":1.05e4},
}

# ---------------------------- DSL adsorption ----------------------------
def dsl_loading_and_slope(gas, T, P_bar, relP_vec, q1_mmolg, q2_mmolg, Qst1_kJ, Qst2_kJ):
    """
    Returns:
      q_vec   [mmol/g]
      dqdp    [mol/kg/Pa]  (slope wrt pressure)
    """
    par = PARAMS[gas]
    P0 = P_bar * 1e5  # Pa
    # DSL affinity coefficients (b) with Qst; b0 chosen as 1e-4 Pa^-1 baseline
    b0 = 1e-4
    b1 = b0 * np.exp(max(Qst1_kJ,0.0)*1e3/(R*T))
    b2 = b0 * np.exp(max(Qst2_kJ,0.0)*1e3/(R*T))

    q_vec  = np.zeros_like(relP_vec, float)
    dqdp_MK = np.zeros_like(relP_vec, float)  # mol/kg/Pa

    # Convert capacity from mmol/g -> mol/kg
    q1_molkg = q1_mmolg * 1e-3 * 1e3
    q2_molkg = q2_mmolg * 1e-3 * 1e3

    for i, rp in enumerate(relP_vec):
        P = max(rp, 1e-9) * P0
        th1 = (b1*P)/(1.0 + b1*P)
        th2 = (b2*P)/(1.0 + b2*P)
        q_molkg = q1_molkg*th1 + q2_molkg*th2
        q_vec[i] = q_molkg / 1e3  # -> mmol/g

        # derivative wrt P: dq/dP = q1*b1/(1+b1P)^2 + q2*b2/(1+b2P)^2  (mol/kg/Pa)
        dqdp_MK[i] = (q1_molkg * b1)/(1.0 + b1*P)**2 + (q2_molkg * b2)/(1.0 + b2*P)**2

    return q_vec, dqdp_MK

# ---------------------------- Mechanisms & permeance (SI) ----------------------------
MECH_ORDER = ["Blocked", "Sieving", "Knudsen", "Surface", "Capillary", "Solution"]
MECH_COLOR = {
    "Blocked":  "#bdbdbd",
    "Sieving":  "#1f78b4",
    "Knudsen":  "#33a02c",
    "Surface":  "#e31a1c",
    "Capillary":"#ff7f00",
    "Solution": "#6a3d9a",
}

def mean_free_path_nm(T, P_bar, d_ang):
    kB = 1.380649e-23; P = P_bar*1e5; d = d_ang*1e-10
    return (kB*T/(np.sqrt(2)*np.pi*d*d*P))*1e9

def classify_mechanism(pore_d_nm, gas1, gas2, T, P_bar, rp):
    if pore_d_nm < SOL_TH_NM:
        return "Solution"
    d1, d2 = PARAMS[gas1]["d"], PARAMS[gas2]["d"]
    lam = mean_free_path_nm(T, P_bar, 0.5*(d1+d2))
    if pore_d_nm >= 2.0 and rp > 0.5:
        return "Capillary"
    if pore_d_nm <= 2.0:
        p_eff_A = pore_d_nm*10.0 + 0.8  # Å (allowance)
        return "Blocked" if p_eff_A <= min(d1,d2) else "Sieving"
    if pore_d_nm < 0.5*lam:
        return "Knudsen"
    return "Surface"

def pintr_knudsen_SI(pore_d_nm, T, M, L_m):
    r = max(pore_d_nm*1e-9/2.0, 1e-12)
    Dk = (2.0/3.0) * r * np.sqrt((8.0*R*T)/(np.pi*M))      # [m^2/s]
    Pi = Dk / (L_m * R * T)                                # [mol m^-2 s^-1 Pa^-1]
    return Pi

def pintr_sieving_SI(pore_d_nm, gas, T, L_m):
    # Geometric opening factor + small Arrhenius
    dA = PARAMS[gas]["d"] # Å
    pA = pore_d_nm*10.0 + 0.8
    if pA <= dA:
        return PI_TINY
    x = max(1.0 - (dA/pA)**2, 0.0)
    f_open = x**2
    Pi_ref = 5e-9  # baseline [mol m^-2 s^-1 Pa^-1] for near-threshold opening (empirical)
    Pi = Pi_ref * f_open * np.exp(-E_SIEVE/(R*T))
    return max(Pi, PI_TINY)

def pintr_surface_SI(pore_d_nm, gas, T, L_m, dqdp_molkgPa):
    # Pi_surf ≈ (D_s/L) * (dq/dp) * rho_eff
    Ds = D0_SURF * np.exp(-PARAMS[gas]["Ea_s"]/(R*T))
    Pi = (Ds / L_m) * (dqdp_molkgPa * RHO_EFF)             # [mol m^-2 s^-1 Pa^-1]
    return max(Pi, 0.0)

def pintr_capillary_SI(pore_d_nm, rp, L_m):
    # Simple proxy: Pi_cap ~ K_cap * sqrt(r)/L if above Kelvin-like threshold
    r_m = max(pore_d_nm*1e-9/2.0, 1e-12)
    thresh = np.exp(-120.0/( (pore_d_nm/2.0)*rp*300.0 + 1e-12 ))  # heuristic; rp in 0-1
    if rp <= thresh:
        return 0.0
    return K_CAP * np.sqrt(r_m) / L_m

def pintr_solution_SI(gas, T, L_m, dqdp_molkgPa):
    # Pi_sol = (D_sol * S) / L ; S ~ dq/dp * rho_eff (mol/m^3/Pa)
    Dsol = D0_SOL * np.exp(-E_D_SOL/(R*T)) / np.sqrt(PARAMS[gas]["M"]/1e-3)  # /sqrt(g/mol) scaling
    Pi = (Dsol / L_m) * (dqdp_molkgPa * RHO_EFF)
    return max(Pi, 0.0)

def permeance_series_SI(pore_d_nm, gas, other, T, P_bar, relP, L_nm,
                        q_mmolg, dqdp_molkgPa, q_other_mmolg):
    """
    Returns Pi[relP] in SI units [mol m^-2 s^-1 Pa^-1]
    Includes competitive weighting via theta_eff = q_i/(q_i+q_j)
    """
    L_m = max(L_nm, 1e-3) * 1e-9  # nm -> m
    M = PARAMS[gas]["M"]
    Pi = np.zeros_like(relP, float)

    for i, rp in enumerate(relP):
        rule = classify_mechanism(pore_d_nm, gas, other, T, P_bar, rp)

        if   rule == "Blocked":
            Pi0 = PI_TINY
        elif rule == "Sieving":
            Pi0 = pintr_sieving_SI(pore_d_nm, gas, T, L_m)
        elif rule == "Knudsen":
            Pi0 = pintr_knudsen_SI(pore_d_nm, T, M, L_m)
        elif rule == "Surface":
            Pi0 = pintr_surface_SI(pore_d_nm, gas, T, L_m, dqdp_molkgPa[i])
        elif rule == "Capillary":
            Pi0 = pintr_capillary_SI(pore_d_nm, rp, L_m)
        elif rule == "Solution":
            Pi0 = pintr_solution_SI(gas, T, L_m, dqdp_molkgPa[i])
        else:
            Pi0 = PI_TINY

        # competitive weighting (IAST-like): adsorption preference at given P
        qi = q_mmolg[i]; qj = q_other_mmolg[i]
        theta = (qi/(qi+qj)) if (qi+qj) > 0 else 0.0
        Pi[i] = Pi0 * theta

    return Pi

def mechanism_band_rgba(g1, g2, T, P_bar, d_nm, relP):
    names = [classify_mechanism(d_nm, g1, g2, T, P_bar, r) for r in relP]
    rgba = np.array([to_rgba(MECH_COLOR[n]) for n in names])[None, :, :]
    return rgba, names

# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="Membrane Permeance (SI)", layout="wide")
st.title("Membrane Transport Simulator (SI units)")

with st.sidebar:
    st.header("Global Conditions")
    T = st.slider("Temperature T (K)", 10.0, 600.0, 300.0, 1.0)
    P_bar = st.slider("Total Pressure P (bar)", 0.1, 10.0, 1.0, 0.1)
    d_nm = st.slider("Pore diameter d (nm)", 0.01, 50.0, 0.34, 0.01)
    L_nm = st.slider("Membrane thickness L (nm)", 10.0, 1000.0, 100.0, 1.0)

    gases = list(PARAMS.keys())
    gas1 = st.selectbox("Gas1 (numerator)", gases, index=gases.index("C3H6"))
    gas2 = st.selectbox("Gas2 (denominator)", gases, index=gases.index("C3H8"))

    st.header("DSL parameters (each gas)")
    st.caption("Qst: 0–100 kJ/mol, q: 0–5 mmol/g (double site)")

    st.subheader("Gas1")
    Q11 = st.slider("Qst1 Gas1 (kJ/mol)", 0.0, 100.0, 27.0, 0.1)
    Q12 = st.slider("Qst2 Gas1 (kJ/mol)", 0.0, 100.0, 18.0, 0.1)
    q11 = st.slider("q1 Gas1 (mmol/g)",   0.0, 5.0,   0.70, 0.01)
    q12 = st.slider("q2 Gas1 (mmol/g)",   0.0, 5.0,   0.30, 0.01)

    st.subheader("Gas2")
    Q21 = st.slider("Qst1 Gas2 (kJ/mol)", 0.0, 100.0, 26.5, 0.1)
    Q22 = st.slider("Qst2 Gas2 (kJ/mol)", 0.0, 100.0, 17.0, 0.1)
    q21 = st.slider("q1 Gas2 (mmol/g)",   0.0, 5.0,   0.70, 0.01)
    q22 = st.slider("q2 Gas2 (mmol/g)",   0.0, 5.0,   0.30, 0.01)

# Calculation
relP = np.linspace(0.01, 0.99, 500)

q1_mmolg, dqdp1 = dsl_loading_and_slope(gas1, T, P_bar, relP, q11, q12, Q11, Q12)
q2_mmolg, dqdp2 = dsl_loading_and_slope(gas2, T, P_bar, relP, q21, q22, Q21, Q22)

Pi1 = permeance_series_SI(d_nm, gas1, gas2, T, P_bar, relP, L_nm, q1_mmolg, dqdp1, q2_mmolg)
Pi2 = permeance_series_SI(d_nm, gas2, gas1, T, P_bar, relP, L_nm, q2_mmolg, dqdp2, q1_mmolg)
Sel = np.divide(Pi1, Pi2, out=np.zeros_like(Pi1), where=(Pi2>0))

# Mechanism band
rgba, mech_names = mechanism_band_rgba(gas1, gas2, T, P_bar, d_nm, relP)

# ---------------------------- Layout: Plots ----------------------------
colA, colB = st.columns([1,2])

with colB:
    st.subheader("Mechanism map (along relative pressure)")
    figBand, axBand = plt.subplots(figsize=(9, 0.7))
    axBand.imshow(rgba, extent=(0,1,0,1), aspect='auto', origin='lower')
    axBand.set_yticks([])
    axBand.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
    axBand.set_xlim(0,1)
    # Legend
    handles = [plt.Rectangle((0,0),1,1, fc=MECH_COLOR[n], ec='none', label=n) for n in MECH_ORDER]
    leg = axBand.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,-0.7),
                        ncol=6, frameon=True)
    leg.get_frame().set_alpha(0.85); leg.get_frame().set_facecolor("white")
    st.pyplot(figBand, use_container_width=True); plt.close(figBand)

    st.subheader("Permeance (SI)")
    fig1, ax1 = plt.subplots(figsize=(9,3))
    ax1.plot(relP, Pi1, label=f"{gas1}")
    ax1.plot(relP, Pi2, '--', label=f"{gas2}")
    ax1.set_ylabel(r"$\Pi$  (mol m$^{-2}$ s$^{-1}$ Pa$^{-1}$)")
    ax1.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1, use_container_width=True); plt.close(fig1)

    st.subheader("Selectivity")
    fig2, ax2 = plt.subplots(figsize=(9,3))
    ax2.plot(relP, Sel, label=f"{gas1}/{gas2}")
    ax2.set_ylabel("Selectivity (–)")
    ax2.set_xlabel(r"Relative pressure, $P/P_0$ (–)")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2, use_container_width=True); plt.close(fig2)

with colA:
    st.subheader("Mechanism (rule) vs intrinsic (Gas1)")
    rp_mid = relP[len(relP)//2]
    # Intrinsic comparison at mid P/P0 for Gas1
    L_m = L_nm*1e-9
    M1 = PARAMS[gas1]["M"]
    # mid-point slopes for solution/surface
    dqdp1_mid = dqdp1[len(relP)//2]
    # candidates (without competitive weighting)
    cand = {
        "Blocked":  PI_TINY,
        "Sieving":  pintr_sieving_SI(d_nm, gas1, T, L_m),
        "Knudsen":  pintr_knudsen_SI(d_nm, T, M1, L_m),
        "Surface":  pintr_surface_SI(d_nm, gas1, T, L_m, dqdp1_mid),
        "Capillary":pintr_capillary_SI(d_nm, rp_mid, L_m),
        "Solution": pintr_solution_SI(gas1, T, L_m, dqdp1_mid),
    }
    best_intrinsic = max(cand, key=cand.get)
    rule_mech = classify_mechanism(d_nm, gas1, gas2, T, P_bar, rp_mid)

    st.markdown(
        f"**Mechanism (rule):** `{rule_mech}` &nbsp;&nbsp; | &nbsp;&nbsp;"
        f"**Best intrinsic (Gas1):** `{best_intrinsic}`"
    )
    st.caption(
        "Intrinsic permeance proxies at mid $P/P_0$ (no competition):  \n"
        + "  • " + "  \n  • ".join([f"{k}: {v:.3e} mol m⁻² s⁻¹ Pa⁻¹" for k,v in cand.items()])
    )

st.markdown("---")
st.caption(
    "Notes: (1) All permeance values are in SI units mol·m⁻²·s⁻¹·Pa⁻¹. "
    "(2) Capillary & Sieving terms use calibrated proxies in SI scale for visualization. "
    "(3) Surface/Solution use DSL slope (∂q/∂p) to convert sorption response into flux capacity."
)
