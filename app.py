# app.py — Membrane mechanism simulator (Streamlit, previous layout)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import streamlit as st

# ---------------- Constants ----------------
R = 8.314
FLEX_A = 0.8           # A
THICK_M = 100e-9       # m (arbitrary scale)
E_D_SOL = 1.8e4        # J/mol
E_D_SRF = 9.0e3        # J/mol

GASES = {
    "H2":  {"M": 2.016,  "d": 2.89, "cap": 0.10},
    "D2":  {"M": 4.028,  "d": 2.89, "cap": 0.10},
    "He":  {"M": 4.003,  "d": 2.60, "cap": 0.08},
    "N2":  {"M": 28.013, "d": 3.64, "cap": 0.20},
    "O2":  {"M": 31.998, "d": 3.46, "cap": 0.20},
    "CO2": {"M": 44.01,  "d": 3.30, "cap": 0.60},
    "CH4": {"M": 16.043, "d": 3.80, "cap": 0.50},
    "C2H6":{"M": 30.070, "d": 4.44, "cap": 0.70},
    "C3H6":{"M": 42.081, "d": 4.00, "cap": 0.80},
    "C3H8":{"M": 44.097, "d": 4.30, "cap": 0.85},
}

MECHS = ["Blocked", "Sieving", "Knudsen", "Surface", "Capillary", "Solution"]
MCOLOR = {
    "Blocked":  "#bdbdbd",
    "Sieving":  "#1f78b4",
    "Knudsen":  "#33a02c",
    "Surface":  "#e31a1c",
    "Capillary":"#ff7f00",
    "Solution": "#6a3d9a",
}

# ---------------- Helpers (slider + number box sync) ----------------
def slider_with_box(label, minv, maxv, default, step, key):
    ss = st.session_state
    vkey = f"{key}_val"
    skey = f"{key}_slider"
    bkey = f"{key}_box"
    if vkey not in ss:
        ss[vkey] = float(default)

    def _from_slider():
        ss[vkey] = float(ss[skey]); ss[bkey] = float(ss[skey])

    def _from_box():
        ss[vkey] = float(ss[bkey]); ss[skey] = float(ss[bkey])

    # 예전 형식: 비율 [4,1]
    col1, col2 = st.columns([4, 1])
    col1.slider(label, float(minv), float(maxv), float(ss[vkey]),
                step=float(step), key=skey, on_change=_from_slider)
    col2.number_input(" ", float(minv), float(maxv), float(ss[vkey]),
                      step=float(step), key=bkey, label_visibility="collapsed",
                      on_change=_from_box)
    return float(ss[vkey])

# ---------------- Models ----------------
def dsl_loading_series(T, P_bar, relP, Q1_kJ, Q2_kJ, q1, q2, b0=1e-4):
    P0 = P_bar * 1e5
    Q1, Q2 = max(Q1_kJ, 0.0)*1e3, max(Q2_kJ, 0.0)*1e3
    b1 = b0 * np.exp(Q1/(R*T))
    b2 = b0 * np.exp(Q2/(R*T))
    P = np.clip(relP, 1e-9, 1.0) * P0
    return q1 * (b1*P)/(1.0 + b1*P) + q2 * (b2*P)/(1.0 + b2*P)

def proxy_knudsen(d_nm, M): return (d_nm/2.0) / np.sqrt(M)

def proxy_sieving(d_nm, d_ang, T):
    p_eff = d_nm*10.0 + FLEX_A
    x = 1.0 - (d_ang/p_eff)**2
    return np.clip(x, 0.0, None)**2 * np.exp(-9.0e3/(R*T))

def proxy_surface(d_nm, T, loading):
    return np.exp(-E_D_SRF/(R*T)) * loading / np.maximum(d_nm, 1e-9)

def proxy_capillary(d_nm, T, relP, cap_scale):
    r = max(d_nm/2.0, 1e-9)
    relP_th = np.exp(-120.0/(r*T))
    return cap_scale*np.sqrt(r) * np.clip(relP - relP_th, 0.0, None)

def proxy_solution(T, loading, M):
    return np.exp(-E_D_SOL/(R*T))/np.sqrt(M) * np.maximum(loading, 0.0)

def proxies_all_for_gas(gas, other, T, P_bar, d_nm, relP, load_self, load_other):
    M = GASES[gas]["M"]; d = GASES[gas]["d"]; cap = GASES[gas]["cap"]
    siev = proxy_sieving(d_nm, d, T) * np.ones_like(relP)
    knud = proxy_knudsen(d_nm, M) * np.ones_like(relP)
    surf = proxy_surface(d_nm, T, load_self)
    capp = proxy_capillary(d_nm, T, relP, cap)
    solu = proxy_solution(T, load_self, M)

    dmin = min(GASES[gas]["d"], GASES[other]["d"])
    blocked = (1.0 if (d_nm*10.0 + FLEX_A) <= dmin else 0.0) * np.ones_like(relP)

    mask = 1.0 - blocked
    return {
        "Blocked": blocked,
        "Sieving": siev * mask,
        "Knudsen": knud * mask,
        "Surface": surf * mask,
        "Capillary": capp * mask,
        "Solution": solu * mask,
    }

def pick_mechanism_from_proxies(prox):
    arr = np.vstack([prox[m] for m in MECHS])
    idx = np.argmax(arr, axis=0)
    return np.array(MECHS)[idx]

def permeance_from_proxies(prox, load_a, load_b):
    arr = np.vstack([prox[m] for m in MECHS])
    maxp = arr.max(axis=0)
    y = load_a / (load_a + load_b + 1e-30)
    return (maxp * y) / THICK_M

# ---------------- UI ----------------
st.set_page_config(page_title="Membrane mechanisms", layout="wide")
st.title("Membrane Transport Mechanisms – robust demo")

# Global controls (예전 레이아웃)
T    = slider_with_box("Temperature (K)",       10.0, 600.0, 300.0, 1.0,  "T")
Pbar = slider_with_box("Total pressure (bar)",   0.1,  10.0,   1.0, 0.1,  "P")
d_nm = slider_with_box("Pore diameter (nm)",    0.01,  50.0,  0.34, 0.01, "d")

gases = list(GASES.keys())
gas1 = st.selectbox("Gas 1 (numerator)", gases, index=gases.index("C3H6"))
gas2 = st.selectbox("Gas 2 (denominator)", gases, index=gases.index("C3H8"))

st.subheader("Double-site Langmuir — Gas 1")
Q11 = slider_with_box("Qst1 (kJ/mol)", 0.0, 100.0, 27.0, 0.1, "Q11")
Q12 = slider_with_box("Qst2 (kJ/mol)", 0.0, 100.0, 18.0, 0.1, "Q12")
q11 = slider_with_box("q1 (mmol/g)",   0.0,   5.0,  0.70, 0.01, "q11")
q12 = slider_with_box("q2 (mmol/g)",   0.0,   5.0,  0.30, 0.01, "q12")

st.subheader("Double-site Langmuir — Gas 2")
Q21 = slider_with_box("Qst1 (kJ/mol)", 0.0, 100.0, 26.5, 0.1, "Q21")
Q22 = slider_with_box("Qst2 (kJ/mol)", 0.0, 100.0, 17.0, 0.1, "Q22")
q21 = slider_with_box("q1 (mmol/g)",   0.0,   5.0,  0.70, 0.01, "q21")
q22 = slider_with_box("q2 (mmol/g)",   0.0,   5.0,  0.30, 0.01, "q22")

# Data
relP  = np.linspace(0.01, 0.99, 300)
load1 = dsl_loading_series(T, Pbar, relP, Q11, Q12, q11, q12)
load2 = dsl_loading_series(T, Pbar, relP, Q21, Q22, q21, q22)
prox1 = proxies_all_for_gas(gas1, gas2, T, Pbar, d_nm, relP, load1, load2)
prox2 = proxies_all_for_gas(gas2, gas1, T, Pbar, d_nm, relP, load2, load1)
mechs = pick_mechanism_from_proxies(prox1)

# -------- Mechanism band (예전 크기) --------
rgba = np.array([to_rgba(MCOLOR[m]) for m in mechs])[None, :, :]
figB, axB = plt.subplots(figsize=(8, 1.4))
axB.imshow(rgba, extent=(0, 1, 0, 1), aspect="auto", origin="lower")
axB.set_yticks([]); axB.set_xlim(0, 1)
axB.set_xlabel("Relative pressure (P/P0)")
axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
st.pyplot(figB)
plt.close(figB)

# -------- Legend only --------
handles = [plt.Rectangle((0, 0), 1, 1, fc=MCOLOR[m], ec="none", label=m) for m in MECHS]
figL, axL = plt.subplots(figsize=(8, 1.2))
axL.axis("off")
figL.legend(handles=handles, loc="center", ncol=6, frameon=True)
st.pyplot(figL)
plt.close(figL)

# -------- Permeance (예전 크기) --------
perm1 = permeance_from_proxies(prox1, load1, load2)
perm2 = permeance_from_proxies(prox2, load2, load1)
fig1, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(relP, perm1, label=f"Permeance {gas1}")
ax1.plot(relP, perm2, "--", label=f"Permeance {gas2}")
ax1.set_ylabel("Permeance (arb. units)")
ax1.set_xlabel("Relative pressure (P/P0)")
ax1.grid(True); ax1.legend()
st.pyplot(fig1)
plt.close(fig1)

# -------- Selectivity (예전 크기) --------
sel = np.where(perm2 > 0, perm1/perm2, 0.0)
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(relP, sel, label=f"Selectivity {gas1}/{gas2}")
ax2.set_ylabel("Selectivity (-)")
ax2.set_xlabel("Relative pressure (P/P0)")
ax2.grid(True); ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

# -------- Summary --------
mid = len(relP)//2
summary = {m: prox1[m][mid] for m in MECHS}
best_m = max(summary, key=summary.get)
st.write(f"Dominant mechanism near P/P0 = {relP[mid]:.2f}: {best_m}")
