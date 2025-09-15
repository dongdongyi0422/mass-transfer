# app.py — Membrane mechanism simulator (sieving boosted, q up to 10 mmol/g)
# 좌(넓은 컨트롤) / 우(결과 플롯) 2단 레이아웃
# 모든 슬라이더는 숫자 입력과 동기화되고 가로 폭을 넓게 사용함

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend (Streamlit Cloud)
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import streamlit as st

# ---------------- Constants (tunable) ----------------
R = 8.314

# ↓↓↓ Sieving이 더 잘 보이도록 조정 ↓↓↓
FLEX_A = 0.2        # [Å] 유효기공(p_eff = d_nm*10 + FLEX_A)에서 유연성 최소화 (기존 0.8 → 0.2)
SIEV_A = 2.5        # [-] sieving 가중치 (기존 1.0 → 2.5로 상향)
CAP_A  = 0.45       # [-] capillary 가중치 (기존 1.0 → 0.45로 하향)
RELPA_CONST = -150  # Kelvin 식 유사 임계 계수 (기존 -120 → -150로 더 보수적)
# ↑↑↑ ------------------------------------------------- ↑↑↑

THICK_M = 100e-9    # m (임의 스케일; permeance 산정용)
E_D_SOL = 1.8e4     # J/mol, solution-diffusion 확산 활성화
E_D_SRF = 9.0e3     # J/mol, 표면확산 활성화

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

# ---------------- Helpers ----------------
def slider_with_box(label, minv, maxv, default, step, key, host=None):
    """
    Render a wide slider + number box, synchronized both ways.
    """
    if host is None:
        host = st
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

    c1, c2 = host.columns([8, 2])
    c1.slider(label, float(minv), float(maxv), float(ss[vkey]),
              step=float(step), key=skey, on_change=_from_slider,
              label_visibility="visible")
    c1.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    c2.number_input(" ", float(minv), float(maxv), float(ss[vkey]),
                    step=float(step), key=bkey, label_visibility="collapsed",
                    on_change=_from_box)
    return float(ss[vkey])

# ---------------- Mechanistic proxy models ----------------
def dsl_loading_series(T, P_bar, relP, Q1_kJ, Q2_kJ, q1, q2, b0=1e-4):
    """Double-site Langmuir mixture loading vs relative pressure."""
    P0 = P_bar * 1e5
    Q1 = max(Q1_kJ, 0.0) * 1e3
    Q2 = max(Q2_kJ, 0.0) * 1e3
    b1 = b0 * np.exp(Q1/(R*T))
    b2 = b0 * np.exp(Q2/(R*T))
    P  = np.clip(relP, 1e-9, 1.0) * P0
    return q1 * (b1*P)/(1.0 + b1*P) + q2 * (b2*P)/(1.0 + b2*P)

def proxy_knudsen(d_nm, M):
    return (d_nm/2.0) / np.sqrt(M)

def proxy_sieving(d_nm, d_ang, T):
    p_eff = d_nm*10.0 + FLEX_A  # Å
    x = 1.0 - (d_ang/p_eff)**2
    return SIEV_A * np.clip(x, 0.0, None)**2 * np.exp(-9.0e3/(R*T))

def proxy_surface(d_nm, T, loading):
    return np.exp(-E_D_SRF/(R*T)) * loading / np.maximum(d_nm, 1e-9)

def proxy_capillary(d_nm, T, relP, cap_scale):
    r = max(d_nm/2.0, 1e-9)  # nm
    # 더 보수적인 임계 (capillary가 쉬이 켜지지 않도록)
    relP_th = np.exp(RELPA_CONST/(r*T))
    return CAP_A * cap_scale*np.sqrt(r) * np.clip(relP - relP_th, 0.0, None)

def proxy_solution(T, loading, M):
    return np.exp(-E_D_SOL/(R*T)) / np.sqrt(M) * np.maximum(loading, 0.0)

def proxies_all_for_gas(gas, other, T, P_bar, d_nm, relP, load_self, load_other):
    """각 메커니즘 proxy (같은 스케일, 상대 비교용)"""
    M   = GASES[gas]["M"]
    d   = GASES[gas]["d"]
    cap = GASES[gas]["cap"]
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
    arr  = np.vstack([prox[m] for m in MECHS])
    maxp = arr.max(axis=0)
    y    = load_a / (load_a + load_b + 1e-30)
    return (maxp * y) / THICK_M

# ---------------- UI ----------------
st.set_page_config(page_title="Membrane mechanisms", layout="wide")
st.title("Membrane Transport Mechanisms – sieving-boosted demo")

# 좌/우: 왼쪽 컨트롤 넓게
left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("Global")
    T    = slider_with_box("Temperature (K)",       10.0, 600.0, 300.0, 1.0,  "T",    host=left)
    Pbar = slider_with_box("Total pressure (bar)",   0.1,  10.0,   1.0, 0.1,  "P",    host=left)
    d_nm = slider_with_box("Pore diameter (nm)",    0.01,  50.0,  0.34, 0.01, "d",    host=left)

    gases = list(GASES.keys())
    gas1 = st.selectbox("Gas 1 (numerator)", gases, index=gases.index("H2"))
    gas2 = st.selectbox("Gas 2 (denominator)", gases, index=gases.index("CH4"))

    st.subheader("Double-site Langmuir — Gas 1")
    # q의 최대를 10 mmol/g로 상향
    Q11 = slider_with_box("Qst1 (kJ/mol)", 0.0, 100.0, 12.0, 0.1, "Q11", host=left)
    Q12 = slider_with_box("Qst2 (kJ/mol)", 0.0, 100.0, 10.0, 0.1, "Q12", host=left)
    q11 = slider_with_box("q1 (mmol/g)",   0.0,  10.0,  0.20, 0.01, "q11", host=left)
    q12 = slider_with_box("q2 (mmol/g)",   0.0,  10.0,  0.10, 0.01, "q12", host=left)

    st.subheader("Double-site Langmuir — Gas 2")
    Q21 = slider_with_box("Qst1 (kJ/mol)", 0.0, 100.0, 12.0, 0.1, "Q21", host=left)
    Q22 = slider_with_box("Qst2 (kJ/mol)", 0.0, 100.0, 10.0, 0.1, "Q22", host=left)
    q21 = slider_with_box("q1 (mmol/g)",   0.0,  10.0,  0.20, 0.01, "q21", host=left)
    q22 = slider_with_box("q2 (mmol/g)",   0.0,  10.0,  0.10, 0.01, "q22", host=left)

# ---- Simulation (single pass) ----
relP  = np.linspace(0.01, 0.99, 300)
load1 = dsl_loading_series(T, Pbar, relP, Q11, Q12, q11, q12)
load2 = dsl_loading_series(T, Pbar, relP, Q21, Q22, q21, q22)
prox1 = proxies_all_for_gas(gas1, gas2, T, Pbar, d_nm, relP, load1, load2)
prox2 = proxies_all_for_gas(gas2, gas1, T, Pbar, d_nm, relP, load2, load1)
mechs = pick_mechanism_from_proxies(prox1)

perm1 = permeance_from_proxies(prox1, load1, load2)
perm2 = permeance_from_proxies(prox2, load2, load1)
sel   = np.where(perm2 > 0, perm1/perm2, 0.0)

# ---- RIGHT: plots ----
with right:
    # Mechanism band
    rgba = np.array([to_rgba(MCOLOR[m]) for m in mechs])[None, :, :]
    figB, axB = plt.subplots(figsize=(9, 1.6))
    axB.imshow(rgba, extent=(0, 1, 0, 1), aspect="auto", origin="lower")
    axB.set_yticks([]); axB.set_xlim(0, 1)
    axB.set_xlabel("Relative pressure (P/P0)")
    axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    st.pyplot(figB); plt.close(figB)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MCOLOR[m], ec="none", label=m) for m in MECHS]
    figL, axL = plt.subplots(figsize=(9, 1.2))
    axL.axis("off")
    figL.legend(handles=handles, loc="center", ncol=6, frameon=True)
    st.pyplot(figL); plt.close(figL)

    # Permeance
    fig1, ax1 = plt.subplots(figsize=(9, 3.2))
    ax1.plot(relP, perm1, label=f"Permeance {gas1}")
    ax1.plot(relP, perm2, "--", label=f"Permeance {gas2}")
    ax1.set_ylabel("Permeance (arb. units)")
    ax1.set_xlabel("Relative pressure (P/P0)")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1); plt.close(fig1)

    # Selectivity
    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    ax2.plot(relP, sel, label=f"Selectivity {gas1}/{gas2}")
    ax2.set_ylabel("Selectivity (-)")
    ax2.set_xlabel("Relative pressure (P/P0)")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2); plt.close(fig2)

    # Summary text
    mid = len(relP)//2
    summary = {m: prox1[m][mid] for m in MECHS}
    best_m = max(summary, key=summary.get)
    st.write(f"Dominant mechanism near P/P0 = {relP[mid]:.2f}: **{best_m}**")
