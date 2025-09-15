# app.py — Mechanism competition without +/- buttons
# realistic, normalized proxies so every mechanism can appear

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import streamlit as st

# -------------------- constants & data --------------------
R = 8.314  # J/mol/K

# (분자량, 운동지름 Å)
GASES = {
    "H2":  (2.016, 2.89),
    "He":  (4.003, 2.60),
    "N2":  (28.0134, 3.64),
    "O2":  (31.998, 3.46),
    "CO2": (44.01, 3.30),
    "CH4": (16.043, 3.80),
    "C2H6":(30.070, 4.44),
    "C3H6":(42.081, 4.68),
    "C3H8":(44.097, 4.65),
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

# Kelvin-like capillary onset 상수(완곡): ln(P/P0) = -A/(r*T)
KELVIN_A = 35.0  # 단위 맞춤용 상수(경향만 반영)

# 난류 방지용 작은 수
EPS = 1e-30

# -------------------- simple 2-way synced control (slider + number) --------------------
def synced_control(label, minv, maxv, val, step, key, fmt="float", host=None):
    if host is None:
        host = st
    ss = st.session_state
    vkey = f"{key}__v"
    if vkey not in ss:
        ss[vkey] = float(val)

    # 한 줄에 슬라이더 | 숫자 입력칸
    c1, c2 = host.columns([7, 3])
    with c1:
        sval = st.slider(label, float(minv), float(maxv), float(ss[vkey]),
                         step=float(step), key=f"{key}__s")
    with c2:
        if fmt == "int":
            nval = st.number_input(" ", min_value=float(minv), max_value=float(maxv),
                                   value=float(ss[vkey]), step=float(step),
                                   key=f"{key}__n", label_visibility="collapsed",
                                   format="%.0f")
        else:
            nval = st.number_input(" ", min_value=float(minv), max_value=float(maxv),
                                   value=float(ss[vkey]), step=float(step),
                                   key=f"{key}__n", label_visibility="collapsed")

    # 최근 입력 우선으로 동기화
    if nval != ss[vkey]:
        ss[vkey] = float(nval)
    elif sval != ss[vkey]:
        ss[vkey] = float(sval)
    return float(ss[vkey])

# -------------------- Double-site Langmuir --------------------
def dsl_loading(T, P_bar, relP, Q1_kJ, Q2_kJ, q1, q2, b0=3e-5):
    P0 = P_bar * 1e5
    Q1 = max(Q1_kJ, 0.0) * 1e3
    Q2 = max(Q2_kJ, 0.0) * 1e3
    b1 = b0 * np.exp(Q1/(R*T))
    b2 = b0 * np.exp(Q2/(R*T))
    P = np.clip(relP, 1e-9, 1.0) * P0
    return q1*(b1*P)/(1+b1*P) + q2*(b2*P)/(1+b2*P)  # mmol/g

# -------------------- proxy building blocks --------------------
def smooth_sigmoid(x, k=10.0):
    # σ(x) = 1/(1+exp(-k x)), k 클수록 계단에 가까움
    return 1.0/(1.0 + np.exp(-k*x))

def between_weight(x, a, b, sharp=4.0):
    # a<b, 구간 [a,b] 내부에서 크게(≈1), 외부로 나가면 0으로
    # σ(x-a)*σ(b-x)
    return smooth_sigmoid((x-a), k=sharp) * smooth_sigmoid((b-x), k=sharp)

def capillary_onset_relP(d_nm, T):
    r_nm = max(d_nm/2.0, 1e-6)
    ln_p = -KELVIN_A/(r_nm*T)         # Kelvin-like
    p = np.exp(np.clip(ln_p, -30, 0)) # 0<p<1
    return float(p)

# -------------------- realistic, normalized proxies --------------------
def mechanism_proxies_for_gas(gas, other, T, P_bar, d_nm, relP, loading_self, loading_other):
    """
    각 메커니즘의 '세기'를 0~1 스케일로 정규화해서 반환.
    - Blocked: p_eff가 두 분자 중 가장 작은 지름보다 작을수록 큼
    - Sieving: p_eff가 두 분자 지름 사이에 있을수록 큼 (부드럽게)
    - Knudsen: r/√M × (1 - 기공이 가득찼을수록 감소) × (낮은 압력에서 유리)
    - Surface: loading_self가 클수록, 기공 작을수록, 낮은 T일수록 유리
    - Capillary: Kelvin-onset 넘어간 relP에서 강함 (좁은 기공·낮은T에서 onset 낮아짐)
    - Solution: loading_self × exp(-Ea/(RT))/√M (Dense 유사)
    """
    relP = np.asarray(relP)
    M, d_ang = GASES[gas]
    _, d_o   = GASES[other]
    d_small, d_large = sorted([d_ang, d_o])
    p_eff = d_nm*10.0  # Å

    # Blocked
    blocked = smooth_sigmoid(d_small - p_eff, k=3.0)

    # Sieving (부드러운 창 함수)
    sieve = between_weight(p_eff, d_small, d_large, sharp=2.5)

    # Knudsen
    r = max(d_nm/2.0, 1e-6)
    porosity_factor = smooth_sigmoid(0.5 - loading_self/10.0, k=3.0)  # loading↑ → 자유공간↓
    lowP_factor = smooth_sigmoid(0.4 - relP, k=8.0)                   # 저압에서 우세
    knud = (r/np.sqrt(M)) * porosity_factor * lowP_factor

    # Surface (흡착↑, 작은 기공, 낮은 T)
    surf = (loading_self/10.0) * smooth_sigmoid(0.8 - d_nm, k=5.0) * smooth_sigmoid(300.0 - T, k=0.02)

    # Capillary (onset 이상에서 (relP-onset)/(1-onset))
    p_on = capillary_onset_relP(d_nm, T)
    cap = np.clip((relP - p_on)/(1.0 - p_on + EPS), 0.0, 1.0)

    # Solution (dense 유사: 흡착↑, 분자 작음, T↑ 유리)
    Ea = 12e3  # J/mol
    solu = (loading_self/10.0) * np.exp(-Ea/(R*T)) * (1.0/np.sqrt(M))
    # T↑에서 exp(-Ea/RT)↑ (완만), 분자 작을수록 유리

    # 0~1 정규화 (이 가스, 이 relP 벡터에서의 상대 경쟁)
    raw = {
        "Blocked":  blocked*np.ones_like(relP),
        "Sieving":  sieve*np.ones_like(relP),
        "Knudsen":  knud,
        "Surface":  surf*np.ones_like(relP),
        "Capillary":cap,
        "Solution": solu*np.ones_like(relP),
    }
    # 각 proxy를 자기 최대값으로 나눠 0~1로 스케일
    for k in raw:
        m = np.max(raw[k])
        if m > 0:
            raw[k] = raw[k]/m
        else:
            raw[k] = raw[k]*0.0
    return raw

def pick_mechanism(prox_dict):
    arr = np.vstack([prox_dict[m] for m in MECHS])
    idx = np.argmax(arr, axis=0)
    return np.array(MECHS)[idx]

# permeance = 정규화된 proxy들의 가중합 (질량전달 모델을 간단히 합성)
def permeance_from_proxies(prox, loading_self, loading_other):
    w = {
        "Blocked":  0.0,
        "Sieving":  0.9,
        "Knudsen":  1.0,
        "Surface":  0.7,
        "Capillary":1.2,
        "Solution": 0.5,
    }
    total = np.zeros_like(loading_self, dtype=float)
    for k in MECHS:
        total += w[k]*prox[k]
    # 흡착된 분율(loading_self/(self+other))로 가볍게 조절
    y = loading_self/(loading_self + loading_other + EPS)
    return total*(0.2 + 0.8*y)  # arbitrary scale

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Membrane mechanisms (balanced proxies)", layout="wide")

st.title("Membrane Transport Mechanisms — balanced & normalized proxies")

left, right = st.columns([6.5, 7], gap="large")

with left:
    st.subheader("Global")
    gases = list(GASES.keys())
    gas1 = st.selectbox("Gas 1 (numerator)", gases, index=gases.index("H2"))
    gas2 = st.selectbox("Gas 2 (denominator)", gases, index=gases.index("CH4"))

    T    = synced_control("Temperature (K)",       10.0, 600.0, 300.0, 1.0,  "T",  host=left)
    Pbar = synced_control("Total pressure (bar)",   0.1,  10.0,   1.0, 0.1,  "P",  host=left)
    d_nm = synced_control("Pore diameter (nm)",    0.10,  10.0,  0.60, 0.01, "d",  host=left)

    st.subheader("Double-site Langmuir — Gas 1")
    Q11 = synced_control("Qst1 (kJ/mol) [Gas 1]",  0.0, 100.0, 20.0, 0.5, "Q11", host=left)
    Q12 = synced_control("Qst2 (kJ/mol) [Gas 1]",  0.0, 100.0, 10.0, 0.5, "Q12", host=left)
    q11 = synced_control("q1 (mmol/g) [Gas 1]",    0.0,  10.0,  1.0,  0.05,"q11", host=left)
    q12 = synced_control("q2 (mmol/g) [Gas 1]",    0.0,  10.0,  0.5,  0.05,"q12", host=left)

    st.subheader("Double-site Langmuir — Gas 2")
    Q21 = synced_control("Qst1 (kJ/mol) [Gas 2]",  0.0, 100.0, 18.0, 0.5, "Q21", host=left)
    Q22 = synced_control("Qst2 (kJ/mol) [Gas 2]",  0.0, 100.0,  8.0, 0.5, "Q22", host=left)
    q21 = synced_control("q1 (mmol/g) [Gas 2]",    0.0,  10.0,  1.0,  0.05,"q21", host=left)
    q22 = synced_control("q2 (mmol/g) [Gas 2]",    0.0,  10.0,  0.5,  0.05,"q22", host=left)

# -------------------- simulation --------------------
relP = np.linspace(0.01, 0.99, 400)

load1 = dsl_loading(T, Pbar, relP, Q11, Q12, q11, q12)  # mmol/g
load2 = dsl_loading(T, Pbar, relP, Q21, Q22, q21, q22)

prox1 = mechanism_proxies_for_gas(gas1, gas2, T, Pbar, d_nm, relP, load1, load2)
prox2 = mechanism_proxies_for_gas(gas2, gas1, T, Pbar, d_nm, relP, load2, load1)

mechs_g1 = pick_mechanism(prox1)

perm1 = permeance_from_proxies(prox1, load1, load2)
perm2 = permeance_from_proxies(prox2, load2, load1)
sel   = np.where(perm2 > 0, perm1/perm2, 0.0)

# -------------------- plots --------------------
with right:
    # mechanism band
    rgba = np.array([to_rgba(MCOLOR[m]) for m in mechs_g1])[None, :, :]
    figB, axB = plt.subplots(figsize=(9.0, 1.6))
    axB.imshow(rgba, extent=(0, 1, 0, 1), origin="lower", aspect="auto")
    axB.set_yticks([]); axB.set_xlim(0, 1)
    axB.set_xlabel("Relative pressure (P/P0)")
    axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    st.pyplot(figB); plt.close(figB)

    # legend
    handles = [plt.Rectangle((0,0), 1,1, fc=MCOLOR[m], ec="none", label=m) for m in MECHS]
    figL, axL = plt.subplots(figsize=(9.0, 1.2))
    axL.axis("off")
    figL.legend(handles=handles, ncol=6, loc="center", frameon=True)
    st.pyplot(figL); plt.close(figL)

    # permeance
    fig1, ax1 = plt.subplots(figsize=(9.0, 3.2))
    ax1.plot(relP, perm1, label=f"Permeance {gas1}")
    ax1.plot(relP, perm2, "--", label=f"Permeance {gas2}")
    ax1.set_xlabel("Relative pressure (P/P0)")
    ax1.set_ylabel("Permeance (arb. units)")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1); plt.close(fig1)

    # selectivity
    fig2, ax2 = plt.subplots(figsize=(9.0, 3.2))
    ax2.plot(relP, sel, label=f"Selectivity {gas1}/{gas2}")
    ax2.set_xlabel("Relative pressure (P/P0)")
    ax2.set_ylabel("Selectivity (-)")
    ax2.grid(True); ax2.legend()
    st.pyplot(fig2); plt.close(fig2)

    mid = len(relP)//2
    st.caption(f"Dominant mechanism near P/P0={relP[mid]:.2f}: **{mechs_g1[mid]}**")
